"""
compare.py —— 彙整各模型結果，產生架構比較表。

對每個訓練好的 backbone：
  - 測試集準確率 / macro-F1 / macro-AUC
  - 參數量
  - 推論速度（單張延遲 ms、批次吞吐量 img/s）
輸出：終端機表格 + outputs/comparison.md + outputs/comparison.csv

前置：先用 train.py 訓練各模型（baseline 也要用模組化流程，
才會和其他模型同尺寸、可公平比較）：
    python train.py --backbone baseline
    python train.py --backbone resnet50
    python train.py --backbone densenet121
    python train.py --backbone efficientnetb0

用法：
    python compare.py                       # 比較全部四個
    python compare.py --backbones resnet50 densenet121
"""
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config
from data import load_datasets, make_loaders
from models import build_model
from train import get_device


def _sync(device):
    """等待 GPU/MPS 運算完成，確保計時準確。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def test_metrics(model, loader, device, num_classes):
    """回傳 (accuracy, macro_f1, macro_auc)。"""
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        probs = F.softmax(model(x.to(device)), dim=1).cpu().numpy()
        y_prob.append(probs)
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        macro_auc = float("nan")   # 測試集若缺某類別可能算不出
    return acc, macro_f1, macro_auc


@torch.no_grad()
def measure_latency(model, device, warmup=10, runs=50):
    """單張影像延遲（ms/img），batch=1。"""
    model.eval()
    x = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    _sync(device)
    return (time.perf_counter() - t0) / runs * 1000.0


@torch.no_grad()
def measure_throughput(model, device, batch=None, warmup=5, runs=20):
    """批次吞吐量（img/s）。"""
    model.eval()
    batch = batch or config.BATCH_SIZE
    x = torch.randn(batch, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    _sync(device)
    elapsed = time.perf_counter() - t0
    return batch * runs / elapsed


def evaluate_one(backbone_name, test_loader, num_classes, device):
    """載入單一模型並量測所有指標。找不到權重回傳 None。"""
    ckpt_path = config.OUTPUT_DIR / f"{backbone_name}_best.pt"
    if not ckpt_path.exists():
        print(f"[跳過] 找不到 {ckpt_path}，請先訓練此模型。")
        return None

    model = build_model(backbone_name, num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    acc, macro_f1, macro_auc = test_metrics(model, test_loader, device, num_classes)
    row = {
        "model": backbone_name,
        "params_M": count_params(model) / 1e6,
        "test_acc": acc,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "latency_ms": measure_latency(model, device),
        "throughput_ips": measure_throughput(model, device),
    }
    print(f"[完成] {backbone_name}: acc={acc:.4f}, params={row['params_M']:.2f}M, "
          f"latency={row['latency_ms']:.2f}ms")
    return row


def print_table(rows):
    header = ["Model", "Params(M)", "Test Acc", "Macro F1", "Macro AUC",
              "Latency(ms)", "Throughput(img/s)"]
    widths = [16, 10, 9, 9, 10, 12, 18]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print("\n" + fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row([
            r["model"],
            f"{r['params_M']:.2f}",
            f"{r['test_acc']:.4f}",
            f"{r['macro_f1']:.4f}",
            f"{r['macro_auc']:.4f}",
            f"{r['latency_ms']:.2f}",
            f"{r['throughput_ips']:.1f}",
        ]))


def save_outputs(rows):
    # CSV
    csv_path = config.OUTPUT_DIR / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Markdown（可直接貼進報告）
    md_path = config.OUTPUT_DIR / "comparison.md"
    cols = ["Model", "Params (M)", "Test Acc", "Macro F1", "Macro AUC",
            "Latency (ms/img)", "Throughput (img/s)"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"],
            f"{r['params_M']:.2f}",
            f"{r['test_acc']:.4f}",
            f"{r['macro_f1']:.4f}",
            f"{r['macro_auc']:.4f}",
            f"{r['latency_ms']:.2f}",
            f"{r['throughput_ips']:.1f}",
        ]) + " |")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\n已存: {csv_path}\n已存: {md_path}")


def compare(backbones):
    device = get_device()
    print(f"使用裝置: {device}")

    train_ds, val_ds, test_ds, class_names, _ = load_datasets()
    num_classes = len(class_names)
    _, _, test_loader = make_loaders(
        train_ds, val_ds, test_ds, use_cuda=(device.type == "cuda"))

    rows = []
    for name in backbones:
        row = evaluate_one(name, test_loader, num_classes, device)
        if row is not None:
            rows.append(row)

    if not rows:
        print("沒有可比較的模型，請先訓練。")
        return

    # 依測試準確率由高到低排序
    rows.sort(key=lambda r: r["test_acc"], reverse=True)
    print_table(rows)
    save_outputs(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+",
                        default=config.AVAILABLE_BACKBONES,
                        choices=config.AVAILABLE_BACKBONES,
                        help="要比較的 backbone（預設全部）")
    args = parser.parse_args()
    compare(args.backbones)
