"""
訓練主程式：兩階段微調（PyTorch）。

用法：
    python train.py --backbone resnet50
    python train.py --backbone densenet121
    python train.py --backbone efficientnetb0
    python train.py --backbone baseline

架構比較（報告主體）就是把上面幾個都跑一遍，
每次都會存下最佳權重與訓練曲線，再用 evaluate.py 統一比較。
"""
import argparse
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from data import load_datasets, make_loaders, compute_class_weights
from models import build_model, freeze_backbone, unfreeze_backbone, set_bn_eval


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")          # RTX 4070 Super
    if torch.backends.mps.is_available():
        return torch.device("mps")           # MacBook M4 Pro
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_bn=False):
    """跑一個 epoch。傳入 optimizer 代表訓練，否則為驗證。回傳 (loss, acc)。"""
    is_train = optimizer is not None
    if is_train:
        model.train()
        if freeze_bn:                        # 微調階段：BN 維持推論模式
            set_bn_eval(model.features)
    else:
        model.eval()

    loss_sum, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(is_train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if is_train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / total, correct / total


def train_loop(model, train_loader, val_loader, criterion, device,
               epochs, lr, ckpt_path, freeze_bn, history, best_acc):
    """通用訓練迴圈，含 early stopping / 最佳權重保存 / LR 排程。"""
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-7)

    epochs_no_improve = 0
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer=optimizer, freeze_bn=freeze_bn)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"  epoch {epoch:2d} | train {tr_loss:.4f}/{tr_acc:.4f} "
              f"| val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_acc:                # 存最佳權重（依 val accuracy）
            best_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)

        if va_loss < best_val_loss - 1e-4:   # early stopping（依 val loss）
            best_val_loss = va_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
                print("  early stopping.")
                break
    return best_acc


def train(backbone_name):
    device = get_device()
    print(f"使用裝置: {device}")

    train_ds, val_ds, test_ds, class_names, train_targets = load_datasets()
    num_classes = len(class_names)
    train_loader, val_loader, _ = make_loaders(
        train_ds, val_ds, test_ds, use_cuda=(device.type == "cuda"))

    class_weights = compute_class_weights(train_targets, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(backbone_name, num_classes).to(device)
    ckpt_path = config.OUTPUT_DIR / f"{backbone_name}_best.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    # ---- 第一階段：凍結 backbone，只訓練分類頭（baseline 無預訓練，直接全部訓練）----
    is_transfer = backbone_name != "baseline"
    if is_transfer:
        freeze_backbone(model)
    print("\n===== Stage 1: 訓練分類頭 =====")
    best_acc = train_loop(model, train_loader, val_loader, criterion, device,
                          epochs=config.STAGE1_EPOCHS, lr=config.STAGE1_LR,
                          ckpt_path=ckpt_path, freeze_bn=False,
                          history=history, best_acc=best_acc)
    stage1_epochs_run = len(history["train_loss"])   # 供曲線圖標記微調起點

    # ---- 第二階段：解凍 backbone，整體微調 ----
    if is_transfer:
        print("\n===== Stage 2: 解凍 backbone 微調 =====")
        unfreeze_backbone(model)
        best_acc = train_loop(model, train_loader, val_loader, criterion, device,
                              epochs=config.STAGE2_EPOCHS, lr=config.STAGE2_LR,
                              ckpt_path=ckpt_path, freeze_bn=True,   # BN 凍結
                              history=history, best_acc=best_acc)

    with open(config.OUTPUT_DIR / f"{backbone_name}_history.json", "w") as f:
        json.dump({"history": history, "class_names": class_names,
                   "best_val_acc": best_acc,
                   "stage1_epochs": stage1_epochs_run}, f, indent=2)

    print(f"\n完成，最佳 val acc={best_acc:.4f}，模型存於: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50",
                        choices=config.AVAILABLE_BACKBONES, help="選擇 backbone")
    args = parser.parse_args()
    train(args.backbone)
