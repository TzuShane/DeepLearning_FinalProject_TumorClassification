"""
評估：分類報告、混淆矩陣、各類別 one-vs-rest ROC-AUC（PyTorch）。

用法：
    python evaluate.py --backbone resnet50

醫療分類不能只看整體 accuracy，重點看各類別的 recall（漏診率）。
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import config
from data import load_datasets, make_loaders
from models import build_model
from train import get_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    """跑過整個資料集，收集真實標籤與 softmax 機率。"""
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        probs = F.softmax(model(x), dim=1).cpu().numpy()
        y_prob.append(probs)
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def evaluate(backbone_name):
    device = get_device()
    train_ds, val_ds, test_ds, class_names, _ = load_datasets()
    num_classes = len(class_names)
    _, _, test_loader = make_loaders(
        train_ds, val_ds, test_ds, use_cuda=(device.type == "cuda"))

    model = build_model(backbone_name, num_classes).to(device)
    ckpt_path = config.OUTPUT_DIR / f"{backbone_name}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_true, y_prob = collect_predictions(model, test_loader, device)
    y_pred = y_prob.argmax(axis=1)

    # ---- 1. 分類報告（precision / recall / F1，逐類別）----
    print("\n===== Classification Report =====")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(config.OUTPUT_DIR / f"{backbone_name}_report.txt", "w") as f:
        f.write(report)

    # ---- 2. 混淆矩陣 ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {backbone_name}")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / f"{backbone_name}_confusion.png", dpi=150)
    plt.close()

    # ---- 3. one-vs-rest ROC-AUC ----
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    print(f"\nMacro-average ROC-AUC (OvR): {macro_auc:.4f}")

    plt.figure(figsize=(6, 5))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_i:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (OvR) - {backbone_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / f"{backbone_name}_roc.png", dpi=150)
    plt.close()

    print(f"圖表已存於 {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50", choices=config.AVAILABLE_BACKBONES)
    args = parser.parse_args()
    evaluate(args.backbone)
