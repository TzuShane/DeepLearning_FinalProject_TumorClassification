"""
plot_curves.py —— 訓練曲線視覺化。

讀取 train.py 存下的 <backbone>_history.json，畫出：
  1. 每個模型的 loss / accuracy 曲線（train vs val），並標記微調起點
  2. 所有模型疊在一起的 val accuracy / val loss 比較圖

用法：
    python plot_curves.py                       # 全部模型：個別圖 + 疊圖
    python plot_curves.py --backbone resnet50   # 只畫單一模型
    python plot_curves.py --overlay-only        # 只畫疊圖比較
"""
import argparse
import json

import matplotlib.pyplot as plt

import config


def load_history(backbone_name):
    path = config.OUTPUT_DIR / f"{backbone_name}_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_single(backbone_name):
    """單一模型：左圖 loss、右圖 accuracy，含微調起點標記。"""
    data = load_history(backbone_name)
    if data is None:
        print(f"[跳過] 找不到 {backbone_name}_history.json")
        return

    h = data["history"]
    epochs = range(1, len(h["train_loss"]) + 1)
    stage1 = data.get("stage1_epochs")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- Loss ----
    axes[0].plot(epochs, h["train_loss"], label="train")
    axes[0].plot(epochs, h["val_loss"], label="val")
    axes[0].set_title(f"{backbone_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # ---- Accuracy ----
    axes[1].plot(epochs, h["train_acc"], label="train")
    axes[1].plot(epochs, h["val_acc"], label="val")
    axes[1].set_title(f"{backbone_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # 微調起點：第一階段之後才進入解凍微調（baseline 無此階段則不畫）
    if stage1 is not None and 0 < stage1 < len(h["train_loss"]):
        for ax in axes:
            ax.axvline(stage1 + 0.5, color="gray", linestyle="--", alpha=0.7)
            ax.text(stage1 + 0.6, ax.get_ylim()[0], " fine-tune",
                    color="gray", fontsize=9, va="bottom")

    plt.tight_layout()
    out = config.OUTPUT_DIR / f"{backbone_name}_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已存: {out}")


def plot_overlay(backbones):
    """所有模型疊圖：左 val loss、右 val accuracy。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = 0
    for name in backbones:
        data = load_history(name)
        if data is None:
            continue
        h = data["history"]
        epochs = range(1, len(h["val_loss"]) + 1)
        axes[0].plot(epochs, h["val_loss"], label=name)
        axes[1].plot(epochs, h["val_acc"], label=name)
        plotted += 1

    if plotted == 0:
        print("[跳過] 沒有可疊圖的歷史檔")
        plt.close()
        return

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    out = config.OUTPUT_DIR / "all_val_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已存: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=None,
                        choices=config.AVAILABLE_BACKBONES,
                        help="只畫單一模型（省略則畫全部）")
    parser.add_argument("--overlay-only", action="store_true",
                        help="只畫疊圖比較")
    args = parser.parse_args()

    if args.backbone:
        plot_single(args.backbone)
    else:
        if not args.overlay_only:
            for name in config.AVAILABLE_BACKBONES:
                plot_single(name)
        plot_overlay(config.AVAILABLE_BACKBONES)
