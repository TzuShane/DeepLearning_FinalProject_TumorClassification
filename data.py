"""
資料載入與擴增（PyTorch / torchvision）。

- 從 Training 資料夾切出 train / val（80/20）
- Testing 資料夾整個當測試集（訓練過程完全不碰）
- 資料擴增只作用在訓練集，且限定「醫學上合理」的幾何變換
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config


def _build_transforms():
    """回傳 (train_tf, eval_tf)。"""
    normalize = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)

    # 醫學影像：只做不改變診斷語意的幾何變換
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),              # 腦部左右翻轉合理
        transforms.RandomRotation(18),                  # 小角度旋轉
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # 輕微縮放
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


class _TransformedSubset(Dataset):
    """
    包住 random_split 出來的 Subset，讓 train / val 能套用不同 transform
    （原始 ImageFolder 不帶 transform，回傳 PIL 影像）。
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]   # img 是 PIL
        return self.transform(img), label


def load_datasets():
    """回傳 (train_ds, val_ds, test_ds, class_names)。"""
    train_tf, eval_tf = _build_transforms()

    # 不帶 transform 的完整訓練集（回傳 PIL），再切 train / val
    full_train = ImageFolder(config.TRAIN_DIR)
    class_names = full_train.classes    # 依字母排序
    print(f"類別（依序）: {class_names}")

    n_val = int(len(full_train) * config.VAL_SPLIT)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(config.SEED)
    train_sub, val_sub = random_split(full_train, [n_train, n_val], generator=generator)

    train_ds = _TransformedSubset(train_sub, train_tf)
    val_ds = _TransformedSubset(val_sub, eval_tf)
    test_ds = ImageFolder(config.TEST_DIR, transform=eval_tf)

    # 供 class weight 計算時用（只看 train 部分的標籤，較嚴謹）
    train_targets = [full_train.targets[i] for i in train_sub.indices]

    return train_ds, val_ds, test_ds, class_names, train_targets


def make_loaders(train_ds, val_ds, test_ds, use_cuda):
    common = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_cuda,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def compute_class_weights(train_targets, num_classes):
    """
    類別權重（處理資料不平衡）。四類雖大致平衡，仍計算並用於 loss，
    是報告可著墨的「不平衡處理」技術點。回傳 torch.Tensor。
    """
    counts = np.bincount(train_targets, minlength=num_classes).astype(float)
    total = counts.sum()
    weights = total / (num_classes * counts)   # sklearn 'balanced' 公式
    print(f"各類別樣本數: {counts.astype(int).tolist()}")
    print(f"類別權重: {weights.round(3).tolist()}")
    return torch.tensor(weights, dtype=torch.float32)
