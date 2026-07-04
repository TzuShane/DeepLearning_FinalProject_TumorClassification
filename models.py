"""
模型建構（PyTorch / torchvision）。

- baseline：自己刻的小型 CNN（從頭訓練，當對照組）
- 遷移學習：ResNet50 / DenseNet121 / EfficientNetB0（ImageNet 預訓練）

統一封裝成 TumorClassifier：
    features -> 全域平均池化 -> Dropout -> Linear
這樣的結構讓「凍結/解凍 backbone」與「Grad-CAM 取最後卷積特徵圖」
都變得單純：features 就是特徵抽取器、也是 Grad-CAM 的目標層。
"""
import torch.nn as nn
from torchvision import models as tvm

import config


def _build_feature_extractor(name):
    """回傳 (features 模組, 輸出通道數)。features 輸出為 (N, C, 7, 7)。"""
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        features = nn.Sequential(*list(m.children())[:-2])   # 去掉 avgpool 與 fc
        in_features = m.fc.in_features                        # 2048
    elif name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
        features = nn.Sequential(m.features, nn.ReLU(inplace=True))
        in_features = m.classifier.in_features               # 1024
    elif name == "efficientnetb0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        features = m.features
        in_features = m.classifier[1].in_features            # 1280
    else:
        raise ValueError(f"未知 backbone: {name}")
    return features, in_features


def _build_baseline_features():
    """對照組：從頭訓練的簡單 CNN 特徵抽取器。"""
    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
    return features, 128


class TumorClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, dropout=config.DROPOUT):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "baseline":
            self.features, in_features = _build_baseline_features()
        else:
            self.features, in_features = _build_feature_extractor(backbone_name)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        f = self.features(x)
        p = self.pool(f).flatten(1)
        return self.head(p)

    @property
    def target_layer(self):
        """Grad-CAM 目標層：最後的卷積特徵抽取器。"""
        return self.features


# ---------------------------------------------------------------------------
# 凍結 / 解凍與 BatchNorm 控制
# ---------------------------------------------------------------------------
def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = True


def set_bn_eval(module):
    """
    把所有 BatchNorm 設為 eval 模式：微調時凍結其統計量，只更新卷積權重。
    對應 Keras 版的 base_model(x, training=False)，是小資料微調穩定的關鍵。
    """
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def build_model(backbone_name, num_classes):
    return TumorClassifier(backbone_name, num_classes)
