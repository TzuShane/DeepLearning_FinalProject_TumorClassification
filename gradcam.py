"""
Grad-CAM 可解釋性視覺化（PyTorch，用 hook 實作）。

原理：對「目標類別分數」相對於「最後一層卷積特徵圖」求梯度，
以梯度的全域平均當作各通道權重，加權組合特徵圖得到熱力圖，
再疊到原圖上，看模型判斷時關注哪個區域（理想上應落在腫瘤上）。

實作：用 forward hook 抓最後卷積特徵圖、full backward hook 抓其梯度。
目標層即 model.target_layer（見 models.py）。

用法：
    python gradcam.py --backbone resnet50 --image path/to/mri.jpg
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import config
from models import build_model
from train import get_device


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)  # 讓 backward hook 正常觸發
        logits = self.model(input_tensor)          # (1, num_classes)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # 各通道權重 = 梯度的全域平均；加權組合特徵圖
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(config.IMG_SIZE, config.IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def _load_image(img_path):
    """回傳 (模型輸入張量, 供顯示的原始 PIL 影像)。"""
    pil = Image.open(img_path).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    tensor = tf(pil).unsqueeze(0)
    return tensor, pil


def run_gradcam(backbone_name, img_path, class_names):
    device = get_device()
    model = build_model(backbone_name, len(class_names)).to(device)
    ckpt_path = config.OUTPUT_DIR / f"{backbone_name}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    cam_engine = GradCAM(model, model.target_layer)
    input_tensor, pil = _load_image(img_path)
    heatmap, pred_index = cam_engine(input_tensor.to(device))
    print(f"預測類別: {class_names[pred_index]}")

    # 熱力圖上色並疊圖
    jet = plt.colormaps["jet"](heatmap)[:, :, :3]        # (H, W, 3)
    original = np.asarray(pil, dtype=np.float32) / 255.0
    overlay = 0.4 * jet + 0.6 * original
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pil)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM: {class_names[pred_index]}")
    axes[1].axis("off")
    plt.tight_layout()
    out_path = config.OUTPUT_DIR / f"{backbone_name}_gradcam.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"已存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50", choices=config.AVAILABLE_BACKBONES)
    parser.add_argument("--image", required=True, help="單張 MRI 影像路徑")
    args = parser.parse_args()

    if args.backbone == "baseline":
        raise SystemExit("Grad-CAM 範例建議用遷移學習模型（如 resnet50）以取得清晰特徵圖。")

    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    run_gradcam(args.backbone, args.image, class_names)
