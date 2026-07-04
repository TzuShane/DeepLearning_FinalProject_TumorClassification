"""
全域設定：資料路徑、超參數、可選 backbone。
把 DATA_DIR 改成你電腦上解壓後的資料夾即可。
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 資料路徑
# ---------------------------------------------------------------------------
# Kaggle「Brain Tumor MRI Dataset」(Masoud Nickparvar) 解壓後的結構：
#   <DATA_DIR>/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
#   <DATA_DIR>/Testing /{glioma,meningioma,notumor,pituitary}/*.jpg
DATA_DIR = Path("/home/shane/Documents/DL_final/brain-tumor-mri-dataset")                 # 改成你的路徑
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

# 產出（模型權重、圖表）存放處
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 影像 / 訓練超參數
# ---------------------------------------------------------------------------
IMG_SIZE = 224            # 遷移學習 backbone 的標準輸入尺寸
BATCH_SIZE = 32           # 12GB 顯存跑 224 綽綽有餘，可調到 64
NUM_WORKERS = 4           # DataLoader 執行緒數（Mac 上若出問題可設 0）
SEED = 42
VAL_SPLIT = 0.2           # 從 Training 再切 20% 當驗證集

# 兩階段微調的 epoch 數與學習率
STAGE1_EPOCHS = 10        # 凍結 backbone，只訓練分類頭
STAGE2_EPOCHS = 15        # 解凍 backbone，整體微調
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5          # 微調用很小的學習率，避免破壞預訓練特徵
EARLY_STOP_PATIENCE = 5

DROPOUT = 0.3

# ImageNet 正規化參數（三個 torchvision backbone 共用）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# 可選 backbone（報告的架構比較就是換這裡跑三次）
# ---------------------------------------------------------------------------
AVAILABLE_BACKBONES = ["baseline", "resnet50", "densenet121", "efficientnetb0"]
