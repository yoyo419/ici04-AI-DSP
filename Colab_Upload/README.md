# 🌩️ LegalDelta Colab 訓練包

這個資料夾包含所有在 Google Colab 上訓練模型所需的檔案。

## 📦 資料夾內容

```
Colab_Upload/
├── data/                      # 訓練數據
│   ├── GRPO_training.json    # 訓練集
│   └── GRPO_dev.json         # 驗證集
├── src/                       # 源代碼
│   └── reward.py             # 獎勵函數
├── train_colab.py            # Colab 訓練腳本 ⭐
├── requirements.txt          # Python 依賴
├── COLAB_QUICKSTART.md       # 快速開始指南
└── README.md                 # 本檔案
```

## 🚀 使用步驟

### 1. 上傳到 Google Drive

選擇以下任一方式：

**方式 A：上傳整個資料夾**
- 直接將 `Colab_Upload` 資料夾拖曳到 Google Drive
- 建議路徑：`MyDrive/LegalDelta/`

**方式 B：上傳壓縮檔**
- 將 `Colab_Upload` 資料夾壓縮成 ZIP
- 上傳到 Google Drive: `MyDrive/LegalDelta/Colab_Upload.zip`

### 2. 在 Colab 執行

#### 2.1 開啟 Colab
- 前往 https://colab.research.google.com
- 新建 Notebook
- **重要**: Runtime → Change runtime type → GPU (T4)

#### 2.2 掛載 Drive

```python
from google.colab import drive
import os

# 掛載 Drive
drive.mount('/content/drive')

# 如果上傳的是資料夾，直接切換目錄
os.chdir('/content/drive/MyDrive/LegalDelta/Colab_Upload')

# 如果上傳的是 ZIP，先解壓縮
# import zipfile
# with zipfile.ZipFile('/content/drive/MyDrive/LegalDelta/Colab_Upload.zip', 'r') as zip_ref:
#     zip_ref.extractall('/content/LegalDelta')
# os.chdir('/content/LegalDelta/Colab_Upload')

print(f"✓ 當前目錄: {os.getcwd()}")
!ls -la
```

#### 2.3 安裝依賴

```python
!pip install -q accelerate peft datasets trl bitsandbytes sentencepiece protobuf

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

#### 2.4 開始訓練

```python
!python train_colab.py
```

### 3. 監控進度

**查看即時日誌：**
```python
!tail -f logs/*.log
```

**使用 TensorBoard：**
```python
%load_ext tensorboard
%tensorboard --logdir outputs
```

### 4. 下載結果

訓練完成後，結果會自動保存到：
```
/content/drive/MyDrive/LegalDelta_Results/qwen3B_colab_YYYYMMDD_HHMMSS/
├── lora_adapter/     # LoRA 權重 ⭐
└── logs/             # 訓練日誌
```

或手動下載：
```python
import shutil
from google.colab import files

# 打包 LoRA adapter
shutil.make_archive('lora_adapter', 'zip', './lora_adapter')

# 下載
files.download('lora_adapter.zip')
```

## ⏱️ 預期時間

- 模型下載: ~10 分鐘 (Qwen2.5-3B, 約 6GB)
- 訓練 (3 epochs): ~2-4 小時
- **總計: 約 2.5-4.5 小時**

## 📊 訓練配置

預設配置（在 `train_colab.py` 中）：

```python
model_size = "3B"              # 模型大小
num_train_epochs = 3           # 訓練輪數
per_device_batch_size = 1      # Batch size
gradient_accumulation_steps = 32  # 梯度累積
max_seq_length = 4096          # 最大序列長度
learning_rate = 5e-5           # 學習率
```

如需調整，編輯 `train_colab.py` 中的 `Config` 類別。

## 🔧 常見問題

### 記憶體不足 (OOM)

修改 `train_colab.py`:
```python
class Config:
    max_seq_length = 2048          # 減少序列長度
    gradient_accumulation_steps = 64  # 增加梯度累積
```

### Colab 斷線

重新執行訓練 cell，Trainer 會從最新的 checkpoint 繼續。

### 找不到檔案

確認當前目錄：
```python
import os
print(os.getcwd())
!ls -la
```

## 📞 更多資訊

詳細說明請參考：
- `COLAB_QUICKSTART.md` - 快速開始指南
- `train_colab.py` - 訓練腳本（含完整註解）

## ✅ 檢查清單

上傳前確認：
- [ ] 已將資料夾/ZIP 上傳到 Google Drive
- [ ] 在 Colab 選擇了 GPU Runtime
- [ ] 已掛載 Google Drive
- [ ] 已切換到正確的目錄
- [ ] 已安裝依賴套件

開始訓練吧！ 🚀

