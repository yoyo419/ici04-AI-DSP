# 🌩️ Colab 快速開始指南

## 📦 步驟 1: 準備檔案（在 Windows 本機）

```powershell
# 在專案目錄執行
.\prepare_colab.ps1
```

這會創建 `LegalDelta_Colab.zip`（約 10-20 MB）

## ☁️ 步驟 2: 上傳到 Google Drive

1. 前往 https://drive.google.com
2. 創建資料夾：`LegalDelta`
3. 上傳 `LegalDelta_Colab.zip`

## 🚀 步驟 3: 在 Colab 執行

### 3.1 開啟 Colab

1. 前往 https://colab.research.google.com
2. 新建 Notebook
3. **重要**: Runtime → Change runtime type → Hardware accelerator → **GPU (T4)**

### 3.2 執行以下代碼

**Cell 1: 掛載 Drive 並解壓縮**

```python
# 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 解壓縮檔案到 /content/Colab
import zipfile
import os

zip_path = '/content/drive/MyDrive/LegalDelta/Colab_Upload.zip'
extract_path = '/content/Colab'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

os.chdir(extract_path)
print(f"✓ 當前目錄: {os.getcwd()}")
print(f"✓ 解壓縮完成")
!ls -la
```

**Cell 2: 安裝依賴**

```python
# 安裝必要套件
!pip install -q accelerate peft datasets trl bitsandbytes sentencepiece protobuf

# 驗證安裝
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Cell 3: 開始訓練**

```python
# 執行訓練腳本
!python train_colab.py
```

### 3.3 監控訓練（可選）

在新的 Cell 中執行：

```python
# 查看即時日誌
!tail -f logs/*.log
```

或使用 TensorBoard：

```python
%load_ext tensorboard
%tensorboard --logdir outputs
```

## 📥 步驟 4: 下載結果

訓練完成後，結果會自動保存到 Google Drive：
```
/content/drive/MyDrive/LegalDelta_Results/qwen3B_colab_YYYYMMDD_HHMMSS/
├── lora_adapter/     # LoRA 權重
└── logs/             # 訓練日誌
```

或手動下載：

```python
# 打包 LoRA adapter
import shutil
shutil.make_archive('lora_adapter', 'zip', './lora_adapter')

# 下載
from google.colab import files
files.download('lora_adapter.zip')
```

## ⏱️ 預期時間

- 環境設定: ~5 分鐘
- 模型下載: ~10 分鐘
- 訓練 (3 epochs): ~2-4 小時
- **總計: ~2.5-4.5 小時**

## ⚠️ 注意事項

1. **保持連線**: Colab 免費版閒置 90 分鐘會斷線
2. **保存進度**: 腳本會定期保存 checkpoint
3. **GPU 限制**: 免費版每天約 12 小時 GPU 時間
4. **自動保存**: 結果會自動同步到 Google Drive

## 🔧 如果遇到問題

### 問題 1: 記憶體不足 (OOM)

修改 `train_colab.py` 中的配置：

```python
class Config:
    per_device_batch_size = 1  # 已經是最小
    gradient_accumulation_steps = 64  # 增加這個
    max_seq_length = 2048  # 減少序列長度
```

### 問題 2: Colab 斷線

重新執行 Cell 3，Trainer 會自動從最新的 checkpoint 繼續。

### 問題 3: 找不到檔案

檢查 Drive 路徑：

```python
!ls -la /content/drive/MyDrive/LegalDelta/
```

確認 `LegalDelta_Colab.zip` 已上傳。

## 📊 完成後的使用

LoRA adapter 保存在：
- Colab: `./lora_adapter/`
- Drive: `/content/drive/MyDrive/LegalDelta_Results/.../lora_adapter/`

下載後，在本地使用 `merge_lora.py` 合併：

```bash
python src/merge_lora.py \
    --model_name_or_path ./lora_adapter \
    --base_model_path ./Qwen2.5-3B-Instruct \
    --save_path ./LegalDelta/Qwen2.5-3B-merge
```

## 🎯 總結

這個方案：
- ✅ 完全在 Linux (Colab) 環境運行
- ✅ 不需要 conda
- ✅ 自動下載模型
- ✅ 訓練只需 2-4 小時
- ✅ 結果自動保存到 Drive
- ✅ 支援斷點續傳

開始訓練吧！ 🚀

