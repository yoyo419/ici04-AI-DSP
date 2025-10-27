# 🔄 Colab 模型合併指南

## 📋 概述

訓練完成後，您需要將 **LoRA adapter** 與 **基礎模型** 合併，才能得到完整的可用模型。

---

## 🎯 合併流程

```
訓練完成
   ↓
生成 LoRA adapter (/content/Colab/lora_adapter)
   ↓
執行 merge_lora_colab.py
   ↓
合併後的完整模型 (/content/Colab/merged_model)
   ↓
可直接用於推理
```

---

## 🚀 使用方法

### 方法一：基本使用（自動配置）

```python
# 在 Colab 中執行
!python merge_lora_colab.py
```

**自動使用的路徑：**
- LoRA adapter: `/content/Colab/lora_adapter`
- Base model: `/content/Colab/Qwen2.5-3B-Instruct`
- 輸出: `/content/Colab/merged_model`

---

### 方法二：自訂路徑

```python
!python merge_lora_colab.py \
  --lora_adapter_path /content/Colab/lora_adapter \
  --base_model_path /content/Colab/Qwen2.5-3B-Instruct \
  --save_path /content/Colab/my_merged_model
```

---

### 方法三：記憶體不足時使用 CPU

```python
# 如果 GPU 記憶體不足
!python merge_lora_colab.py --device_map cpu
```

**注意：** CPU 模式會較慢，但可避免 OOM（Out of Memory）錯誤。

---

### 方法四：自動備份到 Google Drive

```python
# 合併後自動備份到 Google Drive
!python merge_lora_colab.py --save_to_drive
```

**備份位置：** `/content/drive/MyDrive/LegalDelta/merged_models/merged_model_YYYYMMDD_HHMMSS`

---

## 📂 檔案結構

### 合併前

```
/content/Colab/
├── Qwen2.5-3B-Instruct/          ← 基礎模型
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
│
└── lora_adapter/                  ← 訓練產生的 LoRA adapter
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

### 合併後

```
/content/Colab/
└── merged_model/                  ← 合併後的完整模型
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

---

## 🎬 完整範例（從訓練到推理）

```python
# ===== 步驟 1: 訓練 =====
!python train_colab.py

# 訓練完成後會生成 /content/Colab/lora_adapter

# ===== 步驟 2: 合併 =====
!python merge_lora_colab.py --save_to_drive

# 合併完成後會生成 /content/Colab/merged_model

# ===== 步驟 3: 測試推理 =====
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入合併後的模型
model = AutoModelForCausalLM.from_pretrained(
    "/content/Colab/merged_model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/content/Colab/merged_model")

# 準備法律問題
system_prompt = """用户与助手之间的对话。用户提出问题，助手解决它。助手首先思考推理过程，然后提供用户答案。推理过程和答案分别包含在 <reasoning> </reasoning> 和 <answer> </answer> 标签中。"""

question = "依勞動基準法規定，雇主延長勞工之工作時間連同正常工作時間，每日不得超過多少小時？"

# 格式化輸入
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question}
]

# 生成回答
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

# 解碼輸出
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

---

## ⚙️ 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--lora_adapter_path` | LoRA adapter 路徑 | `/content/Colab/lora_adapter` |
| `--base_model_path` | 基礎模型路徑 | `/content/Colab/Qwen2.5-3B-Instruct` |
| `--save_path` | 合併後模型儲存路徑 | `/content/Colab/merged_model` |
| `--device_map` | 裝置映射 (auto/cpu/cuda) | `auto` |
| `--save_to_drive` | 是否備份到 Google Drive | `False` |
| `--drive_path` | Google Drive 備份路徑 | `/content/drive/MyDrive/LegalDelta/merged_models` |

---

## 🛠️ 常見問題

### Q1: 合併時遇到 OOM（記憶體不足）

**解決方案：**
```python
!python merge_lora_colab.py --device_map cpu
```

或者清理記憶體後重試：
```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

!python merge_lora_colab.py
```

---

### Q2: `adapter_config.json` 找不到

**原因：** 訓練未完成或 LoRA adapter 未成功保存。

**解決方案：**
1. 檢查訓練日誌，確認訓練已完成
2. 確認 `/content/Colab/lora_adapter` 目錄存在且包含以下檔案：
   - `adapter_config.json`
   - `adapter_model.safetensors`

```python
# 檢查檔案
!ls -lh /content/Colab/lora_adapter
```

---

### Q3: 合併後模型無法載入

**解決方案：**
檢查合併後的模型檔案是否完整：
```python
!ls -lh /content/Colab/merged_model

# 應該包含：
# - config.json
# - model.safetensors (或 model-00001-of-0000X.safetensors)
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
```

---

### Q4: `frozenset` 錯誤

**錯誤訊息：**
```
AttributeError: 'frozenset' object has no attribute 'discard'
```

**原因：** 在量化模式下訓練的模型，合併時可能遇到此問題。

**解決方案：**
1. 確保訓練時使用 `load_in_4bit=True` 但合併時使用 `float16`
2. 如果問題持續，請在訓練時改用 `torch_dtype=torch.float16` 而非量化

---

### Q5: 合併需要多久？

**時間估算：**
- **3B 模型（Colab T4 GPU）：** 約 5-10 分鐘
- **3B 模型（CPU 模式）：** 約 15-30 分鐘
- **14B 模型：** 不建議在 Colab 免費版合併（記憶體不足）

---

## 📊 合併過程輸出範例

```
======================================================================
🌩️ Colab LoRA 合併工具
======================================================================

[2025-10-26 14:30:00] ✓ 在 Google Colab 環境中

[2025-10-26 14:30:01] 📋 配置信息：
[2025-10-26 14:30:01]   LoRA adapter: /content/Colab/lora_adapter
[2025-10-26 14:30:01]   Base model: /content/Colab/Qwen2.5-3B-Instruct
[2025-10-26 14:30:01]   Save to: /content/Colab/merged_model
[2025-10-26 14:30:01]   Device: auto

[2025-10-26 14:30:02] 🔍 檢查檔案...
[2025-10-26 14:30:02]   ✓ LoRA adapter 找到
[2025-10-26 14:30:02]   ✓ Base model 找到
[2025-10-26 14:30:02]   ✓ adapter_config.json 找到

[2025-10-26 14:30:03] 📋 載入 LoRA 配置...
[2025-10-26 14:30:03]   ✓ 配置載入成功

[2025-10-26 14:30:04] 📝 載入 tokenizer...
[2025-10-26 14:30:04]   ✓ Tokenizer 載入成功

[2025-10-26 14:30:05] 🧠 載入基礎模型...
[2025-10-26 14:30:05]   這可能需要幾分鐘...
[2025-10-26 14:32:15]   ✓ 基礎模型載入成功

[2025-10-26 14:32:16] 🔧 載入 LoRA adapter...
[2025-10-26 14:32:30]   ✓ LoRA adapter 載入成功

[2025-10-26 14:32:31] 🔄 合併 LoRA 權重到基礎模型...
[2025-10-26 14:32:31]   這可能需要幾分鐘...
[2025-10-26 14:35:12]   ✓ 合併完成

[2025-10-26 14:35:13] 🧹 清理記憶體...
[2025-10-26 14:35:14]   ✓ 記憶體清理完成

[2025-10-26 14:35:15] 💾 儲存合併後的模型到 /content/Colab/merged_model...
[2025-10-26 14:35:15]   這可能需要幾分鐘...
[2025-10-26 14:35:16]   移動模型到 CPU...
[2025-10-26 14:38:45]   ✓ 模型儲存成功

[2025-10-26 14:38:46] 🧹 最終清理...

======================================================================
[2025-10-26 14:38:47] ✅ 合併完成！
======================================================================

[2025-10-26 14:38:47] 📂 合併後的模型位置:
[2025-10-26 14:38:47]    /content/Colab/merged_model

[2025-10-26 14:38:47] 🚀 下一步：
[2025-10-26 14:38:47]    1. 使用合併後的模型進行推理
[2025-10-26 14:38:47]    2. 如果在 Colab，記得備份到 Google Drive
[2025-10-26 14:38:47]    3. 推理範例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('/content/Colab/merged_model')
tokenizer = AutoTokenizer.from_pretrained('/content/Colab/merged_model')

prompt = '依勞動基準法規定，雇主延長勞工之工作時間...'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

======================================================================
```

---

## 💡 最佳實踐

1. **訓練完成後立即合併**
   - Colab 會話有時限，訓練完成後儘快合併

2. **備份到 Google Drive**
   ```python
   !python merge_lora_colab.py --save_to_drive
   ```

3. **驗證合併結果**
   ```python
   # 測試載入
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model = AutoModelForCausalLM.from_pretrained("/content/Colab/merged_model")
   tokenizer = AutoTokenizer.from_pretrained("/content/Colab/merged_model")
   
   print("✅ 模型載入成功！")
   ```

4. **清理臨時檔案**
   ```python
   # 合併完成後，如果空間不足，可以刪除 lora_adapter
   # （但建議先備份到 Drive）
   !rm -rf /content/Colab/lora_adapter
   ```

---

## 📚 相關文件

- **`train_colab.py`** - Colab 訓練腳本
- **`COLAB_QUICKSTART.md`** - Colab 快速開始指南
- **`訓練指標說明.md`** - 訓練指標詳解

---

## 🆘 需要幫助？

如果遇到問題，請檢查：
1. 訓練是否成功完成
2. LoRA adapter 檔案是否完整
3. GPU/CPU 記憶體是否足夠
4. 路徑是否正確

**常用除錯指令：**
```python
# 檢查 LoRA adapter
!ls -lh /content/Colab/lora_adapter

# 檢查 GPU 記憶體
!nvidia-smi

# 檢查可用空間
!df -h /content
```

---

**祝您合併順利！** 🎉

