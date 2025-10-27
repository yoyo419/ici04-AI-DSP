"""
🌩️ Google Colab 專用訓練腳本
適用於 Qwen2.5-3B Legal Delta 模型訓練

特點：
- 完全在 Colab Linux 環境運行
- 不依賴 conda（使用 pip）
- 自動下載模型
- 支援 GPU 加速
"""

import os
import sys
import time
from datetime import datetime

# 抑制 TensorFlow/CUDA 警告（Colab 環境常見）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# 環境設定
# ============================================================
print("=" * 70)
print("🌩️ LegalDelta Colab 訓練腳本")
print("=" * 70)
print()

# 檢查環境（使用多種方式檢測）
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    # 檢查環境變數
    if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
        IN_COLAB = True
    # 檢查是否在 /content 目錄（Colab 特有）
    elif os.path.exists('/content') and os.path.exists('/usr/local/lib/python3.10/dist-packages'):
        IN_COLAB = True

if IN_COLAB:
    print("✓ 在 Google Colab 環境中")
else:
    print("⚠️  不在 Colab 環境（本地測試模式）")

# 固定隨機種子
import torch
import numpy as np
import random

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(3407)

# 檢查 GPU
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  無 GPU，訓練將會非常慢")

print()

# ============================================================
# 超參數配置
# ============================================================
class Config:
    # 模型設定
    model_size = "3B"  # 可改為 "14B" 但需要更多記憶體
    base_model_name = f"Qwen/Qwen2.5-{model_size}-Instruct"
    base_model_path = f"/content/Colab/Qwen2.5-{model_size}-Instruct"
    
    # 訓練數據
    train_data = "/content/Colab/data/training_data.json"
    dev_data = "/content/Colab/data/valid_data.json"
    
    # 輸出路徑
    output_dir = "/content/Colab/outputs"
    lora_output = "/content/Colab/lora_adapter"
    log_dir = "/content/Colab/logs"
    
    # LoRA 配置
    lora_rank = 32
    lora_alpha = 32
    lora_dropout = 0.0
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                     "gate_proj", "up_proj", "down_proj"]
    
    # 訓練超參數
    learning_rate = 5e-5
    num_train_epochs = 3  # Colab 時間有限，減少 epochs
    per_device_batch_size = 1
    gradient_accumulation_steps = 32
    max_seq_length = 4096  # 減少以節省記憶體
    max_prompt_length = 2048
    max_completion_length = 512
    
    # GRPO 特定參數
    num_generations = 4
    
    # 其他設定
    save_steps = 100
    logging_steps = 10
    eval_steps = 100
    save_total_limit = 2
    
    # 優化器
    optim = "paged_adamw_8bit"  # 節省記憶體
    warmup_ratio = 0.1
    lr_scheduler_type = "cosine"

config = Config()

# 創建必要目錄
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(config.lora_output, exist_ok=True)

# 設定日誌
experiment_name = f"qwen{config.model_size}_colab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_file = os.path.join(config.log_dir, f"{experiment_name}.log")

def log_print(message):
    """同時輸出到控制台和日誌檔案"""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

log_print(f"實驗名稱: {experiment_name}")
log_print(f"日誌檔案: {log_file}")
log_print("")

# ============================================================
# 1. 檢查/下載基礎模型
# ============================================================
log_print("=" * 70)
log_print("步驟 1: 檢查基礎模型")
log_print("=" * 70)

if not os.path.exists(config.base_model_path):
    log_print(f"⬇️  下載模型: {config.base_model_name}")
    log_print("這可能需要 5-15 分鐘...")
    
    from huggingface_hub import snapshot_download
    
    try:
        model_path = snapshot_download(
            repo_id=config.base_model_name,
            local_dir=config.base_model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        log_print(f"✓ 模型下載完成: {model_path}")
    except Exception as e:
        log_print(f"❌ 模型下載失敗: {e}")
        sys.exit(1)
else:
    log_print(f"✓ 模型已存在: {config.base_model_path}")

log_print("")

# ============================================================
# 2. 檢查訓練數據
# ============================================================
log_print("=" * 70)
log_print("步驟 2: 檢查訓練數據")
log_print("=" * 70)

import json

if not os.path.exists(config.train_data):
    log_print(f"❌ 找不到訓練數據: {config.train_data}")
    log_print("請確認數據檔案已上傳")
    sys.exit(1)

with open(config.train_data, 'r', encoding='utf-8') as f:
    train_data = json.load(f)
log_print(f"✓ 訓練數據: {len(train_data)} 筆")

if os.path.exists(config.dev_data):
    with open(config.dev_data, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    log_print(f"✓ 驗證數據: {len(dev_data)} 筆")
else:
    log_print("⚠️  找不到驗證數據")
    dev_data = None

log_print("")

# ============================================================
# 3. 載入模型和 Tokenizer
# ============================================================
log_print("=" * 70)
log_print("步驟 3: 載入模型")
log_print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 載入 Tokenizer
log_print("載入 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_path,
    trust_remote_code=True,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

log_print("✓ Tokenizer 載入完成")

# 載入基礎模型
log_print("載入基礎模型...")
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
)

log_print("✓ 基礎模型載入完成")

# 應用 LoRA
log_print("應用 LoRA 配置...")
lora_config = LoraConfig(
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.target_modules,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
log_print("✓ LoRA 配置完成")

# 顯示可訓練參數
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
trainable_percent = 100 * trainable_params / all_params

log_print(f"可訓練參數: {trainable_params:,} / {all_params:,} ({trainable_percent:.2f}%)")
log_print("")

# ============================================================
# 4. 準備數據集
# ============================================================
log_print("=" * 70)
log_print("步驟 4: 準備數據集")
log_print("=" * 70)

from datasets import Dataset

SYSTEM_PROMPT = """用户与助手之间的对话。用户提出问题，助手解决它。助手首先思考推理过程，然后提供用户答案。推理过程和答案分别包含在 <reasoning> </reasoning> 和 <answer> </answer> 标签中。

请按照以下格式回答问题：
<reasoning>
在此详细分析问题并展示完整的推理过程，包括思考步骤、相关知识和逻辑分析。
</reasoning>
<answer>
在此提供简洁明确的最终答案。
</answer>"""

def format_dataset(data_list):
    """格式化並 tokenize 數據集"""
    formatted_data = []
    
    for item in data_list:
        # 構建 prompt
        prompt_text = f"{SYSTEM_PROMPT}\n\n[QUERY_ID:{item['id']}]\n你是一个法律专家，请按照以下要求回答：\n{item['instruction']}{item['question']}"
        
        # 構建回答
        answer_text = f"<answer>\n{item['answer']}\n</answer>"
        
        # 完整文本（用於訓練）
        full_text = f"{prompt_text}\n{answer_text}{tokenizer.eos_token}"
        
        # Tokenize 文本
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,  # 由 data_collator 處理 padding
        )
        
        formatted_data.append(tokenized)
    
    return Dataset.from_list(formatted_data)

train_dataset = format_dataset(train_data)
log_print(f"✓ 訓練集: {len(train_dataset)} 筆")

if dev_data:
    eval_dataset = format_dataset(dev_data)
    log_print(f"✓ 驗證集: {len(eval_dataset)} 筆")
else:
    eval_dataset = None

log_print("")

# ============================================================
# 5. 設定訓練參數
# ============================================================
log_print("=" * 70)
log_print("步驟 5: 配置訓練")
log_print("=" * 70)

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir=os.path.join(config.output_dir, experiment_name),
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_batch_size,
    per_device_eval_batch_size=config.per_device_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    warmup_ratio=config.warmup_ratio,
    lr_scheduler_type=config.lr_scheduler_type,
    logging_dir=os.path.join(config.log_dir, experiment_name),
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    eval_steps=config.eval_steps if eval_dataset else None,
    save_total_limit=config.save_total_limit,
    eval_strategy="steps" if eval_dataset else "no",  # 修復：改為 eval_strategy
    load_best_model_at_end=True if eval_dataset else False,
    metric_for_best_model="eval_loss" if eval_dataset else None,
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    bf16=False,
    optim=config.optim,
    max_grad_norm=1.0,
    report_to="tensorboard",
    remove_unused_columns=False,
    dataloader_num_workers=2,
)

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 評估指標函數
def compute_metrics(eval_preds):
    """
    計算評估指標
    
    返回的指標：
    - accuracy: Token 級別的準確率
    - f1: Token 級別的 F1 score (macro average)
    - precision: Token 級別的精確率
    - recall: Token 級別的召回率
    """
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support
    
    predictions, labels = eval_preds
    
    # predictions 是 logits，取 argmax 得到預測的 token
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # 將 logits 轉換為預測的 token ID
    predictions = np.argmax(predictions, axis=-1)
    
    # 計算準確率（忽略 padding tokens，通常是 -100）
    mask = labels != -100
    
    if mask.sum() > 0:
        # 提取有效的預測和標籤（排除 padding）
        valid_predictions = predictions[mask]
        valid_labels = labels[mask]
        
        # 1. Accuracy
        correct = (valid_predictions == valid_labels)
        accuracy = correct.sum() / len(valid_labels)
        
        # 2. Precision, Recall, F1 Score
        # 使用 macro average（對每個類別計算指標後取平均）
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                valid_labels,
                valid_predictions,
                average='macro',  # macro average: 對所有類別平等對待
                zero_division=0   # 避免除零錯誤
            )
        except Exception as e:
            # 如果計算失敗（例如只有一個類別），使用 accuracy 作為後備
            log_print(f"⚠️  F1 計算警告: {e}")
            precision = accuracy
            recall = accuracy
            f1 = accuracy
        
    else:
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics if eval_dataset else None,  # 添加評估指標
)

log_print("✓ 訓練配置完成")
log_print(f"  - Epochs: {config.num_train_epochs}")
log_print(f"  - Batch size: {config.per_device_batch_size}")
log_print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
log_print(f"  - Effective batch size: {config.per_device_batch_size * config.gradient_accumulation_steps}")
log_print(f"  - Learning rate: {config.learning_rate}")
log_print("")

# ============================================================
# 6. 開始訓練
# ============================================================
log_print("=" * 70)
log_print("步驟 6: 開始訓練")
log_print("=" * 70)
log_print("")

start_time = time.time()

try:
    log_print("🚀 訓練開始...")
    log_print("")
    
    # 訓練
    trainer.train()
    
    # 保存最終模型
    log_print("")
    log_print("=" * 70)
    log_print("保存模型...")
    
    trainer.save_model(config.lora_output)
    tokenizer.save_pretrained(config.lora_output)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    log_print(f"✓ 訓練完成！")
    log_print(f"✓ 訓練時間: {hours} 小時 {minutes} 分鐘")
    log_print(f"✓ LoRA adapter 已保存到: {config.lora_output}")
    log_print("=" * 70)
    
    # 如果在 Colab，保存到 Google Drive
    if IN_COLAB:
        log_print("")
        log_print("嘗試保存到 Google Drive...")
        
        try:
            from google.colab import drive
            import shutil
            
            # 檢查 Drive 是否已掛載
            if not os.path.exists('/content/drive'):
                drive.mount('/content/drive')
            
            # 保存路徑
            save_base = f'/content/drive/MyDrive/LegalDelta_Results/{experiment_name}'
            os.makedirs(save_base, exist_ok=True)
            
            # 複製 LoRA adapter
            shutil.copytree(config.lora_output, f'{save_base}/lora_adapter')
            
            # 複製日誌
            if os.path.exists(config.log_dir):
                shutil.copytree(config.log_dir, f'{save_base}/logs', dirs_exist_ok=True)
            
            log_print(f"✓ 結果已保存到 Google Drive: {save_base}")
            
        except Exception as e:
            log_print(f"⚠️  無法保存到 Drive: {e}")
            log_print("請手動複製結果檔案")
    
    log_print("")
    log_print("🎉 訓練流程全部完成！")
    
except KeyboardInterrupt:
    log_print("")
    log_print("⚠️  訓練被中斷")
    log_print("部分結果可能已保存在 checkpoint 中")
    
except Exception as e:
    log_print("")
    log_print(f"❌ 訓練失敗: {e}")
    import traceback
    log_print(traceback.format_exc())
    raise

finally:
    # 清理記憶體
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

log_print("")
log_print("=" * 70)
log_print("腳本執行結束")
log_print("=" * 70)

