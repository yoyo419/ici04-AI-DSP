#!/usr/bin/env python3
"""
🧪 基礎模型測試腳本（僅測試 Qwen2.5-3B-Instruct）
用於測試基礎模型（不使用任何 LoRA adapter）在測試集上的表現
"""

import os
import sys
import argparse
from datetime import datetime
import re
import json

import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from tqdm import tqdm

# 抑制警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
class TestConfig:
    # 模型路徑
    base_model_path = "/content/Colab_Download/Qwen2.5-3B-Instruct"
    
    # 測試數據路徑
    test_data = "/content/Colab_Upload/data/test_data.json"
    
    # 輸出路徑
    output_dir = "/content/Colab/test_results_base"
    
    # 生成參數
    max_seq_length = 2048
    max_new_tokens = 512
    temperature = 0.7
    top_p = 0.9
    do_sample = True
    
    # 測試樣本數（設為 None 測試全部）
    num_test_samples = None

# ============================================================
# System Prompts（與訓練一致）
# ============================================================
SYSTEM_PROMPT = """用户与助手之间的对话。用户提出问题，助手解决它。助手首先思考推理过程，然后提供用户答案。推理过程和答案分别包含在 <reasoning> </reasoning> 和 <answer> </answer> 标签中。

请按照以下格式回答问题：
<reasoning>
在此详细分析问题并展示完整的推理过程，包括思考步骤、相关知识和逻辑分析。
</reasoning>
<answer>
在此提供简洁明确的最终答案(回答至少一個選項，注意，如果答案不只一個選項，如12345，一定要輸出(12345)，不要輸出(1)(2)(3)(4)(5))。
</answer>"""

SYSTEM_PROMPT_BASE = """用户与助手之间的对话。用户提出问题，助手解决它。答案包含在 <answer> </answer> 标签中。

请按照以下格式回答问题：
<answer>
在此提供简洁明确的最终答案(回答至少一個選項，注意，如果答案不只一個選項，如12345，一定要輸出(12345)，不要輸出(1)(2)(3)(4)(5))。
</answer>"""

EXAMPLE_TEXT = """
你是一个職業安全衛生專家，请按照以下要求回答：
 """

# ============================================================
# 工具函數：從文字中抽取選項數字
# ============================================================
def extract_choices_from_text(text: str):
    """
    從文字中抓出所有括號內的數字選項，例如：
    (1)(2)(3)        -> "123"
    <eoa> -> "3"
    若找不到則回傳 None
    """
    # 支援全形/半形括號
    nums = re.findall(r"[（(]([0-9]+)[）)]", text)
    if not nums:
        return None
    return "".join(nums)

# ============================================================
# 數據載入函數
# ============================================================
def get_questions(file_path: str) -> Dataset:
    """
    載入並處理測試數據集
    """
    # 檢查文件是否存在，嘗試本地路徑
    if not os.path.exists(file_path):
        local_paths = [
            file_path.replace("/content/Colab_Upload/", "./"),
            file_path.replace("/content/Colab_Upload/", "../"),
            "./data/test_data.json",
            "../data/test_data.json"
        ]
        
        for local_path in local_paths:
            if os.path.exists(local_path):
                file_path = local_path
                print(f"✓ 使用本地路徑: {file_path}")
                break
        else:
            raise FileNotFoundError(f"❌ 找不到資料檔案: {file_path}")
    
    print(f"📂 載入資料集: {file_path}")
    
    # 載入數據
    data = load_dataset("json", data_files=file_path)
    data = data['train']  # HuggingFace datasets 統一使用 'train' key
    
    print(f"✓ 原始資料集大小: {len(data)}")
    
    # 處理每個樣本
    def process_sample(x: dict) -> dict:
        """處理單個樣本，準備 prompt 格式"""
        return { 
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"[QUERY_ID:{x['id']}]\n[QUERY_SOURCE:{x.get('source', 'unknown')}]\n" + EXAMPLE_TEXT + x['instruction'] + x['question']}
            ],
            'answer': x['answer'],
            'id': x['id'],
            'question': x['question'],
            'instruction': x['instruction'],
            'source': x.get('source', 'unknown')
        }
    
    data = data.map(process_sample)
    print(f"✓ 處理後資料集大小: {len(data)}")
    
    return data

# ============================================================
# 日誌函數
# ============================================================
def log_print(message: str):
    """印出訊息到控制台"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

# ============================================================
# 主測試函數
# ============================================================
def test_model(config: TestConfig):
    """
    測試模型
    """
    print()
    print("=" * 70)
    print("🧪 基礎模型測試（Qwen2.5-3B-Instruct）")
    print("=" * 70)
    print()
    
    # 創建輸出目錄
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ============================================================
    # 1. 載入模型
    # ============================================================
    print("=" * 70)
    print("步驟 1: 載入基礎模型")
    print("=" * 70)
    
    try:
        log_print(f"📦 載入 Qwen2.5-3B-Instruct: {config.base_model_path}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model_path,
            max_seq_length=config.max_seq_length,
            dtype=None,  # 自動選擇 dtype
            load_in_4bit=False,
        )
        log_print("✓ 基礎模型載入完成")
        
        # 設置為推理模式
        FastLanguageModel.for_inference(model)
        model.eval()
        log_print("✓ 模型已設置為推理模式")
        
    except Exception as e:
        log_print(f"❌ 模型載入失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    
    # ============================================================
    # 2. 載入測試數據
    # ============================================================
    print("=" * 70)
    print("步驟 2: 載入測試數據")
    print("=" * 70)
    
    try:
        test_dataset = get_questions(file_path=config.test_data)
        
        # 限制測試樣本數
        if config.num_test_samples is not None:
            original_size = len(test_dataset)
            test_dataset = test_dataset.select(range(min(config.num_test_samples, len(test_dataset))))
            log_print(f"✓ 測試樣本數限制: {len(test_dataset)}/{original_size}")
        else:
            log_print(f"✓ 測試全部樣本: {len(test_dataset)}")
            
    except Exception as e:
        log_print(f"❌ 測試數據載入失敗: {e}")
        sys.exit(1)
    
    print()
    
    # ============================================================
    # 3. 進行測試
    # ============================================================
    print("=" * 70)
    print("步驟 3: 生成答案")
    print("=" * 70)
    print()
    
    results = []
    correct_count = 0
    
    for idx in tqdm(range(len(test_dataset)), desc="生成答案"):
        sample = test_dataset[idx]
        
        try:
            # 準備輸入
            messages = sample['prompt']
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            # 生成答案
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 解碼生成的答案
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # -------- 改良版：只比對選項數字 --------
            ground_truth_raw = sample['answer'].strip()
            gt_choices = extract_choices_from_text(ground_truth_raw)
            pred_choices = extract_choices_from_text(predicted_answer)

            if gt_choices is not None and pred_choices is not None:
                gt_norm = "".join(sorted(gt_choices))
                pred_norm = "".join(sorted(pred_choices))
                is_correct = (gt_norm == pred_norm)
            else:
                # fallback：找不到選項時，退回原本的包含判斷
                is_correct = ground_truth_raw in predicted_answer

            if is_correct:
                correct_count += 1
            
            # 儲存結果
            result = {
                'id': sample['id'],
                'source': sample['source'],
                'question': sample['question'],
                'instruction': sample['instruction'],
                'ground_truth': ground_truth_raw,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
            }
            results.append(result)
            
            # 定期顯示進度和樣本
            if (idx + 1) % 10 == 0 or idx < 3:
                print(f"\n{'='*70}")
                print(f"樣本 {idx+1}/{len(test_dataset)}")
                print(f"當前準確率: {correct_count}/{idx+1} ({100*correct_count/(idx+1):.2f}%)")
                print(f"{'='*70}")
                print(f"📋 ID: {sample['id']}")
                print(f"📂 Source: {sample['source']}")
                print(f"\n❓ 問題:")
                print(f"  {sample['question'][:200]}...")
                print(f"\n🤖 模型預測答案:")
                print(f"  {predicted_answer[:500]}...")
                print(f"\n✅ 真實答案:")
                print(f"  {ground_truth_raw[:500]}...")
                print(f"\n{'✓' if is_correct else '✗'} {'正確' if is_correct else '錯誤'}")
                print()
                
        except Exception as e:
            log_print(f"⚠️  樣本 {idx} 生成失敗: {e}")
            results.append({
                'id': sample['id'],
                'source': sample.get('source', 'unknown'),
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'predicted_answer': f"[ERROR] {str(e)}",
                'is_correct': False
            })
    
    print()
    print("=" * 70)
    print("✓ 答案生成完成")
    print("=" * 70)
    print()

    # ============================================================
    # 4. 計算並顯示統計結果
    # ============================================================
    print("=" * 70)
    print("📊 測試結果統計")
    print("=" * 70)
    
    total_samples = len(results)
    accuracy = (correct_count / total_samples * 100) if total_samples > 0 else 0
    
    log_print(f"總樣本數: {total_samples}")
    log_print(f"正確數量: {correct_count}")
    log_print(f"錯誤數量: {total_samples - correct_count}")
    log_print(f"準確率: {accuracy:.2f}%")
    
    print()
    
    # ============================================================
    # 5. 保存結果
    # ============================================================
    print("=" * 70)
    print("步驟 5: 保存結果")
    print("=" * 70)
    
    # 保存詳細結果（JSON）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.output_dir, f"test_results_base_{timestamp}.json")
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'model': config.base_model_path,
                    'test_data': config.test_data,
                    'timestamp': timestamp,
                    'total_samples': total_samples,
                    'correct_count': correct_count,
                    'accuracy': accuracy,
                    'config': {
                        'max_new_tokens': config.max_new_tokens,
                        'temperature': config.temperature,
                        'top_p': config.top_p
                    }
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        log_print(f"✓ 詳細結果已保存: {results_file}")
    except Exception as e:
        log_print(f"❌ 保存結果失敗: {e}")
    
    # 保存摘要（文本文件）
    summary_file = os.path.join(config.output_dir, f"summary_base_{timestamp}.txt")
    
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("基礎模型測試摘要\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"模型: {config.base_model_path}\n")
            f.write(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"測試數據: {config.test_data}\n\n")
            f.write(f"總樣本數: {total_samples}\n")
            f.write(f"正確數量: {correct_count}\n")
            f.write(f"錯誤數量: {total_samples - correct_count}\n")
            f.write(f"準確率: {accuracy:.2f}%\n\n")
            f.write("=" * 70 + "\n")
        
        log_print(f"✓ 摘要已保存: {summary_file}")
    except Exception as e:
        log_print(f"❌ 保存摘要失敗: {e}")
    
    print()
    print("=" * 70)
    print("🎉 測試完成！")
    print("=" * 70)
    print(f"✓ 準確率: {accuracy:.2f}% ({correct_count}/{total_samples})")
    print(f"✓ 結果保存在: {config.output_dir}")
    print("=" * 70)

# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="基礎模型測試工具（Qwen2.5-3B）")
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="/content/Colab_Download/Qwen2.5-3B-Instruct",
        help="基礎模型路徑"
    )
    parser.add_argument(
        "--test_data", 
        type=str, 
        default="/content/Colab_Upload/data/test_data.json",
        help="測試數據路徑"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/content/Colab/test_results_base",
        help="輸出目錄"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=None,
        help="測試樣本數（None 表示全部）"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="生成溫度"
    )
    
    args = parser.parse_args()
    
    # 更新配置
    config = TestConfig()
    config.base_model_path = args.base_model
    config.test_data = args.test_data
    config.output_dir = args.output_dir
    config.num_test_samples = args.num_samples
    config.temperature = args.temperature
    
    # 執行測試
    test_model(config)

if __name__ == "__main__":
    main()
