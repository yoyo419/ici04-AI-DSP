#!/usr/bin/env python3
"""
🌩️ Google Colab 專用 LoRA 合併腳本
適用於 Qwen2.5 Legal Delta 模型訓練後的合併操作

使用方法（在 Colab 中）:
    python merge_lora_colab.py
    python merge_lora_colab.py --device_map cpu  # 記憶體不足時使用
"""

import argparse
import json
import os
import gc
from datetime import datetime

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def log_print(message):
    """同時輸出到控制台和日誌檔案"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def main():
    """主函數"""
    # ============================================================
    # 環境設定
    # ============================================================
    print("=" * 70)
    print("🌩️ Colab LoRA 合併工具")
    print("=" * 70)
    print()
    
    # 檢查是否在 Colab 環境
    IN_COLAB = False
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            IN_COLAB = True
        elif os.path.exists('/content') and os.path.exists('/usr/local/lib/python3.10/dist-packages'):
            IN_COLAB = True
    
    if IN_COLAB:
        log_print("✓ 在 Google Colab 環境中")
    else:
        log_print("⚠️  不在 Colab 環境（本地測試模式）")
    
    print()
    
    # 設定離線模式（避免不必要的網絡請求）
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    # ============================================================
    # 解析參數
    # ============================================================
    parser = argparse.ArgumentParser(description="Colab LoRA Merge Tool")
    parser.add_argument(
        "--lora_adapter_path", 
        type=str, 
        default="/content/Colab/lora_adapter",
        help="Path to the trained LoRA adapter"
    )
    parser.add_argument(
        "--base_model_path", 
        type=str,
        default="/content/Colab/Qwen2.5-3B-Instruct", 
        help="Path to base model"
    )
    parser.add_argument(
        "--save_path", 
        type=str,
        default="/content/Colab/merged_model",
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--save_to_drive",
        action="store_true",
        help="Save merged model to Google Drive"
    )
    parser.add_argument(
        "--drive_path",
        type=str,
        default="/content/drive/MyDrive/LegalDelta/merged_models",
        help="Google Drive path to save merged model"
    )
    
    args = parser.parse_args()
    
    # ============================================================
    # 顯示配置
    # ============================================================
    log_print("📋 配置信息：")
    log_print(f"  LoRA adapter: {args.lora_adapter_path}")
    log_print(f"  Base model: {args.base_model_path}")
    log_print(f"  Save to: {args.save_path}")
    log_print(f"  Device: {args.device_map}")
    if args.save_to_drive:
        log_print(f"  Drive backup: {args.drive_path}")
    print()
    
    # ============================================================
    # 驗證路徑
    # ============================================================
    log_print("🔍 檢查檔案...")
    
    if not os.path.exists(args.lora_adapter_path):
        log_print(f"❌ LoRA adapter 不存在: {args.lora_adapter_path}")
        log_print("\n💡 請確認訓練已完成，並且 LoRA adapter 已保存。")
        return
    
    if not os.path.exists(args.base_model_path):
        log_print(f"❌ Base model 不存在: {args.base_model_path}")
        log_print("\n💡 請先下載基礎模型。")
        return
    
    # 檢查必要檔案
    adapter_config = os.path.join(args.lora_adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        log_print(f"❌ adapter_config.json 不存在於 {args.lora_adapter_path}")
        return
    
    log_print("  ✓ LoRA adapter 找到")
    log_print("  ✓ Base model 找到")
    log_print("  ✓ adapter_config.json 找到")
    print()
    
    # ============================================================
    # 載入配置
    # ============================================================
    log_print("📋 載入 LoRA 配置...")
    try:
        config = PeftConfig.from_pretrained(
            args.lora_adapter_path, 
            local_files_only=True
        )
        log_print("  ✓ 配置載入成功")
    except Exception as e:
        log_print(f"⚠️  從 PeftConfig 載入失敗: {e}")
        log_print("  嘗試直接從 JSON 載入...")
        try:
            with open(adapter_config, 'r') as f:
                config_dict = json.load(f)
            from peft import LoraConfig
            config = LoraConfig(**config_dict)
            log_print("  ✓ 從 JSON 載入成功")
        except Exception as e2:
            log_print(f"❌ 配置載入失敗: {e2}")
            return
    
    print()
    
    # ============================================================
    # 載入 Tokenizer
    # ============================================================
    log_print("📝 載入 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        log_print("  ✓ Tokenizer 載入成功")
    except Exception as e:
        log_print(f"❌ Tokenizer 載入失敗: {e}")
        return
    
    print()
    
    # ============================================================
    # 載入基礎模型
    # ============================================================
    log_print("🧠 載入基礎模型...")
    log_print("  這可能需要幾分鐘...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16,  # 使用 float16 節省記憶體
            device_map=args.device_map,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        log_print("  ✓ 基礎模型載入成功")
    except Exception as e:
        log_print(f"❌ 基礎模型載入失敗: {e}")
        log_print("\n💡 如果記憶體不足，請嘗試：")
        log_print("   python merge_lora_colab.py --device_map cpu")
        return
    
    print()
    
    # ============================================================
    # 載入 LoRA Adapter
    # ============================================================
    log_print("🔧 載入 LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(
            model,
            args.lora_adapter_path,
            local_files_only=True,
            is_trainable=False
        )
        log_print("  ✓ LoRA adapter 載入成功")
    except Exception as e:
        log_print(f"❌ LoRA adapter 載入失敗: {e}")
        return
    
    print()
    
    # ============================================================
    # 合併模型
    # ============================================================
    log_print("🔄 合併 LoRA 權重到基礎模型...")
    log_print("  這可能需要幾分鐘...")
    
    try:
        model = model.merge_and_unload()
        log_print("  ✓ 合併完成")
    except Exception as e:
        log_print(f"❌ 合併失敗: {e}")
        log_print("\n💡 如果遇到 'frozenset' 錯誤，這是已知問題。")
        log_print("   模型可能需要重新載入為 float16（而非量化格式）後再合併。")
        return
    
    print()
    
    # ============================================================
    # 清理記憶體
    # ============================================================
    log_print("🧹 清理記憶體...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_print("  ✓ 記憶體清理完成")
    
    print()
    
    # ============================================================
    # 儲存合併後的模型
    # ============================================================
    log_print(f"💾 儲存合併後的模型到 {args.save_path}...")
    log_print("  這可能需要幾分鐘...")
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # 移動到 CPU 以節省記憶體
    log_print("  移動模型到 CPU...")
    model = model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 儲存模型
    try:
        model.save_pretrained(
            args.save_path,
            max_shard_size="1GB",  # 分片儲存，每個檔案最大 1GB
            safe_serialization=True
        )
        tokenizer.save_pretrained(args.save_path)
        log_print("  ✓ 模型儲存成功")
    except Exception as e:
        log_print(f"❌ 模型儲存失敗: {e}")
        return
    
    print()
    
    # ============================================================
    # 備份到 Google Drive（可選）
    # ============================================================
    if args.save_to_drive and IN_COLAB:
        log_print("📤 備份到 Google Drive...")
        
        try:
            # 掛載 Google Drive
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            
            # 創建目標目錄
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            drive_save_path = os.path.join(
                args.drive_path, 
                f"merged_model_{timestamp}"
            )
            os.makedirs(drive_save_path, exist_ok=True)
            
            # 複製檔案
            log_print(f"  複製到: {drive_save_path}")
            import shutil
            shutil.copytree(args.save_path, drive_save_path, dirs_exist_ok=True)
            
            log_print("  ✓ 已備份到 Google Drive")
            log_print(f"  位置: {drive_save_path}")
        except Exception as e:
            log_print(f"⚠️  備份到 Drive 失敗: {e}")
            log_print("  合併後的模型仍保存在 Colab 臨時儲存中")
    
    print()
    
    # ============================================================
    # 最終清理
    # ============================================================
    log_print("🧹 最終清理...")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print()
    
    # ============================================================
    # 完成
    # ============================================================
    print("=" * 70)
    log_print("✅ 合併完成！")
    print("=" * 70)
    print()
    log_print("📂 合併後的模型位置:")
    log_print(f"   {args.save_path}")
    
    if args.save_to_drive and IN_COLAB:
        log_print(f"\n📂 Google Drive 備份位置:")
        log_print(f"   {drive_save_path}")
    
    print()
    log_print("🚀 下一步：")
    log_print("   1. 使用合併後的模型進行推理")
    log_print("   2. 如果在 Colab，記得備份到 Google Drive")
    log_print("   3. 推理範例：")
    print()
    print("```python")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print()
    print(f"model = AutoModelForCausalLM.from_pretrained('{args.save_path}')")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{args.save_path}')")
    print()
    print("prompt = '依勞動基準法規定，雇主延長勞工之工作時間...'")
    print("inputs = tokenizer(prompt, return_tensors='pt')")
    print("outputs = model.generate(**inputs, max_length=512)")
    print("print(tokenizer.decode(outputs[0], skip_special_tokens=True))")
    print("```")
    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  合併被用戶中斷")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        print("\n💡 可能的解決方案:")
        print("   1. 記憶體不足 → 使用 --device_map cpu")
        print("   2. 檔案不存在 → 檢查路徑是否正確")
        print("   3. 權限問題 → 確認有寫入權限")
        import traceback
        traceback.print_exc()

