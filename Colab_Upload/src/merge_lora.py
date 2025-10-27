#!/usr/bin/env python3
"""
合併 LoRA adapter 與基礎模型

使用方法:
    python src/merge_lora.py
    python src/merge_lora.py --device_map cpu  # 記憶體不足時使用
"""

import argparse
import json
import os
import gc

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """主函數"""
    # 設定離線模式
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 解析參數
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="./scripts/grpo_trainer_lora_model",
        help="Path to the LoRA adapter"
    )
    parser.add_argument(
        "--save_path", 
        type=str,
        default="./LegalDelta/Qwen2.5-3B-merge",
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--base_model_path", 
        type=str,
        default="./Qwen2.5-3B-Instruct", 
        help="Path to base model"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map (auto/cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    # 標準化路徑
    if args.model_name_or_path.startswith('./'):
        args.model_name_or_path = args.model_name_or_path[2:]
    
    print("=" * 60)
    print("LoRA 模型合併工具")
    print("=" * 60)
    print(f"\nLoRA adapter: {args.model_name_or_path}")
    print(f"Base model: {args.base_model_path}")
    print(f"Save to: {args.save_path}")
    print(f"Device: {args.device_map}\n")
    
    # 驗證路徑
    if not os.path.exists(args.model_name_or_path):
        raise FileNotFoundError(f"LoRA adapter not found: {args.model_name_or_path}")
    
    # 載入配置
    print("📋 Loading LoRA config...")
    try:
        config = PeftConfig.from_pretrained(args.model_name_or_path, local_files_only=True)
    except Exception:
        # Fallback: 直接從 JSON 載入
        config_path = os.path.join(args.model_name_or_path, "adapter_config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        from peft import LoraConfig
        config = LoraConfig(**config_dict)
    
    # 確定基礎模型路徑
    base_model_path = args.base_model_path
    if not os.path.exists(base_model_path):
        # 嘗試從配置中獲取
        base_model_path = config.base_model_name_or_path
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
    
    # 載入 tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    # 載入基礎模型
    print("🧠 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    print("   ✓ Base model loaded")
    
    # 載入 LoRA adapter
    print("🔧 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        args.model_name_or_path,
        local_files_only=True,
        is_trainable=False
    )
    print("   ✓ LoRA adapter loaded")
    
    # 合併
    print("🔄 Merging LoRA with base model...")
    model = model.merge_and_unload()
    print("   ✓ Merge completed")
    
    # 清理記憶體
    print("🧹 Cleaning memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 儲存
    print(f"💾 Saving merged model to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)
    
    # 移動到 CPU 以節省記憶體
    model = model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 儲存模型
    model.save_pretrained(
        args.save_path,
        max_shard_size="1GB",
        safe_serialization=True
    )
    tokenizer.save_pretrained(args.save_path)
    
    # 最終清理
    del model
    del tokenizer
    gc.collect()
    
    print("\n" + "=" * 60)
    print("✅ Model merged successfully!")
    print(f"✅ Saved to: {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tips:")
        print("   - If out of memory: python src/merge_lora.py --device_map cpu")
        print("   - Check paths are correct")
        raise
