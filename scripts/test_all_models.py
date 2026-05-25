#!/usr/bin/env python3
"""
测试所有待测试模型的可用性
包括：Qwen3系列、GPT-4o、本地Llama3
"""

import os
import sys
import time
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

def test_qwen_models():
    """测试Qwen3系列模型"""
    qwen_models = [
        "qwen3-0.6b",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-32b",
        "qwen3-235b-a22b"
    ]
    
    api_key = os.environ.get("QWEN_API_KEY")
    base_url = os.environ.get("QWEN_BASE_URL")
    
    if not api_key or not base_url:
        print("❌ Qwen API配置未找到")
        return False
    
    print("="*60)
    print("测试Qwen3系列模型")
    print("="*60)
    
    from dashscope import Generation
    
    success_count = 0
    for model in qwen_models:
        try:
            start_time = time.time()
            response = Generation.call(
                model=model,
                prompt="Hello",
                api_key=api_key,
                max_tokens=50,
                temperature=0.1,
                enable_thinking=False
            )
            latency = time.time() - start_time
            if response.status_code == 200:
                print(f"✅ {model}: 响应成功 ({latency:.2f}s)")
                success_count += 1
            else:
                print(f"❌ {model}: {response.message}")
        except Exception as e:
            print(f"❌ {model}: {str(e)[:100]}")
    
    print(f"\nQwen3系列测试完成: {success_count}/{len(qwen_models)} 通过")
    return success_count == len(qwen_models)

def test_gpt4o():
    """测试GPT-4o"""
    print("\n" + "="*60)
    print("测试GPT-4o")
    print("="*60)
    
    api_key = os.environ.get("EASE_DOCTOR_API_KEY")
    base_url = os.environ.get("EASE_DOCTOR_BASE_URL")
    model = os.environ.get("EASE_DOCTOR_MODEL", "gpt-4o")
    
    if not api_key or not base_url:
        print("❌ GPT-4o API配置未找到")
        return False
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个助手"},
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        latency = time.time() - start_time
        print(f"✅ GPT-4o: 响应成功 ({latency:.2f}s)")
        return True
    except Exception as e:
        print(f"❌ GPT-4o: {str(e)[:100]}")
        return False

def test_local_llama3():
    """测试本地Llama3模型"""
    print("\n" + "="*60)
    print("测试本地Llama3模型")
    print("="*60)
    
    model_path = "/mnt/pvc-data.common/ChenZikang/huggingface/shenzhi-wang/Llama3-8B-Chinese-Chat"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    try:
        print("⏳ 加载模型中...")
        start_time = time.time()
        
        import torch
        from accelerate import Accelerator
        
        accelerator = Accelerator()
        print(f"   可用GPU: {accelerator.num_processes}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        model = accelerator.prepare(model)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=0.1,
            device=accelerator.device.index if accelerator.device.type == 'cuda' else -1
        )
        
        response = pipe("Hello")
        latency = time.time() - start_time
        
        print(f"✅ 本地Llama3: 响应成功 ({latency:.2f}s)")
        print(f"   设备: {accelerator.device}")
        return True
    except Exception as e:
        print(f"❌ 本地Llama3: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False

def test_deepseek_models():
    """测试DeepSeek模型（作为系统核心组件）"""
    print("\n" + "="*60)
    print("测试DeepSeek模型")
    print("="*60)
    
    api_key = os.environ.get("EASE_VP_API_KEY")
    base_url = os.environ.get("EASE_VP_BASE_URL")
    
    if not api_key or not base_url:
        print("❌ DeepSeek API配置未找到")
        return False
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 测试deepseek-v4-pro（虚拟患者）
        start_time = time.time()
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "你是一个助手"},
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=50,
            temperature=0.1,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )
        latency = time.time() - start_time
        print(f"✅ deepseek-v4-pro (thinking): 响应成功 ({latency:.2f}s)")
        
        # 测试deepseek-v4-flash（监控器）
        start_time = time.time()
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "你是一个助手"},
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        latency = time.time() - start_time
        print(f"✅ deepseek-v4-flash: 响应成功 ({latency:.2f}s)")
        
        return True
    except Exception as e:
        print(f"❌ DeepSeek: {str(e)[:100]}")
        return False

if __name__ == "__main__":
    print("开始测试所有模型...\n")
    
    results = []
    
    # 测试DeepSeek（核心组件）
    results.append(("DeepSeek", test_deepseek_models()))
    
    # 测试Qwen3系列
    results.append(("Qwen3系列", test_qwen_models()))
    
    # 测试GPT-4o
    results.append(("GPT-4o", test_gpt4o()))
    
    # 测试本地Llama3
    results.append(("本地Llama3", test_local_llama3()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    all_pass = True
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name}: {status}")
        if not success:
            all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("🎉 所有模型测试通过！可以开始正式实验")
        sys.exit(0)
    else:
        print("⚠️ 部分模型测试失败，请检查配置或联系管理员")
        sys.exit(1)