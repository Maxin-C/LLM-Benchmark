#!/usr/bin/env python3
"""
模型区分度评估脚本 - 测试不同大小模型的性能差异

正确的角色分配：
- virtual_patient: deepseek-v4-pro (thinking模式) - 固定
- virtual_doctor: 待测试模型 (Qwen3系列等) - 可变
- judger: deepseek-v4-pro - 固定
- dialogue_monitor: deepseek-v4-pro - 固定

功能：
- 进度条显示评估进度
- 断点保存（每评估完一个模型保存一次）
- 支持从断点继续评估
- 生成最终评估报告和性能对比图
"""

import argparse
import json
import os
import sys
import time
import subprocess
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 需要评估的模型列表（作为virtual doctor）
MODELS_TO_EVALUATE = [
    {
        "name": "qwen3-0.6b",
        "type": "qwen",
        "model_name": "qwen3-0.6b"
    },
    {
        "name": "qwen3-8b",
        "type": "qwen",
        "model_name": "qwen3-8b"
    },
    {
        "name": "qwen3-14b",
        "type": "qwen",
        "model_name": "qwen3-14b"
    },
    {
        "name": "qwen3-32b",
        "type": "qwen",
        "model_name": "qwen3-32b"
    },
    {
        "name": "qwen3-235b-a22b",
        "type": "qwen",
        "model_name": "qwen3-235b-a22b"
    },
    {
        "name": "gpt-4o",
        "type": "openai",
        "model_name": "gpt-4o"
    },
    {
        "name": "llama3-8b-chinese",
        "type": "local",
        "model_path": "/mnt/pvc-data.common/ChenZikang/huggingface/shenzhi-wang/Llama3-8B-Chinese-Chat"
    }
]

def load_checkpoint(output_dir: str) -> dict:
    """加载断点检查点"""
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载检查点失败: {e}")
            return {}
    return {}

def save_checkpoint(output_dir: str, completed_models: list, results: dict):
    """保存断点检查点"""
    checkpoint = {
        "completed_models": completed_models,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

def create_model_config(model_info: dict) -> str:
    """
    创建模型特定的配置文件
    
    角色分配：
    - virtual_patient: deepseek-v4-pro (thinking模式，固定)
    - virtual_doctor: 待测试模型
    - judger: deepseek-v4-pro (固定)
    - dialogue_monitor: deepseek-v4-pro (固定)
    """
    model_type = model_info['type']
    
    # DeepSeek官方API配置（用于virtual_patient, judger, monitor）
    deepseek_api_key = os.environ.get('EASE_VP_API_KEY', '')
    deepseek_base_url = os.environ.get('EASE_VP_BASE_URL', '')
    
    # Qwen API配置（用于virtual_doctor）
    qwen_api_key = os.environ.get('QWEN_API_KEY', '')
    qwen_base_url = os.environ.get('QWEN_BASE_URL', '')
    
    # 待测试模型（virtual_doctor）的配置
    if model_type == 'qwen':
        doctor_api_key = qwen_api_key
        doctor_base_url = qwen_base_url
        doctor_model = model_info['model_name']
    elif model_type == 'openai':
        # GPT-4o使用EASE_LLM配置
        doctor_api_key = os.environ.get('EASE_LLM_API_KEY', '')
        doctor_base_url = os.environ.get('EASE_LLM_BASE_URL', '')
        doctor_model = model_info['model_name']
    elif model_type == 'local':
        # 本地模型配置
        doctor_model = model_info['model_path']
        # 本地模型需要特殊的配置格式
        return f"""
virtual_patient:
  api:
    api_key: {deepseek_api_key}
    base_url: {deepseek_base_url}
  model: deepseek-v4-pro
  temperature: 0.8
  top_p: 0.9
  thinking_enabled: true
  reasoning_effort: high

virtual_doctor:
  model_type: local
  local:
    model_path: {doctor_model}
  temperature: 0.8
  top_p: 0.9

judger:
  api:
    api_key: {deepseek_api_key}
    base_url: {deepseek_base_url}
  model: deepseek-v4-pro
  temperature: 0.1
  top_p: 1.0

dialogue_monitor:
  api:
    api_key: {deepseek_api_key}
    base_url: {deepseek_base_url}
  model: deepseek-v4-pro
  temperature: 0.1
  max_tokens: 10
"""
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    return f"""
virtual_patient:
  api:
    api_key: {deepseek_api_key}
    base_url: {deepseek_base_url}
  model: deepseek-v4-pro
  temperature: 0.8
  top_p: 0.9
  thinking_enabled: true
  reasoning_effort: high

virtual_doctor:
  api:
    api_key: {doctor_api_key}
    base_url: {doctor_base_url}
  model: {doctor_model}
  temperature: 0.8
  top_p: 0.9

judger:
  api:
    api_key: {deepseek_api_key}
    base_url: {deepseek_base_url}
  model: deepseek-v4-pro
  temperature: 0.1
  top_p: 1.0

dialogue_monitor:
  api:
    api_key: {deepseek_api_key}
    base_url: {deepseek_base_url}
  model: deepseek-v4-pro
  temperature: 0.1
  max_tokens: 10
"""

def evaluate_model(model_info: dict, num_cases: int, output_dir: str) -> dict:
    """评估单个模型"""
    model_name = model_info['name']
    
    print(f"\n🚀 开始评估模型: {model_name}")
    print("-" * 60)
    
    # 创建模型特定的配置文件（覆盖写入）
    config_content = create_model_config(model_info)
    config_path = os.path.join(output_dir, f'config_{model_name}.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    # 运行基准测试
    output_file = f'benchmark_results_{model_name}.json'
    
    cmd = [
        sys.executable, 'scripts/run_benchmark.py',
        '--config', config_path,
        '--output', output_dir,
        '--num_cases', str(num_cases),
        '--output_file', output_file
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # 运行子进程（50个cases需要更长超时时间：8小时）
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=28800  # 8小时超时（50个cases）
        )
        
        duration = time.time() - start_time
        
        # 读取评估结果
        result_path = os.path.join(output_dir, output_file)
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)
            
            # benchmark_data 是一个列表，每个元素是一个案例的结果
            if isinstance(benchmark_data, list):
                # 计算平均分数和通过率
                total_score = 0
                passed_count = 0
                case_results = []
                
                for i, case in enumerate(benchmark_data):
                    evaluation = case.get('evaluation', {})
                    overall_score = evaluation.get('overall_score', 0)
                    is_passed = evaluation.get('is_passed', False)
                    
                    total_score += overall_score
                    if is_passed:
                        passed_count += 1
                    
                    case_results.append({
                        "case_id": i + 1,
                        "overall_score": overall_score,
                        "is_passed": is_passed,
                        "duration": case.get('duration', 0)
                    })
                
                avg_score = total_score / len(benchmark_data) if benchmark_data else 0
                pass_rate = (passed_count / len(benchmark_data)) * 100 if benchmark_data else 0
                
                return {
                    "model_name": model_name,
                    "model_type": model_info['type'],
                    "num_cases": num_cases,
                    "average_score": avg_score,
                    "pass_rate": pass_rate,
                    "duration": duration,
                    "completed": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "case_results": case_results
                }
            else:
                return {
                    "model_name": model_name,
                    "model_type": model_info['type'],
                    "num_cases": num_cases,
                    "average_score": 0,
                    "pass_rate": 0,
                    "duration": duration,
                    "completed": False,
                    "error": "结果格式错误（不是列表）"
                }
        else:
            return {
                "model_name": model_name,
                "model_type": model_info['type'],
                "num_cases": num_cases,
                "average_score": 0,
                "pass_rate": 0,
                "duration": duration,
                "completed": False,
                "error": "结果文件未生成"
            }
    
    except subprocess.TimeoutExpired:
        # 超时时检查是否已有部分结果
        result_path = os.path.join(output_dir, output_file)
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)
                
                if isinstance(benchmark_data, list) and len(benchmark_data) > 0:
                    # 计算已有结果
                    total_score = sum(case.get('evaluation', {}).get('overall_score', 0) for case in benchmark_data)
                    passed_count = sum(1 for case in benchmark_data if case.get('evaluation', {}).get('is_passed', False))
                    avg_score = total_score / len(benchmark_data)
                    pass_rate = (passed_count / len(benchmark_data)) * 100
                    
                    print(f"⚠️ 评估超时，但已保存 {len(benchmark_data)}/{num_cases} 个案例的结果")
                    return {
                        "model_name": model_name,
                        "model_type": model_info['type'],
                        "num_cases": len(benchmark_data),
                        "average_score": avg_score,
                        "pass_rate": pass_rate,
                        "duration": time.time() - start_time,
                        "completed": False,
                        "partial": True,
                        "error": f"评估超时（仅完成{len(benchmark_data)}/{num_cases}个案例）"
                    }
            except:
                pass
        
        return {
            "model_name": model_name,
            "model_type": model_info['type'],
            "num_cases": num_cases,
            "average_score": 0,
            "pass_rate": 0,
            "duration": time.time() - start_time,
            "completed": False,
            "error": "评估超时"
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "model_type": model_info['type'],
            "num_cases": num_cases,
            "average_score": 0,
            "pass_rate": 0,
            "duration": time.time() - start_time,
            "completed": False,
            "error": str(e)
        }

def generate_final_report(results: dict, output_dir: str):
    """生成最终评估报告"""
    report_path = os.path.join(output_dir, 'benchmark_summary.json')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 评估报告已保存到: {report_path}")
    
    # 打印汇总表格
    print("\n" + "=" * 80)
    print("模型区分度评估汇总报告")
    print("=" * 80)
    print(f"{'模型名称':<20} {'平均分数':<10} {'通过率':<10} {'用时(秒)':<10} {'状态':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        avg_score = result.get('average_score', 0)
        pass_rate = result.get('pass_rate', 0)
        duration = result.get('duration', 0)
        completed = result.get('completed', False)
        status = "✅ 完成" if completed else "❌ 失败"
        
        print(f"{model_name:<20} {avg_score:<10.2f} {pass_rate:<10.1f}% {duration:<10.1f} {status:<10}")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='模型区分度评估脚本')
    parser.add_argument('--num_cases', type=int, default=3,
                        help='每个模型评估的案例数量（默认: 3）')
    parser.add_argument('--output_dir', type=str, default='outputs/model_evaluation',
                        help='输出目录（默认: outputs/model_evaluation）')
    parser.add_argument('--models', type=str, nargs='+',
                        help='指定要评估的模型名称（默认评估所有模型）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载检查点
    checkpoint = load_checkpoint(output_dir)
    completed_models = checkpoint.get('completed_models', [])
    results = checkpoint.get('results', {})
    
    print("=" * 60)
    print("    模型区分度评估 - 进度恢复")
    print("=" * 60)
    print(f"已完成评估的模型: {len(completed_models)}/{len(MODELS_TO_EVALUATE)}")
    if completed_models:
        print(f"已完成: {', '.join(completed_models)}")
    print("-" * 60)
    
    # 确定要评估的模型
    if args.models:
        models_to_evaluate = [m for m in MODELS_TO_EVALUATE if m['name'] in args.models]
    else:
        models_to_evaluate = MODELS_TO_EVALUATE
    
    # 使用tqdm显示整体进度
    with tqdm(total=len(models_to_evaluate), desc="整体进度", unit="模型") as pbar:
        for model_info in models_to_evaluate:
            model_name = model_info['name']
            
            # 跳过已完成的模型
            if model_name in completed_models:
                print(f"\n⏭️ 跳过已完成的模型: {model_name}")
                pbar.update(1)
                continue
            
            # 评估模型
            result = evaluate_model(model_info, args.num_cases, output_dir)
            results[model_name] = result
            
            if result['completed']:
                completed_models.append(model_name)
                print(f"\n✅ {model_name} 评估完成")
                print(f"   平均分数: {result['average_score']:.2f}/5")
                print(f"   通过率: {result['pass_rate']:.1f}%")
                print(f"   用时: {result['duration']:.2f}秒")
            else:
                print(f"\n❌ {model_name} 评估失败: {result.get('error', '未知错误')}")
            
            # 保存断点
            save_checkpoint(output_dir, completed_models, results)
            
            pbar.update(1)
    
    # 生成最终报告
    generate_final_report(results, output_dir)
    
    print("\n🎉 所有模型评估完成！")

if __name__ == '__main__':
    main()
