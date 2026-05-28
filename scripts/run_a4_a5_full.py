#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4/A5 完整实验脚本（带进度条）
- A4: Judge Ablation - 3模型 × 3配置 × 25cases = 225次
- A5: Multi-turn vs Single-turn - 3模型 × 2模式 × 25cases = 150次
- 总计：375次调用
"""

import json
import os
import yaml
import subprocess
import time
import sys


def progress_bar(completed, total, bar_length=50):
    """显示进度条"""
    progress = completed / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percentage = int(progress * 100)
    sys.stdout.write(f'\r[{bar}] {percentage}% ({completed}/{total})')
    sys.stdout.flush()


def create_config(base_config, model, judge_temp=None, max_turns=None):
    """创建配置文件"""
    config = base_config.copy()
    config['virtual_doctor']['model'] = model
    
    if judge_temp is not None:
        config['judger']['temperature'] = judge_temp
    
    if max_turns is not None:
        config['virtual_patient']['max_turns'] = max_turns
    
    return config


def save_config(config, filename):
    """保存配置文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def run_benchmark(config_file, output_file, num_cases=25):
    """运行基准测试"""
    cmd = [
        'python', 'scripts/run_benchmark.py',
        '--config', config_file,
        '--output', 'outputs/experiments/A4_A5_full',
        '--num_cases', str(num_cases),
        '--output_file', output_file,
        '--parallel', '1'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main():
    print("="*60)
    print("A4/A5 完整实验（带进度条）")
    print(f"预计调用次数: 375")
    print("="*60)
    
    # 创建输出目录
    os.makedirs('outputs/experiments/A4_A5_full', exist_ok=True)
    
    # 加载基础配置
    with open('outputs/model_evaluation_50cases/config_qwen3-32b.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 实验配置
    models = ['qwen3-8b', 'qwen3-32b', 'qwen3-235b-a22b']
    
    # A4: Judge配置
    judge_configs = [
        {'name': 'original', 'temp': 0.1, 'desc': '原始配置'},
        {'name': 'strict', 'temp': 0.05, 'desc': '更严格'},
        {'name': 'lenient', 'temp': 0.2, 'desc': '更宽松'}
    ]
    
    # A5: 对话模式
    dialogue_modes = [
        {'name': 'multi-turn', 'max_turns': 5, 'desc': '多轮对话'},
        {'name': 'single-turn', 'max_turns': 1, 'desc': '单轮问答'}
    ]
    
    # 计算总任务数
    total_tasks = len(models) * len(judge_configs) + len(models) * len(dialogue_modes)
    completed_tasks = 0
    
    print("\n--- A4: Judge Ablation 实验 ---")
    for model in models:
        for judge in judge_configs:
            config_name = f'config_{model}_{judge["name"]}.yaml'
            output_name = f'benchmark_results_{model}_{judge["name"]}.json'
            
            # 创建配置
            config = create_config(base_config, model, judge_temp=judge['temp'])
            save_config(config, config_name)
            
            # 运行实验
            print(f"\n运行: {model} + {judge['desc']}")
            success = run_benchmark(config_name, output_name)
            
            # 更新进度
            completed_tasks += 1
            progress_bar(completed_tasks, total_tasks)
            
            # 清理配置文件
            os.remove(config_name)
            
            # 等待避免限流
            time.sleep(2)
    
    print("\n\n--- A5: Multi-turn vs Single-turn 实验 ---")
    for model in models:
        for mode in dialogue_modes:
            config_name = f'config_{model}_{mode["name"]}.yaml'
            output_name = f'benchmark_results_{model}_{mode["name"]}.json'
            
            # 创建配置
            config = create_config(base_config, model, max_turns=mode['max_turns'])
            save_config(config, config_name)
            
            # 运行实验
            print(f"\n运行: {model} + {mode['desc']}")
            success = run_benchmark(config_name, output_name)
            
            # 更新进度
            completed_tasks += 1
            progress_bar(completed_tasks, total_tasks)
            
            # 清理配置文件
            os.remove(config_name)
            
            # 等待避免限流
            time.sleep(2)
    
    # 总结
    print("\n\n" + "="*60)
    print(f"实验完成: {completed_tasks}/{total_tasks}")
    print(f"输出目录: outputs/experiments/A4_A5_full")
    print("="*60)


if __name__ == '__main__':
    main()
