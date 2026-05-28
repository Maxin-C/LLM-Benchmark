#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4/A5 压缩版实验脚本
- A4: Judge Ablation - 使用3个模型 × 3种配置 × 25 cases = 225次
- A5: Multi-turn vs Single-turn - 使用3个模型 × 2种模式 × 25 cases = 150次
- 总计：375次调用 ✅
"""

import json
import os
import yaml
import subprocess
import time

# 实验配置
MODELS = ['qwen3-8b', 'qwen3-32b', 'qwen3-235b-a22b']
CASES_PER_MODEL = 25

# A4: Judge 配置
JUDGE_CONFIGS = {
    'original': {
        'temperature': 0.1,
        'top_p': 1.0,
        'description': '原始配置'
    },
    'strict': {
        'temperature': 0.05,
        'top_p': 0.9,
        'description': '更严格'
    },
    'lenient': {
        'temperature': 0.2,
        'top_p': 1.0,
        'description': '更宽松'
    }
}

# A5: 对话模式
DIALOGUE_MODES = {
    'multi-turn': {
        'max_turns': 5,
        'description': '多轮对话'
    },
    'single-turn': {
        'max_turns': 1,
        'description': '单轮问答'
    }
}


def create_judge_config(judge_type, base_config):
    """创建不同的judge配置"""
    config = base_config.copy()
    config['judger']['temperature'] = JUDGE_CONFIGS[judge_type]['temperature']
    config['judger']['top_p'] = JUDGE_CONFIGS[judge_type]['top_p']
    return config


def create_dialogue_config(mode, base_config):
    """创建不同的对话模式配置"""
    config = base_config.copy()
    config['virtual_patient']['max_turns'] = DIALOGUE_MODES[mode]['max_turns']
    return config


def save_config(config, filename):
    """保存配置文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def run_benchmark(config_file, output_file, num_cases=CASES_PER_MODEL):
    """运行基准测试"""
    cmd = [
        'python', 'scripts/run_benchmark.py',
        '--config', config_file,
        '--output', 'outputs/experiments/A4_A5_compressed',
        '--num_cases', str(num_cases),
        '--output_file', output_file,
        '--parallel', '1'
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"✓ 成功完成: {output_file}")
            return True
        else:
            print(f"✗ 失败: {output_file}")
            print(f"错误信息: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ 超时: {output_file}")
        return False


def main():
    print("="*60)
    print("A4/A5 压缩版实验")
    print(f"预计调用次数: {len(MODELS) * len(JUDGE_CONFIGS) * CASES_PER_MODEL + len(MODELS) * len(DIALOGUE_MODES) * CASES_PER_MODEL}")
    print("="*60)
    
    # 创建输出目录
    os.makedirs('outputs/experiments/A4_A5_compressed', exist_ok=True)
    
    # 加载基础配置
    with open('outputs/model_evaluation_50cases/config_qwen3-32b.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    success_count = 0
    total_tasks = len(MODELS) * (len(JUDGE_CONFIGS) + len(DIALOGUE_MODES))
    
    # A4: Judge Ablation
    print("\n--- A4: Judge Ablation 实验 ---")
    for model in MODELS:
        for judge_type in JUDGE_CONFIGS:
            config_name = f'config_{model}_{judge_type}.yaml'
            output_name = f'benchmark_results_{model}_{judge_type}.json'
            
            # 更新模型配置
            base_config['virtual_doctor']['model'] = model
            
            # 创建judge配置
            config = create_judge_config(judge_type, base_config)
            save_config(config, config_name)
            
            # 运行实验
            print(f"\n运行: {model} + {judge_type}")
            if run_benchmark(config_name, output_name):
                success_count += 1
            
            # 清理配置文件
            os.remove(config_name)
            
            # 等待一下避免API限流
            time.sleep(5)
    
    # A5: Multi-turn vs Single-turn
    print("\n--- A5: Multi-turn vs Single-turn 实验 ---")
    for model in MODELS:
        for mode in DIALOGUE_MODES:
            config_name = f'config_{model}_{mode}.yaml'
            output_name = f'benchmark_results_{model}_{mode}.json'
            
            # 更新模型配置
            base_config['virtual_doctor']['model'] = model
            
            # 创建对话模式配置
            config = create_dialogue_config(mode, base_config)
            save_config(config, config_name)
            
            # 运行实验
            print(f"\n运行: {model} + {mode}")
            if run_benchmark(config_name, output_name):
                success_count += 1
            
            # 清理配置文件
            os.remove(config_name)
            
            # 等待一下避免API限流
            time.sleep(5)
    
    # 总结
    print("\n" + "="*60)
    print(f"实验完成: {success_count}/{total_tasks}")
    print(f"输出目录: outputs/experiments/A4_A5_compressed")
    print("="*60)


if __name__ == '__main__':
    main()
