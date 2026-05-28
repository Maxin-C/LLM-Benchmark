#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A5 完整实验：Multi-turn vs Single-turn 对比

根据实验设计文档，本实验将对同一批案例分别进行：
- Static setting：单轮问答模式
- Dynamic setting：多轮对话模式

使用 Qwen3-32b 作为虚拟医生
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


def main():
    print("="*60)
    print("A5 完整实验：Multi-turn vs Single-turn")
    print("虚拟医生: Qwen3-32b")
    print("案例数: 25")
    print("预计API调用: 约200次")
    print("="*60)
    
    # 创建输出目录
    os.makedirs('outputs/experiments/A5_full', exist_ok=True)
    
    # 加载基础配置
    with open('outputs/model_evaluation_50cases/config_qwen3-32b.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 配置虚拟医生为Qwen3-32b
    base_config['virtual_doctor']['model'] = 'qwen3-32b'
    
    # 实验配置
    modes = [
        {'name': 'single-turn', 'max_turns': 1, 'desc': '单轮问答'},
        {'name': 'multi-turn', 'max_turns': 5, 'desc': '多轮对话'}
    ]
    
    total_tasks = 2  # single-turn + multi-turn
    completed_tasks = 0
    
    for mode in modes:
        # 创建配置文件
        config = base_config.copy()
        config['virtual_patient']['max_turns'] = mode['max_turns']
        
        config_file = f'config_a5_{mode["name"]}.yaml'
        output_file = f'benchmark_results_qwen3-32b_{mode["name"]}.json'
        
        # 保存配置
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 运行实验
        print(f"\n运行 {mode['desc']} 模式...")
        print(f"配置: max_turns = {mode['max_turns']}")
        
        cmd = [
            'python', 'scripts/run_benchmark.py',
            '--config', config_file,
            '--output', 'outputs/experiments/A5_full',
            '--num_cases', '25',
            '--output_file', output_file,
            '--parallel', '1'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                print(f"✓ {mode['desc']} 完成")
            else:
                print(f"✗ {mode['desc']} 失败")
                print(f"错误: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            print(f"✗ {mode['desc']} 超时")
        
        # 更新进度
        completed_tasks += 1
        progress_bar(completed_tasks, total_tasks)
        
        # 清理配置文件
        os.remove(config_file)
        
        # 等待避免限流
        time.sleep(5)
    
    print("\n\n" + "="*60)
    print("A5 实验完成！")
    print(f"输出目录: outputs/experiments/A5_full")
    print("="*60)
    
    # 分析结果
    analyze_results()


def analyze_results():
    """分析单轮vs多轮的对比结果"""
    output_dir = 'outputs/experiments/A5_full'
    
    # 加载结果
    results = {}
    for mode in ['single-turn', 'multi-turn']:
        file_path = os.path.join(output_dir, f'benchmark_results_qwen3-32b_{mode}.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                results[mode] = json.load(f)
            print(f"✓ 加载 {mode}: {len(results[mode])} cases")
    
    if len(results) != 2:
        print("警告: 未找到完整的对比数据")
        return
    
    # 计算统计数据
    stats = {}
    for mode, data in results.items():
        scores = [case['evaluation']['overall_score'] for case in data]
        stats[mode] = {
            'mean': sum(scores) / len(scores),
            'std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
            'min': min(scores),
            'max': max(scores),
            'pass_rate': sum(1 for s in scores if s >= 3) / len(scores) * 100
        }
    
    # 生成报告
    report_lines = []
    report_lines.append("# A5 实验：Multi-turn vs Single-turn 对比报告")
    report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("虚拟医生: Qwen3-32b")
    report_lines.append("案例数: 25")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 对比结果")
    report_lines.append("")
    report_lines.append("| 指标 | 单轮对话 | 多轮对话 | 差异 |")
    report_lines.append("|------|----------|----------|------|")
    report_lines.append(f"| 平均分 | {stats['single-turn']['mean']:.2f} | {stats['multi-turn']['mean']:.2f} | {(stats['multi-turn']['mean'] - stats['single-turn']['mean']):.2f} |")
    report_lines.append(f"| 标准差 | {stats['single-turn']['std']:.2f} | {stats['multi-turn']['std']:.2f} | {(stats['multi-turn']['std'] - stats['single-turn']['std']):.2f} |")
    report_lines.append(f"| 最低分 | {stats['single-turn']['min']:.2f} | {stats['multi-turn']['min']:.2f} | {(stats['multi-turn']['min'] - stats['single-turn']['min']):.2f} |")
    report_lines.append(f"| 最高分 | {stats['single-turn']['max']:.2f} | {stats['multi-turn']['max']:.2f} | {(stats['multi-turn']['max'] - stats['single-turn']['max']):.2f} |")
    report_lines.append(f"| 通过率 | {stats['single-turn']['pass_rate']:.1f}% | {stats['multi-turn']['pass_rate']:.1f}% | {(stats['multi-turn']['pass_rate'] - stats['single-turn']['pass_rate']):.1f}% |")
    report_lines.append("")
    report_lines.append("## 结论")
    
    if stats['multi-turn']['mean'] > stats['single-turn']['mean']:
        report_lines.append("- ✅ 多轮对话模式表现优于单轮对话模式")
    else:
        report_lines.append("- ❌ 单轮对话模式表现优于多轮对话模式")
    
    report_lines.append(f"- 评分差异: {abs(stats['multi-turn']['mean'] - stats['single-turn']['mean']):.2f} 分")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'a5_comparison_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ 生成对比报告: {report_file}")


if __name__ == '__main__':
    main()
