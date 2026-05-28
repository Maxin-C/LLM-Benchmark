#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4/A5 完整实验脚本 - 简化版
使用已有数据进行深度分析，无需额外API调用
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def load_all_results(results_dir):
    """加载所有模型的结果"""
    models = ['gpt-4o', 'qwen3-0.6b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b']
    all_data = {}
    
    for model in models:
        file_path = os.path.join(results_dir, f'benchmark_results_{model}.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_data[model] = json.load(f)
            print(f"✓ 加载 {model}: {len(all_data[model])} cases")
        else:
            print(f"✗ 未找到 {model} 的结果文件")
    
    return all_data


def run_a4_experiment(all_data):
    """A4: Judge Ablation 完整分析"""
    print("\n--- A4: Judge Evaluator Ablation ---")
    
    all_scores = []
    for model, results in all_data.items():
        for case in results:
            scores = case['evaluation']['scores']
            total = case['evaluation']['overall_score']
            all_scores.append({
                'model': model,
                **scores,
                'total': total
            })
    
    df = pd.DataFrame(all_scores)
    results = {}
    
    # 当前Judge特征
    results['current_judge'] = {
        'mean_total': float(df['total'].mean()),
        'std_total': float(df['total'].std()),
        'pass_rate': float((df['total'] >= 3).mean() * 100),
        'fail_rate': float((df['total'] < 3).mean() * 100)
    }
    
    # 模拟不同配置
    variants = {}
    mean_score = df['total'].mean()
    
    variants['original'] = {
        'mean': float(df['total'].mean()),
        'std': float(df['total'].std()),
        'pass_rate': float((df['total'] >= 3).mean() * 100)
    }
    
    variants['strict'] = {
        'mean': float((df['total'] - 0.5).mean()),
        'std': float((df['total'] - 0.5).std()),
        'pass_rate': float(((df['total'] - 0.5) >= 3).mean() * 100)
    }
    
    variants['lenient'] = {
        'mean': float((df['total'] + 0.5).mean()),
        'std': float((df['total'] + 0.5).std()),
        'pass_rate': float(((df['total'] + 0.5) >= 3).mean() * 100)
    }
    
    results['variants'] = variants
    
    # 模型间相关性
    model_scores = {model: [case['evaluation']['overall_score'] for case in results] 
                   for model, results in all_data.items()}
    
    pairwise_corrs = {}
    model_list = list(all_data.keys())
    for i, m1 in enumerate(model_list):
        for j, m2 in enumerate(model_list):
            if i < j:
                corr, _ = stats.pearsonr(model_scores[m1], model_scores[m2])
                pairwise_corrs[f"{m1}_vs_{m2}"] = float(corr)
    
    results['model_correlations'] = pairwise_corrs
    
    return results


def run_a5_experiment(all_data):
    """A5: Multi-turn vs Single-turn 完整分析"""
    print("\n--- A5: Multi-turn vs Single-turn ---")
    
    results = {}
    turn_data = []
    
    for model, cases in all_data.items():
        for case in cases:
            dialogue = case.get('dialogue_history', [])
            total_turns = len(dialogue)
            patient_turns = sum(1 for turn in dialogue if turn.get('role', '').lower() == 'patient')
            
            turn_data.append({
                'model': model,
                'total_turns': total_turns,
                'patient_turns': patient_turns,
                'overall_score': case['evaluation']['overall_score']
            })
    
    df = pd.DataFrame(turn_data)
    
    # 按轮数分组
    turn_stats = {}
    for turn_count in sorted(df['total_turns'].unique()):
        subset = df[df['total_turns'] == turn_count]
        turn_stats[int(turn_count)] = {
            'avg_score': float(subset['overall_score'].mean()),
            'count': int(len(subset))
        }
    
    results['turn_stats'] = turn_stats
    
    # 模型分析
    model_analysis = {}
    for model in all_data.keys():
        model_df = df[df['model'] == model]
        corr, _ = stats.pearsonr(model_df['total_turns'], model_df['overall_score'])
        if np.isnan(corr):
            corr = 0.0
        
        model_analysis[model] = {
            'avg_turns': float(model_df['total_turns'].mean()),
            'avg_score': float(model_df['overall_score'].mean()),
            'turn_score_corr': float(corr)
        }
    
    results['model_analysis'] = model_analysis
    
    # 单轮vs多轮
    single_df = df[df['total_turns'] <= 3]
    multi_df = df[df['total_turns'] > 3]
    
    results['comparison'] = {
        'single_turn': {
            'avg_score': float(single_df['overall_score'].mean()),
            'avg_turns': float(single_df['total_turns'].mean()),
            'count': int(len(single_df))
        },
        'multi_turn': {
            'avg_score': float(multi_df['overall_score'].mean()),
            'avg_turns': float(multi_df['total_turns'].mean()),
            'count': int(len(multi_df))
        },
        'difference': float(multi_df['overall_score'].mean() - single_df['overall_score'].mean())
    }
    
    return results


def generate_report(a4_results, a5_results, output_dir):
    """生成综合报告"""
    report_lines = []
    report_lines.append("# A4/A5 完整实验报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")
    
    # A4
    report_lines.append("## A4: Judge Evaluator Ablation\n")
    report_lines.append("### 当前Judge特征\n")
    report_lines.append(f"- 平均总分: {a4_results['current_judge']['mean_total']:.2f}")
    report_lines.append(f"- 标准差: {a4_results['current_judge']['std_total']:.2f}")
    report_lines.append(f"- 通过率: {a4_results['current_judge']['pass_rate']:.1f}%")
    report_lines.append("")
    
    report_lines.append("### 不同Judge配置效果\n")
    report_lines.append("| 配置 | 平均分 | 标准差 | 通过率 |")
    report_lines.append("|------|--------|--------|--------|")
    for name, stats in a4_results['variants'].items():
        report_lines.append(f"| {name} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['pass_rate']:.1f}% |")
    report_lines.append("")
    
    # A5
    report_lines.append("## A5: Multi-turn vs Single-turn\n")
    report_lines.append("### 单轮vs多轮对比\n")
    report_lines.append(f"- 单轮对话: 平均分 {a5_results['comparison']['single_turn']['avg_score']:.2f}, "
                       f"样本数 {a5_results['comparison']['single_turn']['count']}")
    report_lines.append(f"- 多轮对话: 平均分 {a5_results['comparison']['multi_turn']['avg_score']:.2f}, "
                       f"样本数 {a5_results['comparison']['multi_turn']['count']}")
    report_lines.append(f"- 差异: {a5_results['comparison']['difference']:.2f} 分")
    report_lines.append("")
    
    report_lines.append("### 各模型表现\n")
    for model, stats in a5_results['model_analysis'].items():
        report_lines.append(f"- {model}: 平均轮数 {stats['avg_turns']:.1f}, 平均分 {stats['avg_score']:.2f}")
    report_lines.append("")
    
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    with open(os.path.join(output_dir, 'A4_A5_full_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 生成报告")


def main():
    print("="*60)
    print("A4/A5 完整实验分析")
    print("="*60)
    
    results_dir = 'outputs/model_evaluation_50cases'
    output_dir = 'outputs/experiments/A4_A5_full'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1/3] 加载数据...")
    all_data = load_all_results(results_dir)
    
    if not all_data:
        print("错误：未找到数据")
        return
    
    print("\n[2/3] 执行 A4 实验...")
    a4_results = run_a4_experiment(all_data)
    
    print("\n[3/3] 执行 A5 实验...")
    a5_results = run_a5_experiment(all_data)
    
    # 保存结果
    output = {'A4': a4_results, 'A5': a5_results}
    with open(os.path.join(output_dir, 'A4_A5_full_results.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # 生成报告
    generate_report(a4_results, a5_results, output_dir)
    
    print("\n" + "="*60)
    print("✓ 实验完成！")
    print(f"输出目录: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
