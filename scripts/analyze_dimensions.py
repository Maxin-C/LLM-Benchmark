#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 实验：Five-dimensional score 分解分析

基于已有评估数据，按 5 个评分维度分解分析模型表现
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi

# 定义 5 个维度
DIMENSIONS = ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']
DIMENSION_LABELS = {
    'accuracy': 'Accuracy\n(临床准确性)',
    'effectiveness': 'Effectiveness\n(临床有效性)',
    'safety': 'Safety\n(安全性)',
    'personalization': 'Personalization\n(个体化)',
    'empathy': 'Empathy\n(共情沟通)'
}


def load_all_results(results_dir):
    """加载所有模型的结果"""
    models = ['gpt-4o', 'qwen3-0.6b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b']
    all_data = {}
    
    for model in models:
        if model == 'gpt-4o':
            file_path = os.path.join(results_dir, f'benchmark_results_{model}.json')
        else:
            file_path = os.path.join(results_dir, f'benchmark_results_{model}.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_data[model] = json.load(f)
            print(f"✓ 加载 {model}: {len(all_data[model])} cases")
        else:
            print(f"✗ 未找到 {model} 的结果文件")
    
    return all_data


def extract_dimension_scores(all_data):
    """提取各维度评分"""
    dimension_stats = {}
    
    for model, results in all_data.items():
        dimension_stats[model] = {dim: [] for dim in DIMENSIONS}
        
        for case in results:
            evaluation = case.get('evaluation', {})
            scores = evaluation.get('scores', {})
            
            for dim in DIMENSIONS:
                score = scores.get(dim, 0)
                dimension_stats[model][dim].append(score)
    
    return dimension_stats


def calculate_statistics(dimension_stats):
    """计算各维度统计指标"""
    stats = {}
    
    for model, dimensions in dimension_stats.items():
        stats[model] = {}
        for dim in DIMENSIONS:
            scores = dimensions[dim]
            if scores:
                stats[model][dim] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores))
                }
    
    return stats


def create_radar_chart(stats, output_dir):
    """生成雷达图"""
    models = list(stats.keys())
    categories = DIMENSIONS
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 颜色映射
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # 绘制每个模型的雷达图
    for i, model in enumerate(models):
        values = [stats[model][dim]['mean'] for dim in categories]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model.replace('qwen3-', 'Qwen3-').upper(), 
                color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    # 设置标签
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), [DIMENSION_LABELS.get(cat, cat) for cat in categories],
                      fontsize=11, fontweight='bold')
    
    # 设置网格
    ax.set_rgrids([1, 2, 3, 4, 5], angle=0, fontsize=10)
    ax.set_ylim(0, 5)
    
    # 添加标题
    plt.title('Model Performance Comparison\n(五维评分雷达图)', 
              size=16, fontweight='bold', pad=30)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # 保存图片
    output_file = os.path.join(output_dir, 'dimension_radar.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存雷达图到：{output_file}")
    
    plt.close()


def create_bar_chart(stats, output_dir):
    """生成柱状对比图"""
    models = list(stats.keys())
    categories = DIMENSIONS
    
    # 准备数据
    x = np.arange(len(categories))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    for i, model in enumerate(models):
        values = [stats[model][dim]['mean'] for dim in categories]
        ax.bar(x + i * width, values, width, label=model.replace('qwen3-', 'Qwen3-').upper(),
               color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
    
    # 设置标签
    ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance by Dimension\n(五维评分柱状对比)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([DIMENSION_LABELS.get(cat, cat).replace('\n', ' ') for cat in categories],
                       fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, model in enumerate(models):
        values = [stats[model][dim]['mean'] for dim in categories]
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(output_dir, 'dimension_bar.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存柱状图到：{output_file}")
    
    plt.close()


def create_stacked_bar_chart(stats, output_dir):
    """生成堆叠柱状图（显示各维度贡献）"""
    models = list(stats.keys())
    
    # 准备数据
    model_names = [m.replace('qwen3-', 'Qwen3-').upper() for m in models]
    dimensions = DIMENSIONS
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    bottom = np.zeros(len(models))
    
    for i, dim in enumerate(dimensions):
        values = [stats[model][dim]['mean'] for model in models]
        ax.bar(model_names, values, bottom=bottom, label=DIMENSION_LABELS.get(dim, dim).replace('\n', ' '),
               color=colors[i], edgecolor='black', linewidth=0.5)
        bottom += values
    
    # 设置
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Dimension Contribution by Model\n(各维度分数贡献)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 27)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(output_dir, 'dimension_stacked_bar.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存堆叠柱状图到：{output_file}")
    
    plt.close()


def generate_summary_table(stats, output_dir):
    """生成维度统计表"""
    models = list(stats.keys())
    
    # 准备数据
    data = []
    for model in models:
        row = {'Model': model}
        for dim in DIMENSIONS:
            row[f'{dim}_mean'] = f"{stats[model][dim]['mean']:.2f}"
            row[f'{dim}_std'] = f"{stats[model][dim]['std']:.2f}"
        data.append(row)
    
    # 保存为 JSON
    results = {
        'dimensions': DIMENSIONS,
        'models': models,
        'statistics': stats
    }
    
    output_file = os.path.join(output_dir, 'dimension_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存维度统计结果到：{output_file}")
    
    return data


def generate_summary_report(stats, output_dir):
    """生成文字总结报告"""
    models = list(stats.keys())
    
    report_lines = []
    report_lines.append("# A2 实验：Five-dimensional score 分解分析报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**数据源**: 6 个模型 × 50 cases\n")
    report_lines.append(f"**评分维度**: {', '.join(DIMENSIONS)}\n")
    report_lines.append("---\n")
    
    # 1. 各维度最佳模型
    report_lines.append("## 1. 各维度表现最佳模型\n")
    for dim in DIMENSIONS:
        best_model = max(models, key=lambda m: stats[m][dim]['mean'])
        worst_model = min(models, key=lambda m: stats[m][dim]['mean'])
        report_lines.append(f"- **{DIMENSION_LABELS.get(dim, dim)}**: "
                          f"最佳 {best_model.replace('qwen3-', 'Qwen3-').upper()} "
                          f"({stats[best_model][dim]['mean']:.2f}), "
                          f"最差 {worst_model.replace('qwen3-', 'Qwen3-').upper()} "
                          f"({stats[worst_model][dim]['mean']:.2f})")
    report_lines.append("")
    
    # 2. 模型优势维度分析
    report_lines.append("## 2. 模型优势维度分析\n")
    for model in models:
        # 找出最强和最弱维度
        dim_means = {dim: stats[model][dim]['mean'] for dim in DIMENSIONS}
        strongest_dim = max(dim_means, key=dim_means.get)
        weakest_dim = min(dim_means, key=dim_means.get)
        
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                          f"最强 {DIMENSION_LABELS.get(strongest_dim, strongest_dim)} "
                          f"({dim_means[strongest_dim]:.2f}), "
                          f"最弱 {DIMENSION_LABELS.get(weakest_dim, weakest_dim)} "
                          f"({dim_means[weakest_dim]:.2f})")
    report_lines.append("")
    
    # 3. 维度难度分析
    report_lines.append("## 3. 维度难度分析（所有模型平均）\n")
    dim_averages = {}
    for dim in DIMENSIONS:
        avg = np.mean([stats[model][dim]['mean'] for model in models])
        dim_averages[dim] = avg
    
    sorted_dims = sorted(dim_averages.items(), key=lambda x: x[1], reverse=True)
    for dim, avg in sorted_dims:
        report_lines.append(f"- **{DIMENSION_LABELS.get(dim, dim)}**: {avg:.2f} 分")
    report_lines.append("")
    
    # 4. 关键发现
    report_lines.append("## 4. 关键发现\n")
    
    # 找出整体最佳模型
    overall_best = max(models, key=lambda m: np.mean([stats[m][dim]['mean'] for dim in DIMENSIONS]))
    report_lines.append(f"- **综合表现最佳**: {overall_best.replace('qwen3-', 'Qwen3-').upper()}")
    
    # 找出最均衡的模型（标准差最小）
    model_variance = {}
    for model in models:
        variances = [stats[model][dim]['std'] for dim in DIMENSIONS]
        model_variance[model] = np.mean(variances)
    
    most_consistent = min(model_variance, key=model_variance.get)
    report_lines.append(f"- **维度稳定性最佳**: {most_consistent.replace('qwen3-', 'Qwen3-').upper()} "
                       f"(平均标准差 {model_variance[most_consistent]:.2f})")
    
    # 找出区分度最大的维度
    dim_ranges = {}
    for dim in DIMENSIONS:
        max_score = max(stats[model][dim]['mean'] for model in models)
        min_score = min(stats[model][dim]['mean'] for model in models)
        dim_ranges[dim] = max_score - min_score
    
    most_discriminative = max(dim_ranges, key=dim_ranges.get)
    report_lines.append(f"- **模型区分度最大维度**: {DIMENSION_LABELS.get(most_discriminative, most_discriminative)} "
                       f"(分差 {dim_ranges[most_discriminative]:.2f} 分)")
    
    report_lines.append("")
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'dimension_summary.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 保存总结报告到：{report_file}")


def main():
    results_dir = 'outputs/model_evaluation_50cases'
    output_dir = 'outputs/experiments/A2_dimension_analysis'
    
    print("="*60)
    print("A2 实验：Five-dimensional score 分解分析")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载所有模型结果...")
    all_data = load_all_results(results_dir)
    
    if not all_data:
        print("错误：未找到任何模型结果文件")
        return
    
    # 2. 提取维度评分
    print("\n2. 提取各维度评分...")
    dimension_stats = extract_dimension_scores(all_data)
    
    # 3. 计算统计指标
    print("\n3. 计算统计指标...")
    stats = calculate_statistics(dimension_stats)
    
    # 4. 生成汇总表
    print("\n4. 生成维度统计表...")
    data = generate_summary_table(stats, output_dir)
    
    # 5. 生成雷达图
    print("\n5. 生成五维雷达图...")
    create_radar_chart(stats, output_dir)
    
    # 6. 生成柱状图
    print("\n6. 生成维度对比柱状图...")
    create_bar_chart(stats, output_dir)
    
    # 7. 生成堆叠柱状图
    print("\n7. 生成堆叠柱状图...")
    create_stacked_bar_chart(stats, output_dir)
    
    # 8. 生成总结报告
    print("\n8. 生成总结报告...")
    generate_summary_report(stats, output_dir)
    
    print("\n" + "="*60)
    print("✓ A2 实验完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
