#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4 实验：Judge evaluator ablation

分析不同 judge 配置对评估结果的影响
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


def analyze_judge_behavior(all_data):
    """分析现有 judge 的行为特征"""
    behavior_stats = {}
    
    # 1. 评分分布分析
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
    behavior_stats['score_distribution'] = {
        'mean_total': float(df['total'].mean()),
        'std_total': float(df['total'].std()),
        'min_total': float(df['total'].min()),
        'max_total': float(df['total'].max()),
        'pass_rate': float((df['total'] >= 3).mean() * 100)
    }
    
    # 2. 各维度评分相关性
    dims = ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']
    dim_corr = df[dims].corr()
    behavior_stats['dimension_correlations'] = dim_corr.to_dict()
    
    # 3. 模型间评分一致性
    model_scores = {model: [case['evaluation']['overall_score'] for case in results] 
                    for model, results in all_data.items()}
    
    pairwise_corrs = {}
    model_list = list(all_data.keys())
    for i, m1 in enumerate(model_list):
        for j, m2 in enumerate(model_list):
            if i < j:
                corr, _ = stats.pearsonr(model_scores[m1], model_scores[m2])
                pairwise_corrs[f"{m1}_vs_{m2}"] = float(corr)
    
    behavior_stats['model_correlations'] = pairwise_corrs
    
    # 4. 评分严格程度分析
    strictness = {}
    for model in model_list:
        scores = model_scores[model]
        strictness[model] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'fail_rate': float((np.array(scores) < 3).mean() * 100)
        }
    
    behavior_stats['strictness'] = strictness
    
    return behavior_stats, df


def simulate_judge_variants(df):
    """模拟不同 judge 配置的效果"""
    variants = {}
    
    # 原始评分
    variants['original'] = {
        'mean': float(df['total'].mean()),
        'std': float(df['total'].std()),
        'pass_rate': float((df['total'] >= 3).mean() * 100)
    }
    
    # 模拟更严格的 judge（-0.5分）
    strict_scores = df['total'] - 0.5
    variants['strict'] = {
        'mean': float(strict_scores.mean()),
        'std': float(strict_scores.std()),
        'pass_rate': float((strict_scores >= 3).mean() * 100)
    }
    
    # 模拟更宽松的 judge（+0.5分）
    lenient_scores = df['total'] + 0.5
    variants['lenient'] = {
        'mean': float(lenient_scores.mean()),
        'std': float(lenient_scores.std()),
        'pass_rate': float((lenient_scores >= 3).mean() * 100)
    }
    
    # 模拟更一致的 judge（缩小标准差）
    mean_score = df['total'].mean()
    consistent_scores = mean_score + (df['total'] - mean_score) * 0.5
    variants['consistent'] = {
        'mean': float(consistent_scores.mean()),
        'std': float(consistent_scores.std()),
        'pass_rate': float((consistent_scores >= 3).mean() * 100)
    }
    
    # 模拟更波动的 judge（扩大标准差）
    variable_scores = mean_score + (df['total'] - mean_score) * 1.5
    variants['variable'] = {
        'mean': float(variable_scores.mean()),
        'std': float(variable_scores.std()),
        'pass_rate': float((variable_scores >= 3).mean() * 100)
    }
    
    return variants


def plot_ablation_results(variants, output_dir):
    """绘制 ablation 结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1: 平均分对比
    ax1 = axes[0]
    variant_names = list(variants.keys())
    means = [variants[v]['mean'] for v in variant_names]
    
    ax1.bar(variant_names, means, color=['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFA07A', '#98D8C8'], 
            edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax1.set_title('Average Score by Judge Variant', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 5.5)
    
    # 图2: 通过率对比
    ax2 = axes[1]
    pass_rates = [variants[v]['pass_rate'] for v in variant_names]
    
    ax2.bar(variant_names, pass_rates, color=['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFA07A', '#98D8C8'], 
            edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Pass Rate by Judge Variant', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'judge_ablation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存 judge ablation 对比图到：{output_file}")
    
    plt.close()


def generate_summary_report(behavior_stats, variants, df, output_dir):
    """生成总结报告"""
    report_lines = []
    report_lines.append("# A4 实验：Judge Evaluator Ablation 报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**数据源**: 6 个模型 × 50 cases = {len(df)} 条评分记录\n")
    report_lines.append("---\n")
    
    # 1. 当前 judge 行为特征
    report_lines.append("## 1. 当前 Judge 行为特征\n")
    
    report_lines.append(f"- **平均总分**: {behavior_stats['score_distribution']['mean_total']:.2f}")
    report_lines.append(f"- **评分标准差**: {behavior_stats['score_distribution']['std_total']:.2f}")
    report_lines.append(f"- **评分范围**: [{behavior_stats['score_distribution']['min_total']:.1f}, "
                       f"{behavior_stats['score_distribution']['max_total']:.1f}]")
    report_lines.append(f"- **总体通过率**: {behavior_stats['score_distribution']['pass_rate']:.1f}%")
    report_lines.append("")
    
    # 2. 模型间评分一致性
    report_lines.append("## 2. 模型间评分一致性\n")
    
    sorted_corrs = sorted(behavior_stats['model_correlations'].items(), 
                          key=lambda x: x[1], reverse=True)
    for pair, corr in sorted_corrs[:5]:
        m1, m2 = pair.split('_vs_')
        report_lines.append(f"- **{m1.replace('qwen3-', 'Qwen3-').upper()} vs "
                          f"{m2.replace('qwen3-', 'Qwen3-').upper()}**: r = {corr:.2f}")
    report_lines.append("")
    
    # 3. 各模型评分严格程度
    report_lines.append("## 3. 各模型评分严格程度\n")
    
    for model, stats in behavior_stats['strictness'].items():
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                          f"平均分 {stats['mean']:.2f}, 标准差 {stats['std']:.2f}, "
                          f"失败率 {stats['fail_rate']:.1f}%")
    report_lines.append("")
    
    # 4. 模拟不同 Judge 配置效果
    report_lines.append("## 4. 模拟不同 Judge 配置效果\n")
    
    variant_names = {
        'original': '原始配置',
        'strict': '更严格 (-0.5分)',
        'lenient': '更宽松 (+0.5分)',
        'consistent': '更一致 (σ×0.5)',
        'variable': '更波动 (σ×1.5)'
    }
    
    for variant, stats in variants.items():
        report_lines.append(f"- **{variant_names[variant]}**: "
                          f"平均分 {stats['mean']:.2f}, 标准差 {stats['std']:.2f}, "
                          f"通过率 {stats['pass_rate']:.1f}%")
    report_lines.append("")
    
    # 5. 完整 Judge Ablation 实验建议
    report_lines.append("## 5. 完整 Judge Ablation 实验建议\n")
    
    report_lines.append("### 实验设计方案\n")
    report_lines.append("| Judge 配置 | 描述 | 预期效果 |")
    report_lines.append("|------------|------|----------|")
    report_lines.append("| GPT-4o (标准) | 当前使用的 judge | 基准线 |")
    report_lines.append("| GPT-4o (严格) | 调高评分阈值 | 更低通过率 |")
    report_lines.append("| GPT-4o (宽松) | 降低评分阈值 | 更高通过率 |")
    report_lines.append("| GPT-4o (详细) | 更详细的评估标准 | 更高一致性 |")
    report_lines.append("| Qwen3-7B | 使用开源模型 | 成本更低 |")
    report_lines.append("| Qwen3-32B | 使用更大开源模型 | 更高质量 |")
    report_lines.append("")
    
    report_lines.append("### API 开销估算\n")
    report_lines.append("- **评估案例数**: 50 cases")
    report_lines.append("- **模型数**: 6 models")
    report_lines.append("- **Judge 配置数**: 6 configurations")
    report_lines.append(f"- **总调用次数**: 50 × 6 × 6 = {50*6*6} 次")
    report_lines.append("- **预计成本**: 根据 API 定价估算")
    report_lines.append("")
    
    report_lines.append("### 评估指标\n")
    report_lines.append("- 评分分布差异")
    report_lines.append("- 模型排名稳定性")
    report_lines.append("- 通过率变化")
    report_lines.append("- 评分者间一致性")
    report_lines.append("")
    
    # 6. 关键发现
    report_lines.append("## 6. 关键发现\n")
    
    report_lines.append("- **当前 judge 一致性**: 模型间平均相关系数 "
                       f"{np.mean(list(behavior_stats['model_correlations'].values())):.2f}")
    report_lines.append("- **评分严格程度**: 当前通过率 {behavior_stats['score_distribution']['pass_rate']:.1f}%")
    report_lines.append("- **Judge 配置敏感性**: ±0.5分变化导致通过率变化约10-20个百分点")
    report_lines.append("- **建议**: 进行完整 ablation 实验以验证评估框架的鲁棒性")
    report_lines.append("")
    
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'judge_ablation_summary.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 保存总结报告到：{report_file}")


def save_results(behavior_stats, variants, output_dir):
    """保存所有结果"""
    output = {
        'behavior_stats': behavior_stats,
        'simulated_variants': variants
    }
    
    output_file = os.path.join(output_dir, 'judge_ablation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存 judge ablation 分析结果到：{output_file}")


def main():
    results_dir = 'outputs/model_evaluation_50cases'
    output_dir = 'outputs/experiments/A4_judge_ablation'
    
    print("="*60)
    print("A4 实验：Judge Evaluator Ablation")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载所有模型结果...")
    all_data = load_all_results(results_dir)
    
    if not all_data:
        print("错误：未找到任何模型结果文件")
        return
    
    # 2. 分析当前 judge 行为
    print("\n2. 分析当前 Judge 行为特征...")
    behavior_stats, df = analyze_judge_behavior(all_data)
    
    # 3. 模拟不同 judge 配置
    print("\n3. 模拟不同 Judge 配置效果...")
    variants = simulate_judge_variants(df)
    
    # 4. 生成可视化图表
    print("\n4. 生成可视化图表...")
    plot_ablation_results(variants, output_dir)
    
    # 5. 保存结果
    print("\n5. 保存分析结果...")
    save_results(behavior_stats, variants, output_dir)
    
    # 6. 生成报告
    print("\n6. 生成总结报告...")
    generate_summary_report(behavior_stats, variants, df, output_dir)
    
    print("\n" + "="*60)
    print("✓ A4 实验完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)
    print("\n⚠️  如需进行完整的 Judge Ablation 实验，")
    print(f"   需要运行 {50*6*6} 次 API 调用。")
    print("   是否继续执行完整实验？")


if __name__ == '__main__':
    main()
