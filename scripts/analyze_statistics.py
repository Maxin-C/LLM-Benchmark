#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A6 实验：Benchmark 区分度的统计检验

基于已有评估数据进行 bootstrap、显著性检验和 ranking 稳定性分析
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import seaborn as sns


def bonferroni_correction(p_values, alpha=0.05):
    """手动实现 Bonferroni 校正"""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values], [min(p * n, 1.0) for p in p_values]


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


def extract_scores(all_data):
    """提取各模型的 overall_score"""
    scores_dict = {}
    
    for model, results in all_data.items():
        scores = [case['evaluation']['overall_score'] for case in results 
                  if 'evaluation' in case and 'overall_score' in case['evaluation']]
        scores_dict[model] = np.array(scores)
    
    return scores_dict


def bootstrap_confidence_intervals(scores_dict, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap 重采样计算置信区间
    
    Args:
        scores_dict: 各模型分数
        n_bootstrap: bootstrap 次数
        ci: 置信水平
    
    Returns:
        bootstrap_results: 包含统计量和置信区间的字典
    """
    print(f"进行 Bootstrap 重采样 ({n_bootstrap} 次)...")
    
    bootstrap_results = {}
    
    for model, scores in scores_dict.items():
        n = len(scores)
        bootstrap_means = []
        
        # Bootstrap 重采样
        for _ in range(n_bootstrap):
            # 有放回抽样
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # 计算置信区间
        alpha = 1 - ci
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        bootstrap_results[model] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'bootstrap_mean': float(np.mean(bootstrap_means)),
            'bootstrap_std': float(np.std(bootstrap_means)),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'ci_width': float(upper - lower)
        }
    
    return bootstrap_results


def anova_test(scores_dict):
    """
    单因素方差分析 (ANOVA)
    检验所有模型之间是否存在显著差异
    """
    print("\n进行 ANOVA 检验...")
    
    # 准备数据
    data = [scores for scores in scores_dict.values()]
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(*data)
    
    return {
        'F_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def posthoc_ttests(scores_dict, correction='bonferroni'):
    """
    事后两两比较 (Post-hoc t-tests)
    
    Args:
        scores_dict: 各模型分数
        correction: 多重检验校正方法 ('bonferroni', 'holm', 'fdr')
    
    Returns:
        comparisons: 所有两两比较结果
    """
    print(f"\n进行事后两两比较 (校正方法：{correction})...")
    
    models = list(scores_dict.keys())
    comparisons = []
    p_values = []
    
    # 所有两两组合
    for model1, model2 in combinations(models, 2):
        # Welch's t-test (不假设方差相等)
        t_stat, p_value = stats.ttest_ind(
            scores_dict[model1], 
            scores_dict[model2], 
            equal_var=False
        )
        
        comparisons.append({
            'model1': model1,
            'model2': model2,
            't_statistic': float(t_stat),
            'p_value_uncorrected': float(p_value),
            'mean_diff': float(np.mean(scores_dict[model1]) - np.mean(scores_dict[model2]))
        })
        p_values.append(p_value)
    
    # 多重检验校正
    if correction == 'bonferroni':
        _, corrected_p = bonferroni_correction(p_values, alpha=0.05)
    elif correction == 'holm':
        # Holm 校正（简化版）
        sorted_indices = np.argsort(p_values)
        n = len(p_values)
        corrected_p = np.zeros(n)
        for i, idx in enumerate(sorted_indices):
            corrected_p[idx] = min(min((n - j) * p_values[idx] for j in range(i + 1)), 1.0)
    elif correction == 'fdr':
        # FDR 校正（简化版）
        sorted_indices = np.argsort(p_values)
        n = len(p_values)
        corrected_p = np.zeros(n)
        for i, idx in enumerate(sorted_indices):
            corrected_p[idx] = min(n * p_values[idx] / (i + 1), 1.0)
    else:
        corrected_p = p_values
    
    # 添加校正后的 p 值
    for i, comp in enumerate(comparisons):
        comp['p_value_corrected'] = float(corrected_p[i])
        comp['significant'] = bool(corrected_p[i] < 0.05)
    
    return comparisons


def calculate_effect_sizes(scores_dict):
    """
    计算效应量 (Cohen's d)
    """
    print("\n计算效应量 (Cohen's d)...")
    
    models = list(scores_dict.keys())
    effect_sizes = []
    
    for model1, model2 in combinations(models, 2):
        # Cohen's d
        mean1 = np.mean(scores_dict[model1])
        mean2 = np.mean(scores_dict[model2])
        std1 = np.std(scores_dict[model1], ddof=1)
        std2 = np.std(scores_dict[model2], ddof=1)
        
        # 合并标准差
        n1 = len(scores_dict[model1])
        n2 = len(scores_dict[model2])
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std > 0:
            cohens_d = (mean1 - mean2) / pooled_std
        else:
            cohens_d = 0
        
        # 效应量解释
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = 'negligible'
        elif abs_d < 0.5:
            magnitude = 'small'
        elif abs_d < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'
        
        effect_sizes.append({
            'model1': model1,
            'model2': model2,
            'cohens_d': float(cohens_d),
            'magnitude': magnitude
        })
    
    return effect_sizes


def analyze_ranking_stability(scores_dict, n_bootstrap=1000):
    """
    分析 ranking 稳定性
    
    Args:
        scores_dict: 各模型分数
        n_bootstrap: bootstrap 次数
    
    Returns:
        ranking_stats: ranking 统计信息
    """
    print(f"\n分析 Ranking 稳定性 ({n_bootstrap} 次重采样)...")
    
    models = list(scores_dict.keys())
    n_models = len(models)
    
    # 记录每个模型在每个 bootstrap 样本中的排名
    bootstrap_ranks = {model: [] for model in models}
    
    for _ in range(n_bootstrap):
        # 重采样
        sample_means = {}
        for model, scores in scores_dict.items():
            sample = np.random.choice(scores, size=len(scores), replace=True)
            sample_means[model] = np.mean(sample)
        
        # 排名（1=最好）
        sorted_models = sorted(sample_means.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, _) in enumerate(sorted_models, 1):
            bootstrap_ranks[model].append(rank)
    
    # 统计
    ranking_stats = {}
    for model in models:
        ranks = bootstrap_ranks[model]
        ranking_stats[model] = {
            'mean_rank': float(np.mean(ranks)),
            'rank_std': float(np.std(ranks)),
            'rank_5th': float(np.percentile(ranks, 5)),
            'rank_95th': float(np.percentile(ranks, 95)),
            'rank_mode': float(stats.mode(ranks, keepdims=True)[0][0])
        }
    
    return ranking_stats


def plot_confidence_intervals(bootstrap_results, output_dir):
    """绘制置信区间图"""
    print("\n绘制置信区间图...")
    
    models = list(bootstrap_results.keys())
    means = [bootstrap_results[m]['mean'] for m in models]
    ci_lower = [bootstrap_results[m]['ci_lower'] for m in models]
    ci_upper = [bootstrap_results[m]['ci_upper'] for m in models]
    
    # 按均值排序
    sorted_idx = np.argsort(means)[::-1]
    models = [models[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    ci_lower = [ci_lower[i] for i in sorted_idx]
    ci_upper = [ci_upper[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(models))
    
    # 误差线
    ax.errorbar(means, y_pos, xerr=[np.array(means) - np.array(ci_lower), 
                                     np.array(ci_upper) - np.array(means)],
                fmt='o', capsize=5, capthick=2, ecolor='red', 
                color='#2E86AB', markersize=12, linewidth=0)
    
    # 设置
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('qwen3-', 'Qwen3-').upper() for m in models], fontsize=11)
    ax.set_xlabel('Overall Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance with 95% Confidence Intervals\n(Bootstrap 1000 次)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 5.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'confidence_intervals.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存置信区间图到：{output_file}")
    
    plt.close()


def plot_significance_matrix(comparisons, models, output_dir):
    """绘制显著性矩阵热力图"""
    print("\n绘制显著性矩阵...")
    
    n = len(models)
    significance_matrix = np.zeros((n, n))
    
    for comp in comparisons:
        i = models.index(comp['model1'])
        j = models.index(comp['model2'])
        sig = 1 if comp['significant'] else 0
        significance_matrix[i, j] = sig
        significance_matrix[j, i] = sig
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    im = ax.imshow(significance_matrix, cmap='RdYlGn', vmin=0, vmax=1, alpha=0.7)
    
    # 添加文本
    for i in range(n):
        for j in range(n):
            if i != j:
                sig_text = '✓' if significance_matrix[i, j] > 0 else '✗'
                ax.text(j, i, sig_text, ha='center', va='center', 
                       fontsize=16, fontweight='bold',
                       color='green' if significance_matrix[i, j] > 0 else 'gray')
            else:
                ax.text(j, i, '-', ha='center', va='center', 
                       fontsize=16, color='lightgray')
    
    # 设置标签
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([m.replace('qwen3-', 'Qwen3-').upper() for m in models], 
                       fontsize=10, fontweight='bold')
    ax.set_yticklabels([m.replace('qwen3-', 'Qwen3-').upper() for m in models], 
                       fontsize=10, fontweight='bold')
    
    # 旋转标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_title('Pairwise Significance Test (p < 0.05 after correction)\n(✓ = 显著差异，✗ = 无显著差异)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'significance_matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存显著性矩阵到：{output_file}")
    
    plt.close()


def plot_ranking_stability(ranking_stats, output_dir):
    """绘制 ranking 稳定性图"""
    print("\n绘制 Ranking 稳定性图...")
    
    models = list(ranking_stats.keys())
    
    # 按平均排名排序
    sorted_models = sorted(models, key=lambda m: ranking_stats[m]['mean_rank'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(sorted_models))
    means = [ranking_stats[m]['mean_rank'] for m in sorted_models]
    stds = [ranking_stats[m]['rank_std'] for m in sorted_models]
    
    # 绘制条形图
    bars = ax.barh(y_pos, means, xerr=stds, capsize=5, 
                   color='#6A994E', edgecolor='black', linewidth=0.5, height=0.6)
    
    # 设置
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('qwen3-', 'Qwen3-').upper() for m in sorted_models], fontsize=11)
    ax.set_xlabel('Average Rank (1=Best)', fontsize=12, fontweight='bold')
    ax.set_title('Model Ranking Stability Analysis\n(Bootstrap 1000 次)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.invert_yaxis()  # 排名 1 在顶部
    ax.set_xlim(0, len(models) + 1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (bar, model) in enumerate(zip(bars, sorted_models)):
        width = means[i]
        ax.text(width + 0.1, i, f'{width:.2f}±{stds[i]:.2f}', 
               va='center', fontsize=10)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'ranking_stability.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存 Ranking 稳定性图到：{output_file}")
    
    plt.close()


def generate_summary_report(bootstrap_results, anova_result, comparisons, 
                           effect_sizes, ranking_stats, output_dir):
    """生成总结报告"""
    models = list(bootstrap_results.keys())
    
    report_lines = []
    report_lines.append("# A6 实验：Benchmark 区分度的统计检验报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**数据源**: 6 个模型 × 50 cases\n")
    report_lines.append(f"**Bootstrap 次数**: 1000\n")
    report_lines.append("---\n")
    
    # 1. Bootstrap 置信区间
    report_lines.append("## 1. Bootstrap 置信区间 (95%)\n")
    sorted_models = sorted(models, key=lambda m: bootstrap_results[m]['mean'], reverse=True)
    for model in sorted_models:
        stats_data = bootstrap_results[model]
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                          f"{stats_data['mean']:.2f} "
                          f"95% CI [{stats_data['ci_lower']:.2f}, {stats_data['ci_upper']:.2f}]")
    report_lines.append("")
    
    # 2. ANOVA 检验
    report_lines.append("## 2. 方差分析 (ANOVA)\n")
    report_lines.append(f"- **F 统计量**: {anova_result['F_statistic']:.2f}")
    report_lines.append(f"- **p 值**: {anova_result['p_value']:.2e}")
    report_lines.append(f"- **结论**: {'模型间存在显著差异 ✓' if anova_result['significant'] else '模型间无显著差异 ✗'}")
    report_lines.append("")
    
    # 3. 事后两两比较
    report_lines.append("## 3. 事后两两比较 (Bonferroni 校正)\n")
    significant_comps = [c for c in comparisons if c['significant']]
    non_significant_comps = [c for c in comparisons if not c['significant']]
    
    report_lines.append(f"### 显著差异的模型对 ({len(significant_comps)} 对)\n")
    for comp in significant_comps:
        report_lines.append(f"- {comp['model1'].replace('qwen3-', 'Qwen3-').upper()} vs "
                          f"{comp['model2'].replace('qwen3-', 'Qwen3-').upper()}: "
                          f"p={comp['p_value_corrected']:.4f}, "
                          f"均值差={comp['mean_diff']:.2f}")
    
    report_lines.append(f"\n### 无显著差异的模型对 ({len(non_significant_comps)} 对)\n")
    for comp in non_significant_comps:
        report_lines.append(f"- {comp['model1'].replace('qwen3-', 'Qwen3-').upper()} vs "
                          f"{comp['model2'].replace('qwen3-', 'Qwen3-').upper()}: "
                          f"p={comp['p_value_corrected']:.4f}")
    report_lines.append("")
    
    # 4. 效应量
    report_lines.append("## 4. 效应量分析 (Cohen's d)\n")
    large_effects = [e for e in effect_sizes if e['magnitude'] == 'large']
    medium_effects = [e for e in effect_sizes if e['magnitude'] == 'medium']
    
    report_lines.append(f"### 大效应量 (|d| > 0.8): {len(large_effects)} 对\n")
    for eff in large_effects:
        report_lines.append(f"- {eff['model1'].replace('qwen3-', 'Qwen3-').upper()} vs "
                          f"{eff['model2'].replace('qwen3-', 'Qwen3-').upper()}: "
                          f"d={eff['cohens_d']:.2f}")
    
    report_lines.append(f"\n### 中等效应量 (0.5 < |d| < 0.8): {len(medium_effects)} 对\n")
    for eff in medium_effects:
        report_lines.append(f"- {eff['model1'].replace('qwen3-', 'Qwen3-').upper()} vs "
                          f"{eff['model2'].replace('qwen3-', 'Qwen3-').upper()}: "
                          f"d={eff['cohens_d']:.2f}")
    report_lines.append("")
    
    # 5. Ranking 稳定性
    report_lines.append("## 5. Ranking 稳定性分析\n")
    sorted_by_rank = sorted(models, key=lambda m: ranking_stats[m]['mean_rank'])
    for model in sorted_by_rank:
        rs = ranking_stats[model]
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                          f"平均排名 {rs['mean_rank']:.2f}±{rs['rank_std']:.2f} "
                          f"(95% CI: [{rs['rank_5th']:.1f}, {rs['rank_95th']:.1f}])")
    report_lines.append("")
    
    # 6. 关键发现
    report_lines.append("## 6. 关键发现\n")
    
    # 最佳模型
    best_model = sorted_models[0]
    report_lines.append(f"- **综合表现最佳**: {best_model.replace('qwen3-', 'Qwen3-').upper()} "
                       f"({bootstrap_results[best_model]['mean']:.2f})")
    
    # 最稳定模型
    most_stable = min(models, key=lambda m: ranking_stats[m]['rank_std'])
    report_lines.append(f"- **Ranking 最稳定**: {most_stable.replace('qwen3-', 'Qwen3-').upper()} "
                       f"(标准差 {ranking_stats[most_stable]['rank_std']:.2f})")
    
    # 区分度
    report_lines.append(f"- **显著差异的模型对**: {len(significant_comps)}/{len(comparisons)} "
                       f"({len(significant_comps)/len(comparisons)*100:.1f}%)")
    
    # 效应量
    report_lines.append(f"- **大效应量**: {len(large_effects)}/{len(effect_sizes)} "
                       f"({len(large_effects)/len(effect_sizes)*100:.1f}%)")
    
    report_lines.append("")
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'statistical_summary.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 保存总结报告到：{report_file}")


def save_results(bootstrap_results, anova_result, comparisons, 
                effect_sizes, ranking_stats, output_dir):
    """保存所有结果到 JSON"""
    # 转换 numpy 类型为 Python 原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results = {
        'bootstrap_results': convert_numpy(bootstrap_results),
        'anova_test': convert_numpy(anova_result),
        'posthoc_comparisons': convert_numpy(comparisons),
        'effect_sizes': convert_numpy(effect_sizes),
        'ranking_stability': convert_numpy(ranking_stats)
    }
    
    output_file = os.path.join(output_dir, 'bootstrap_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存统计结果到：{output_file}")


def main():
    results_dir = 'outputs/model_evaluation_50cases'
    output_dir = 'outputs/experiments/A6_statistical_test'
    
    print("="*60)
    print("A6 实验：Benchmark 区分度的统计检验")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载所有模型结果...")
    all_data = load_all_results(results_dir)
    
    if not all_data:
        print("错误：未找到任何模型结果文件")
        return
    
    # 2. 提取分数
    print("\n2. 提取 overall_score...")
    scores_dict = extract_scores(all_data)
    
    # 3. Bootstrap 置信区间
    print("\n3. Bootstrap 置信区间分析...")
    bootstrap_results = bootstrap_confidence_intervals(scores_dict, n_bootstrap=1000)
    
    # 4. ANOVA 检验
    print("\n4. 方差分析 (ANOVA)...")
    anova_result = anova_test(scores_dict)
    print(f"   F={anova_result['F_statistic']:.2f}, p={anova_result['p_value']:.2e}")
    
    # 5. 事后两两比较
    print("\n5. 事后两两比较...")
    comparisons = posthoc_ttests(scores_dict, correction='bonferroni')
    
    # 6. 效应量
    print("\n6. 效应量计算...")
    effect_sizes = calculate_effect_sizes(scores_dict)
    
    # 7. Ranking 稳定性
    print("\n7. Ranking 稳定性分析...")
    ranking_stats = analyze_ranking_stability(scores_dict, n_bootstrap=1000)
    
    # 8. 可视化
    print("\n8. 生成可视化图表...")
    plot_confidence_intervals(bootstrap_results, output_dir)
    plot_significance_matrix(comparisons, list(scores_dict.keys()), output_dir)
    plot_ranking_stability(ranking_stats, output_dir)
    
    # 9. 保存结果
    print("\n9. 保存统计结果...")
    save_results(bootstrap_results, anova_result, comparisons, 
                effect_sizes, ranking_stats, output_dir)
    
    # 10. 生成报告
    print("\n10. 生成总结报告...")
    generate_summary_report(bootstrap_results, anova_result, comparisons, 
                           effect_sizes, ranking_stats, output_dir)
    
    print("\n" + "="*60)
    print("✓ A6 实验完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
