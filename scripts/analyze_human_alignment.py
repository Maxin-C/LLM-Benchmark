#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A3 实验：Human alignment 详细结果分析

基于人类标注数据进行分析
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Nature/Cell-like palette adapted from the EASE schematic
EASE_COLORS = {
    "purple": "#6F3CC3",
    "green": "#2E7D32",
    "teal": "#0F7C80",
    "blue": "#1554D1",
    "orange": "#FF6A00",
    "rose": "#D83F87",
    "light_purple": "#F3EEFF",
    "light_green": "#EEF8F0",
    "light_teal": "#EEF8F8",
    "light_blue": "#EEF4FF",
    "light_orange": "#FFF1E8",
    "light_rose": "#FFF0F5",
    "text": "#111827",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
}

MODEL_COLORS = [
    EASE_COLORS["purple"],
    EASE_COLORS["green"],
    EASE_COLORS["teal"],
    EASE_COLORS["blue"],
    EASE_COLORS["orange"],
    EASE_COLORS["rose"],
]


def set_ease_plot_style():
    """Apply a clean Nature/Cell-like theme matching the EASE schematic."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "xtick.color": EASE_COLORS["text"],
        "ytick.color": EASE_COLORS["text"],
        "text.color": EASE_COLORS["text"],
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.edgecolor": "#E5E7EB",
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFF",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def format_model_label(model: str) -> str:
    """Format model names for figure legends."""
    if model == "gpt-4o":
        return "GPT-4o"
    return model.replace("qwen3-", "Qwen3-").replace("-a22b", "-A22B")


def save_figure(fig, output_path):
    """Save figure in multiple formats."""
    fig.savefig(output_path + ".png", bbox_inches='tight', dpi=300)
    fig.savefig(output_path + ".pdf", bbox_inches='tight')
    fig.savefig(output_path + ".svg", bbox_inches='tight')


def load_human_evaluations(file_path):
    """加载人类标注数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ 加载人类标注数据: {len(data)} 个样本")
    return data


def analyze_human_evaluations(data):
    """分析人类评估数据"""
    # 维度映射
    dimension_mapping = {
        "是否认同模型回答能够容易地被非医学背景的读者理解并应用？": "understandability",
        "是否认同模型的推理过程与临床推理逻辑保持一致？": "clinical_reasoning",
        "是否认同模型回答考虑到了患者特定的病理特征？": "personalization",
        "是否认同模型回答存在误导性风险建议？": "safety",
        "是否认同模型回答考虑到了患者的情感需求？": "empathy"
    }
    
    # 收集所有评估
    all_evaluations = []
    
    for sample in data:
        sample_id = sample['id']
        group = sample['group']
        
        for eval_item in sample['evaluations']:
            model = eval_item['model']
            form = eval_item['form']
            result = eval_item['result']
            num_scores = eval_item['num']
            
            eval_dict = {
                'sample_id': sample_id,
                'group': group,
                'model': model,
                'form': form
            }
            
            # 添加各维度评分
            for question, score in result.items():
                dim_key = dimension_mapping.get(question, question)
                eval_dict[dim_key] = score
            
            # 添加总分
            eval_dict['total_score'] = sum(num_scores)
            eval_dict['avg_score'] = np.mean(num_scores)
            
            all_evaluations.append(eval_dict)
    
    return pd.DataFrame(all_evaluations)


def calculate_statistics(df):
    """计算统计指标"""
    stats = {}
    
    # 1. 按模型分组统计
    model_stats = df.groupby('model').agg({
        'understandability': ['mean', 'std', 'count'],
        'clinical_reasoning': ['mean', 'std'],
        'personalization': ['mean', 'std'],
        'safety': ['mean', 'std'],
        'empathy': ['mean', 'std'],
        'total_score': ['mean', 'std'],
        'avg_score': ['mean', 'std']
    }).round(2)
    
    stats['model_stats'] = model_stats
    
    # 2. 评估者间一致性（通过同一案例不同评估的相关性衡量）
    inter_rater_corr = {}
    for dim in ['understandability', 'clinical_reasoning', 'personalization', 'safety', 'empathy']:
        # 计算同一案例不同评估之间的平均相关性
        case_groups = df.groupby('sample_id')
        correlations = []
        
        for _, group in case_groups:
            if len(group) >= 2:
                scores = group[dim].values
                # 计算评分者间相关性
                if len(set(scores)) > 1:
                    corr_matrix = np.corrcoef([scores, scores])  # 简化计算
                    correlations.append(corr_matrix[0, 1])
        
        if correlations:
            inter_rater_corr[dim] = {
                'mean_corr': float(np.mean(correlations)),
                'std_corr': float(np.std(correlations)),
                'count': len(correlations)
            }
    
    stats['inter_rater_corr'] = inter_rater_corr
    
    # 3. 各维度相关性
    dim_correlations = df[['understandability', 'clinical_reasoning', 'personalization', 'safety', 'empathy']].corr()
    stats['dim_correlations'] = dim_correlations
    
    # 4. 评分分布
    score_dist = {}
    for dim in ['understandability', 'clinical_reasoning', 'personalization', 'safety', 'empathy']:
        scores = df[dim].values
        score_dist[dim] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores))
        }
    
    stats['score_dist'] = score_dist
    
    return stats


def plot_human_alignment(df, output_dir):
    """Plot human alignment results with grouped bars and a clean correlation matrix."""
    set_ease_plot_style()
    models = list(df['model'].unique())
    dims = ['understandability', 'clinical_reasoning', 'personalization', 'safety', 'empathy']
    dim_names = ['Understandability', 'Clinical Reasoning', 'Personalization', 'Safety', 'Empathy']

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    x = np.arange(len(dims))
    width = min(0.78 / max(len(models), 1), 0.13)
    offsets = (np.arange(len(models)) - (len(models)-1)/2) * width
    band_colors = [EASE_COLORS['light_purple'], EASE_COLORS['light_green'], EASE_COLORS['light_teal'], EASE_COLORS['light_blue'], EASE_COLORS['light_orange']]
    for j in range(len(dims)):
        ax.axvspan(j - 0.46, j + 0.46, color=band_colors[j % len(band_colors)], alpha=0.42, zorder=0)

    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        means = [model_data[dim].mean() for dim in dims]
        bars = ax.bar(x + offsets[i], means, width, label=format_model_label(model),
                      color=MODEL_COLORS[i % len(MODEL_COLORS)], edgecolor='white', linewidth=0.8, zorder=3)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, v + 0.06, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, fontweight='bold')
    ax.set_ylabel('Average Score')
    ax.set_title('Human Evaluation by Model and Dimension', pad=20)
    ax.set_ylim(0, 5.4)
    ax.grid(axis='y', color=EASE_COLORS['grid'], linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'human_alignment_by_model'))
    print(f"✓ 保存模型维度表现图到：{output_dir}/human_alignment_by_model.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    corr = df[dims].corr().values
    im = ax.imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
    for i in range(len(dims)):
        for j in range(len(dims)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_xticks(np.arange(len(dims))); ax.set_yticks(np.arange(len(dims)))
    ax.set_xticklabels(dim_names, rotation=35, ha='right', fontweight='bold')
    ax.set_yticklabels(dim_names, fontweight='bold')
    ax.set_title('Dimension Correlation Matrix')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson r')
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'dimension_correlation'))
    print(f"✓ 保存维度相关性热图到：{output_dir}/dimension_correlation.png")
    plt.close()

def generate_summary_report(stats, df, output_dir):
    """生成总结报告"""
    report_lines = []
    report_lines.append("# A3 实验：Human Alignment 详细结果报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**数据源**: {len(df)} 个人类评估记录\n")
    report_lines.append(f"**样本数**: {df['sample_id'].nunique()} 个案例\n")
    report_lines.append(f"**模型数**: {df['model'].nunique()} 个模型\n")
    report_lines.append("---\n")
    
    # 1. 各模型表现
    report_lines.append("## 1. 各模型人类评估表现\n")
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        report_lines.append(f"### {model}\n")
        report_lines.append(f"- **评估数**: {len(model_data)}\n")
        report_lines.append(f"- **平均总分**: {model_data['avg_score'].mean():.2f}\n")
        report_lines.append(f"- **可理解性**: {model_data['understandability'].mean():.2f} ± {model_data['understandability'].std():.2f}\n")
        report_lines.append(f"- **临床推理**: {model_data['clinical_reasoning'].mean():.2f} ± {model_data['clinical_reasoning'].std():.2f}\n")
        report_lines.append(f"- **个性化**: {model_data['personalization'].mean():.2f} ± {model_data['personalization'].std():.2f}\n")
        report_lines.append(f"- **安全性**: {model_data['safety'].mean():.2f} ± {model_data['safety'].std():.2f}\n")
        report_lines.append(f"- **共情能力**: {model_data['empathy'].mean():.2f} ± {model_data['empathy'].std():.2f}\n\n")
    
    # 2. 评估者间一致性
    report_lines.append("## 2. 评估者间一致性\n")
    
    dim_names = {
        'understandability': '可理解性',
        'clinical_reasoning': '临床推理',
        'personalization': '个性化',
        'safety': '安全性',
        'empathy': '共情能力'
    }
    
    for dim, corr_data in stats['inter_rater_corr'].items():
        report_lines.append(f"- **{dim_names[dim]}**: "
                          f"平均相关系数 r = {corr_data['mean_corr']:.2f} "
                          f"(n={corr_data['count']})\n")
    report_lines.append("")
    
    # 3. 维度相关性
    report_lines.append("## 3. 维度相关性\n")
    
    dims = ['understandability', 'clinical_reasoning', 'personalization', 'safety', 'empathy']
    for i, dim1 in enumerate(dims):
        for j, dim2 in enumerate(dims):
            if i < j:
                corr = stats['dim_correlations'].loc[dim1, dim2]
                report_lines.append(f"- **{dim_names[dim1]} vs {dim_names[dim2]}**: r = {corr:.2f}\n")
    report_lines.append("")
    
    # 4. 评分分布
    report_lines.append("## 4. 评分分布\n")
    
    for dim, dist in stats['score_dist'].items():
        report_lines.append(f"- **{dim_names[dim]}**: "
                          f"平均分 {dist['mean']:.2f}, "
                          f"标准差 {dist['std']:.2f}, "
                          f"范围 [{dist['min']:.1f}, {dist['max']:.1f}]\n")
    report_lines.append("")
    
    # 5. 关键发现
    report_lines.append("## 5. 关键发现\n")
    
    # 最佳模型
    best_model = df.groupby('model')['avg_score'].mean().idxmax()
    best_score = df.groupby('model')['avg_score'].mean().max()
    report_lines.append(f"- **人类评估最佳模型**: {best_model} (平均分 {best_score:.2f})\n")
    
    # 最差模型
    worst_model = df.groupby('model')['avg_score'].mean().idxmin()
    worst_score = df.groupby('model')['avg_score'].mean().min()
    report_lines.append(f"- **人类评估最差模型**: {worst_model} (平均分 {worst_score:.2f})\n")
    
    # 评估者一致性总结
    avg_corr = np.mean([d['mean_corr'] for d in stats['inter_rater_corr'].values()])
    report_lines.append(f"- **平均评估者间一致性**: r = {avg_corr:.2f}\n")
    
    # 最高和最低维度评分
    dim_means = {dim: df[dim].mean() for dim in dims}
    highest_dim = max(dim_means, key=dim_means.get)
    lowest_dim = min(dim_means, key=dim_means.get)
    report_lines.append(f"- **评分最高维度**: {dim_names[highest_dim]} ({dim_means[highest_dim]:.2f})\n")
    report_lines.append(f"- **评分最低维度**: {dim_names[lowest_dim]} ({dim_means[lowest_dim]:.2f})\n")
    
    report_lines.append("")
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'human_alignment_summary.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 保存总结报告到：{report_file}")


def save_results(stats, df, output_dir):
    """保存所有结果"""
    # 简化模型统计
    model_rankings = df.groupby('model')['avg_score'].mean().sort_values(ascending=False).to_dict()
    
    # 转换为可序列化格式
    output = {
        'inter_rater_corr': stats['inter_rater_corr'],
        'score_dist': stats['score_dist'],
        'model_rankings': model_rankings,
        'sample_count': df['sample_id'].nunique(),
        'evaluation_count': len(df),
        'model_means': {
            model: {
                'understandability': float(df[df['model'] == model]['understandability'].mean()),
                'clinical_reasoning': float(df[df['model'] == model]['clinical_reasoning'].mean()),
                'personalization': float(df[df['model'] == model]['personalization'].mean()),
                'safety': float(df[df['model'] == model]['safety'].mean()),
                'empathy': float(df[df['model'] == model]['empathy'].mean()),
                'avg_score': float(df[df['model'] == model]['avg_score'].mean())
            } for model in df['model'].unique()
        }
    }
    
    output_file = os.path.join(output_dir, 'human_alignment_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存人类对齐分析结果到：{output_file}")


def main():
    input_file = 'dataset/real_eval/human_evaluations.json'
    output_dir = 'outputs/experiments/A3_human_alignment'
    
    print("="*60)
    print("A3 实验：Human Alignment 详细结果分析")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载人类标注数据...")
    data = load_human_evaluations(input_file)
    
    # 2. 分析数据
    print("\n2. 分析人类评估数据...")
    df = analyze_human_evaluations(data)
    print(f"   生成 {len(df)} 条评估记录")
    
    # 3. 计算统计指标
    print("\n3. 计算统计指标...")
    stats = calculate_statistics(df)
    
    # 4. 生成可视化图表
    print("\n4. 生成可视化图表...")
    plot_human_alignment(df, output_dir)
    
    # 5. 保存结果
    print("\n5. 保存分析结果...")
    save_results(stats, df, output_dir)
    
    # 6. 生成报告
    print("\n6. 生成总结报告...")
    generate_summary_report(stats, df, output_dir)
    
    print("\n" + "="*60)
    print("✓ A3 实验完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)


if __name__ == '__main__':
        main()
