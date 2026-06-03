#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A5 实验：Dynamic multi-turn vs static single-turn 对比分析

分析多轮对话与单轮对话的效果差异
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


def load_all_results(results_dir):
    """加载所有模型的结果"""
    models = ['gpt-4o', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b']
    all_data = {}
    
    for model in models:
        file_path = os.path.join(results_dir, f'{model}_results.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_data[model] = json.load(f)
            print(f"✓ 加载 {model}: {len(all_data[model])} cases")
        else:
            print(f"✗ 未找到 {model} 的结果文件")
    
    return all_data


def analyze_dialogue_turns(all_data):
    """分析对话轮数特征"""
    turn_stats = {}
    
    for model, results in all_data.items():
        model_turns = []
        
        for case in results:
            dialogue = case.get('dialogue_history', [])
            
            # 统计患者和医生的轮数
            patient_turns = sum(1 for turn in dialogue if turn.get('role', '').lower() == 'patient')
            doctor_turns = sum(1 for turn in dialogue if turn.get('role', '').lower() in ['assistant', 'doctor'])
            total_turns = len(dialogue)
            
            # 获取评分
            evaluation = case.get('evaluation', {})
            overall_score = evaluation.get('overall_score', 0)
            is_passed = evaluation.get('is_passed', False)
            
            model_turns.append({
                'patient_turns': patient_turns,
                'doctor_turns': doctor_turns,
                'total_turns': total_turns,
                'overall_score': overall_score,
                'is_passed': is_passed
            })
        
        turn_stats[model] = pd.DataFrame(model_turns)
    
    return turn_stats


def compare_turn_performance(turn_stats):
    """比较不同轮数的表现"""
    comparisons = {}
    
    for model, df in turn_stats.items():
        # 按总轮数分组统计
        turn_groups = df.groupby('total_turns').agg({
            'overall_score': ['mean', 'std', 'count'],
            'is_passed': ['mean']
        }).round(2)
        
        comparisons[model] = {
            'by_turns': turn_groups.to_dict(),
            'avg_turns': float(df['total_turns'].mean()),
            'avg_score': float(df['overall_score'].mean()),
            'turn_score_corr': float(stats.pearsonr(df['total_turns'], df['overall_score'])[0])
        }
    
    return comparisons


def plot_turn_analysis(turn_stats, output_dir):
    """Plot dialogue turn distributions and score-turn relationship."""
    set_ease_plot_style()
    models = list(turn_stats.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), sharey=False)
    axes = axes.flatten()
    for i, model in enumerate(models):
        ax = axes[i]
        df = turn_stats[model]
        counts = df['total_turns'].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=MODEL_COLORS[i % len(MODEL_COLORS)],
               edgecolor='white', linewidth=0.8, width=0.75)
        ax.set_title(f"{format_model_label(model)}  |  mean={df['total_turns'].mean():.1f}", fontsize=11)
        ax.set_xlabel('Total Turns'); ax.set_ylabel('Count')
        ax.grid(axis='y', color=EASE_COLORS['grid'], linestyle='--', linewidth=0.7)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for j in range(len(models), len(axes)):
        axes[j].axis('off')
    fig.suptitle('Dialogue Turn Distribution by Model', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'turn_distribution'))
    print(f"✓ 保存轮数分布图到：{output_dir}/turn_distribution.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for i, model in enumerate(models):
        df = turn_stats[model]
        grouped = df.groupby('total_turns')['overall_score'].agg(['mean', 'count']).reset_index()
        ax.plot(grouped['total_turns'], grouped['mean'], marker='o', markersize=6,
                label=format_model_label(model), color=MODEL_COLORS[i % len(MODEL_COLORS)], linewidth=2.2)
    ax.set_xlabel('Number of Turns')
    ax.set_ylabel('Average Score')
    ax.set_title('Score–Dialogue Length Relationship')
    ax.set_ylim(0, 5.4)
    ax.grid(color=EASE_COLORS['grid'], linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.10))
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'turn_score_correlation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    print(f"✓ 保存轮数-评分相关性图到：{output_file}")
    plt.close()

def generate_summary_report(turn_stats, comparisons, output_dir):
    """生成总结报告"""
    report_lines = []
    report_lines.append("# A5 实验：Multi-turn vs Single-turn 对比分析报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**虚拟医生**: Qwen3-32B\n")
    report_lines.append("---\n")
    
    # 1. 对话轮数统计
    report_lines.append("## 1. 对话轮数统计\n")
    
    for model, df in turn_stats.items():
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                          f"平均轮数 {df['total_turns'].mean():.1f}, "
                          f"范围 [{df['total_turns'].min()}, {df['total_turns'].max()}]")
    report_lines.append("")
    
    # 2. 轮数与评分相关性
    report_lines.append("## 2. 轮数与评分相关性\n")
    
    for model, comp in comparisons.items():
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                          f"相关系数 r = {comp['turn_score_corr']:.2f}, "
                          f"平均评分 {comp['avg_score']:.2f}")
    report_lines.append("")
    
    # 3. 完整实验设计方案
    report_lines.append("## 3. 完整 Multi-turn vs Single-turn 对比实验设计\n")
    
    report_lines.append("### 实验设置\n")
    report_lines.append("| 设置 | 描述 |")
    report_lines.append("|------|------|")
    report_lines.append("| 虚拟医生 | Qwen3-32B |")
    report_lines.append("| 评估模型 | 6 个模型（GPT-4o, Qwen3-0.6B/8B/14B/32B/235B） |")
    report_lines.append("| 案例数 | 50 cases |")
    report_lines.append("| 条件1 | Multi-turn（动态多轮对话） |")
    report_lines.append("| 条件2 | Single-turn（静态单轮问答） |")
    report_lines.append("")
    
    report_lines.append("### 评估指标\n")
    report_lines.append("- **准确性** (Accuracy)")
    report_lines.append("- **有效性** (Effectiveness)")
    report_lines.append("- **安全性** (Safety)")
    report_lines.append("- **个性化** (Personalization)")
    report_lines.append("- **共情能力** (Empathy)")
    report_lines.append("- **对话轮数**")
    report_lines.append("- **响应时间**")
    report_lines.append("")
    
    report_lines.append("### API 开销估算\n")
    report_lines.append("- **评估案例数**: 50 cases")
    report_lines.append("- **模型数**: 6 models")
    report_lines.append("- **实验条件**: 2 (multi-turn vs single-turn)")
    report_lines.append("- **每轮对话**: 平均 3-5 轮")
    report_lines.append(f"- **总 API 调用次数**: 约 {50 * 6 * 2 * 4} 次（含对话和评估）")
    report_lines.append("")
    
    report_lines.append("### 预期假设\n")
    report_lines.append("1. **Multi-turn 优势假设**: 多轮对话能更好地理解患者需求，提高准确性和个性化")
    report_lines.append("2. **Single-turn 效率假设**: 单轮问答响应更快，适合简单问题")
    report_lines.append("3. **复杂场景假设**: 在复杂场景下，多轮对话优势更明显")
    report_lines.append("")
    
    # 4. 关键发现
    report_lines.append("## 4. 关键发现\n")
    
    avg_corr = np.mean([comp['turn_score_corr'] for comp in comparisons.values()])
    report_lines.append(f"- **轮数-评分平均相关性**: r = {avg_corr:.2f}")
    
    # 找出轮数最多和最少的模型
    avg_turns = {model: comp['avg_turns'] for model, comp in comparisons.items()}
    most_turns = max(avg_turns, key=avg_turns.get)
    least_turns = min(avg_turns, key=avg_turns.get)
    report_lines.append(f"- **对话轮数最多**: {most_turns.replace('qwen3-', 'Qwen3-').upper()} "
                       f"({avg_turns[most_turns]:.1f} 轮)")
    report_lines.append(f"- **对话轮数最少**: {least_turns.replace('qwen3-', 'Qwen3-').upper()} "
                       f"({avg_turns[least_turns]:.1f} 轮)")
    
    report_lines.append("")
    
    report_lines.append("## 5. 建议\n")
    report_lines.append("基于当前分析，建议进行完整的 multi-turn vs single-turn 对比实验：")
    report_lines.append("1. 使用 Qwen3-32B 作为虚拟医生")
    report_lines.append("2. 对同一批案例分别进行多轮和单轮评估")
    report_lines.append("3. 比较两种模式在各维度的表现差异")
    report_lines.append("4. 分析不同场景下的表现差异")
    report_lines.append("")
    
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'multi_vs_single_summary.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 保存总结报告到：{report_file}")


def save_results(turn_stats, comparisons, output_dir):
    """保存所有结果"""
    # 转换 DataFrame 为字典
    turn_data = {}
    for model, df in turn_stats.items():
        # 处理相关性计算中的 NaN
        corr, _ = stats.pearsonr(df['total_turns'], df['overall_score'])
        if np.isnan(corr):
            corr = 0.0
        
        turn_data[model] = {
            'avg_turns': float(df['total_turns'].mean()),
            'min_turns': int(df['total_turns'].min()),
            'max_turns': int(df['total_turns'].max()),
            'avg_score': float(df['overall_score'].mean()),
            'turn_score_corr': float(corr),
            'turn_distribution': {int(k): int(v) for k, v in df['total_turns'].value_counts().sort_index().to_dict().items()}
        }
    
    # 简化 comparisons（只保留需要的字段）
    simplified_comparisons = {}
    for model, comp in comparisons.items():
        simplified_comparisons[model] = {
            'avg_turns': comp['avg_turns'],
            'avg_score': comp['avg_score'],
            'turn_score_corr': float(comp['turn_score_corr']) if not np.isnan(comp['turn_score_corr']) else 0.0
        }
    
    output = {
        'turn_data': turn_data,
        'comparisons': simplified_comparisons
    }
    
    output_file = os.path.join(output_dir, 'multi_vs_single_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存多轮vs单轮分析结果到：{output_file}")


def main():
    results_dir = 'outputs/model_evaluation_100cases'
    output_dir = 'outputs/experiments/A5_multi_vs_single'
    
    print("="*60)
    print("A5 实验：Multi-turn vs Single-turn 对比分析")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载所有模型结果...")
    all_data = load_all_results(results_dir)
    
    if not all_data:
        print("错误：未找到任何模型结果文件")
        return
    
    # 2. 分析对话轮数
    print("\n2. 分析对话轮数特征...")
    turn_stats = analyze_dialogue_turns(all_data)
    
    # 3. 比较不同轮数的表现
    print("\n3. 比较不同轮数的表现...")
    comparisons = compare_turn_performance(turn_stats)
    
    # 4. 生成可视化图表
    print("\n4. 生成可视化图表...")
    plot_turn_analysis(turn_stats, output_dir)
    
    # 5. 保存结果
    print("\n5. 保存分析结果...")
    save_results(turn_stats, comparisons, output_dir)
    
    # 6. 生成报告
    print("\n6. 生成总结报告...")
    generate_summary_report(turn_stats, comparisons, output_dir)
    
    print("\n" + "="*60)
    print("✓ A5 实验完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)
    print("\n⚠️  如需进行完整的 Multi-turn vs Single-turn 对比实验，")
    print(f"   需要运行约 {50 * 6 * 2 * 4} 次 API 调用。")
    print("   是否继续执行完整实验？")


if __name__ == '__main__':
    main()
