#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B3 实验：Case difficulty / risk-level stratification

按风险等级分层分析模型表现
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


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


def classify_risk_level(ehr_data):
    """
    根据 EHR 数据分类风险等级
    
    风险等级定义：
    - Low: 早期诊断，无严重症状，常规治疗
    - Medium: 中期诊断，有症状，需要关注
    - High: 晚期诊断，严重症状，复杂治疗
    - Safety-critical: 紧急情况，严重并发症，危及生命
    
    Returns:
        risk_level: 风险等级 (Low/Medium/High/Safety-critical)
        risk_score: 风险分数 (0-10)
    """
    risk_score = 0
    
    # 1. 基于病理类型评分
    pathology = ehr_data.get('pathology_type', '')
    pathology_scores = {
        '原位导管癌': 1,
        '浸润性导管癌': 3,
        '浸润性小叶癌': 3,
        '三阴性乳腺癌': 5,
        'HER2阳性': 4,
        '炎性乳腺癌': 7,
        '转移性乳腺癌': 8
    }
    risk_score += pathology_scores.get(pathology, 2)
    
    # 2. 基于分期评分
    stage = ehr_data.get('stage', '')
    stage_patterns = [
        (r'0期|原位', 1),
        (r'I期|1期', 2),
        (r'IIA期|2A期', 3),
        (r'IIB期|2B期', 4),
        (r'IIIA期|3A期', 5),
        (r'IIIB期|3B期', 6),
        (r'IIIC期|3C期', 7),
        (r'IV期|4期', 9)
    ]
    for pattern, score in stage_patterns:
        if pattern in stage:
            risk_score += score
            break
    
    # 3. 基于症状评分
    symptoms = ehr_data.get('current_symptoms', [])
    if isinstance(symptoms, list):
        symptoms_str = ' '.join(symptoms)
    else:
        symptoms_str = str(symptoms)
    
    # 严重症状
    severe_symptoms = ['疼痛剧烈', '呼吸困难', '发热', '出血', '肿胀明显', '无法进食']
    for symptom in severe_symptoms:
        if symptom in symptoms_str:
            risk_score += 2
    
    # 4. 基于治疗阶段评分
    treatment_stage = ehr_data.get('treatment_stage', '')
    treatment_scores = {
        '术前准备': 2,
        '手术治疗中': 3,
        '手术后恢复期': 2,
        '化疗中': 4,
        '放疗中': 4,
        '内分泌治疗中': 2,
        '靶向治疗中': 3,
        '晚期姑息治疗': 7,
        '复发治疗中': 6
    }
    risk_score += treatment_scores.get(treatment_stage, 1)
    
    # 5. 基于并发症评分
    concerns = ehr_data.get('concerns', '')
    if isinstance(concerns, list):
        concerns = ' '.join(concerns)
    
    complications = ['转移', '复发', '感染', '并发症', '危急', '紧急']
    for comp in complications:
        if comp in concerns:
            risk_score += 3
    
    # 归一化到 0-10 分
    risk_score = min(risk_score, 10)
    
    # 确定风险等级
    if risk_score <= 3:
        risk_level = 'Low'
    elif risk_score <= 5:
        risk_level = 'Medium'
    elif risk_score <= 7:
        risk_level = 'High'
    else:
        risk_level = 'Safety-critical'
    
    return risk_level, risk_score


def analyze_by_risk(all_data):
    """按风险等级分析模型表现"""
    risk_stats = {}
    
    for model, results in all_data.items():
        model_stats = {}
        
        for case in results:
            ehr_data = case.get('ehr_data', {})
            evaluation = case.get('evaluation', {})
            
            # 分类风险等级
            risk_level, risk_score = classify_risk_level(ehr_data)
            
            # 提取评分
            overall_score = evaluation.get('overall_score', 0)
            is_passed = evaluation.get('is_passed', False)
            
            # 记录统计
            if risk_level not in model_stats:
                model_stats[risk_level] = []
            
            model_stats[risk_level].append({
                'score': overall_score,
                'passed': is_passed,
                'risk_score': risk_score
            })
        
        risk_stats[model] = model_stats
    
    return risk_stats


def calculate_statistics(risk_stats):
    """计算各风险等级的统计指标"""
    stats = {}
    
    for model, risk_levels in risk_stats.items():
        stats[model] = {}
        
        for risk_level, cases in risk_levels.items():
            scores = [c['score'] for c in cases]
            passed = sum(1 for c in cases if c['passed'])
            total = len(cases)
            
            if total > 0:
                stats[model][risk_level] = {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'pass_rate': float(passed / total * 100),
                    'case_count': total,
                    'min_score': float(np.min(scores)),
                    'max_score': float(np.max(scores))
                }
    
    return stats


def plot_risk_performance(stats, output_dir):
    """绘制风险等级表现图"""
    models = list(stats.keys())
    risk_levels = ['Low', 'Medium', 'High', 'Safety-critical']
    
    # 准备数据
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图1: 平均分
    ax1 = axes[0]
    x = np.arange(len(risk_levels))
    width = 0.13
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    for i, model in enumerate(models):
        means = []
        for risk in risk_levels:
            means.append(stats[model].get(risk, {}).get('mean_score', 0))
        
        ax1.bar(x + i * width, means, width, 
                label=model.replace('qwen3-', 'Qwen3-').upper(),
                color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
    
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(risk_levels, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance by Risk Level', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 5.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: 通过率
    ax2 = axes[1]
    
    for i, model in enumerate(models):
        pass_rates = []
        for risk in risk_levels:
            pass_rates.append(stats[model].get(risk, {}).get('pass_rate', 0))
        
        ax2.bar(x + i * width, pass_rates, width, 
                label=model.replace('qwen3-', 'Qwen3-').upper(),
                color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
    
    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(risk_levels, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Pass Rate by Risk Level', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'risk_performance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存风险等级表现图到：{output_file}")
    
    plt.close()


def plot_risk_correlation(risk_stats, output_dir):
    """绘制风险分数与模型评分的相关性图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    models = list(risk_stats.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    for i, model in enumerate(models):
        ax = axes[i // 3, i % 3]
        
        # 收集所有数据点
        risk_scores = []
        model_scores = []
        
        for risk_level, cases in risk_stats[model].items():
            for case in cases:
                risk_scores.append(case['risk_score'])
                model_scores.append(case['score'])
        
        # 计算相关性
        if len(risk_scores) >= 2:
            corr_matrix = np.corrcoef(risk_scores, model_scores)
            # 处理标量情况
            if corr_matrix.ndim == 0:
                corr = float(corr_matrix)
            elif corr_matrix.ndim == 1:
                corr = 0
            else:
                corr = float(corr_matrix[0, 1])
        else:
            corr = 0
        
        # 散点图
        ax.scatter(risk_scores, model_scores, alpha=0.6, color=colors[i % len(colors)], s=50)
        
        # 添加趋势线
        z = np.polyfit(risk_scores, model_scores, 1)
        p = np.poly1d(z)
        ax.plot(risk_scores, p(risk_scores), "r--", linewidth=2)
        
        ax.set_title(f"{model.replace('qwen3-', 'Qwen3-').upper()}\n(r = {corr:.2f})", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Risk Score', fontsize=10)
        ax.set_ylabel('Model Score', fontsize=10)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(0, 5.5)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Correlation between Case Risk and Model Performance', 
                 fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'risk_correlation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存风险相关性图到：{output_file}")
    
    plt.close()


def generate_summary_table(stats, output_dir):
    """生成风险等级统计表"""
    models = list(stats.keys())
    risk_levels = ['Low', 'Medium', 'High', 'Safety-critical']
    
    data = []
    for model in models:
        row = {'Model': model}
        for risk in risk_levels:
            if risk in stats[model]:
                s = stats[model][risk]
                row[f'{risk}_score'] = f"{s['mean_score']:.2f}"
                row[f'{risk}_pass'] = f"{s['pass_rate']:.1f}%"
                row[f'{risk}_count'] = s['case_count']
            else:
                row[f'{risk}_score'] = 'N/A'
                row[f'{risk}_pass'] = 'N/A'
                row[f'{risk}_count'] = 0
        data.append(row)
    
    # 保存为 JSON
    results = {
        'models': models,
        'risk_levels': risk_levels,
        'statistics': stats,
        'summary': data
    }
    
    output_file = os.path.join(output_dir, 'risk_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存风险分析结果到：{output_file}")


def generate_summary_report(stats, risk_stats, output_dir):
    """生成总结报告"""
    models = list(stats.keys())
    risk_levels = ['Low', 'Medium', 'High', 'Safety-critical']
    
    report_lines = []
    report_lines.append("# B3 实验：Case Difficulty / Risk-level Stratification 报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**数据源**: 6 个模型 × 50 cases\n")
    report_lines.append("---\n")
    
    # 1. 风险等级分布
    report_lines.append("## 1. 风险等级分布\n")
    
    # 统计各风险等级的案例数
    risk_dist = {}
    for risk in risk_levels:
        count = sum(stats[m].get(risk, {}).get('case_count', 0) for m in models)
        risk_dist[risk] = count
    
    total = sum(risk_dist.values())
    for risk in risk_levels:
        report_lines.append(f"- **{risk}**: {risk_dist[risk]} cases ({risk_dist[risk]/total*100:.1f}%)")
    report_lines.append("")
    
    # 2. 各风险等级表现
    report_lines.append("## 2. 各风险等级模型表现\n")
    
    for risk in risk_levels:
        report_lines.append(f"### {risk} 风险等级\n")
        
        # 按平均分排序
        model_perf = []
        for model in models:
            if risk in stats[model]:
                model_perf.append({
                    'model': model,
                    'score': stats[model][risk]['mean_score'],
                    'pass_rate': stats[model][risk]['pass_rate']
                })
        
        model_perf.sort(key=lambda x: x['score'], reverse=True)
        
        for mp in model_perf:
            report_lines.append(f"- **{mp['model'].replace('qwen3-', 'Qwen3-').upper()}**: "
                              f"平均分 {mp['score']:.2f}, 通过率 {mp['pass_rate']:.1f}%")
        report_lines.append("")
    
    # 3. 模型在不同风险等级的表现变化
    report_lines.append("## 3. 模型跨风险等级稳定性\n")
    
    for model in models:
        scores = []
        for risk in risk_levels:
            if risk in stats[model]:
                scores.append(stats[model][risk]['mean_score'])
        
        if scores:
            report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                              f"平均分范围 {min(scores):.2f}-{max(scores):.2f}, "
                              f"变化幅度 {max(scores)-min(scores):.2f}")
    report_lines.append("")
    
    # 4. 风险-评分相关性
    report_lines.append("## 4. 风险-评分相关性分析\n")
    
    for model in models:
        risk_scores = []
        model_scores = []
        
        for risk_level, cases in risk_stats[model].items():
            for case in cases:
                risk_scores.append(case['risk_score'])
                model_scores.append(case['score'])
        
        if len(risk_scores) >= 2:
            corr_matrix = np.corrcoef(risk_scores, model_scores)
            if corr_matrix.ndim == 0:
                corr = float(corr_matrix)
            elif corr_matrix.ndim == 1:
                corr = 0
            else:
                corr = float(corr_matrix[0, 1])
            report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                              f"相关系数 r = {corr:.2f}")
    report_lines.append("")
    
    # 5. 关键发现
    report_lines.append("## 5. 关键发现\n")
    
    # 最难风险等级
    overall_perf = {}
    for risk in risk_levels:
        scores = []
        for model in models:
            if risk in stats[model]:
                scores.append(stats[model][risk]['mean_score'])
        if scores:
            overall_perf[risk] = np.mean(scores)
    
    hardest = min(overall_perf, key=overall_perf.get)
    easiest = max(overall_perf, key=overall_perf.get)
    
    report_lines.append(f"- **最难风险等级**: {hardest} (平均分 {overall_perf[hardest]:.2f})")
    report_lines.append(f"- **最容易风险等级**: {easiest} (平均分 {overall_perf[easiest]:.2f})")
    
    # 最稳定模型
    stability = {}
    for model in models:
        scores = []
        for risk in risk_levels:
            if risk in stats[model]:
                scores.append(stats[model][risk]['mean_score'])
        if len(scores) >= 2:
            stability[model] = np.std(scores)
    
    most_stable = min(stability, key=stability.get)
    report_lines.append(f"- **跨风险等级最稳定**: {most_stable.replace('qwen3-', 'Qwen3-').upper()} "
                       f"(标准差 {stability[most_stable]:.2f})")
    
    report_lines.append("")
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'risk_summary.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 保存总结报告到：{report_file}")


def main():
    results_dir = 'outputs/model_evaluation_50cases'
    output_dir = 'outputs/experiments/B3_risk_stratification'
    
    print("="*60)
    print("B3 实验：Case Difficulty / Risk-level Stratification")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载所有模型结果...")
    all_data = load_all_results(results_dir)
    
    if not all_data:
        print("错误：未找到任何模型结果文件")
        return
    
    # 2. 按风险等级分析
    print("\n2. 按风险等级分析模型表现...")
    risk_stats = analyze_by_risk(all_data)
    
    # 3. 计算统计指标
    print("\n3. 计算统计指标...")
    stats = calculate_statistics(risk_stats)
    
    # 4. 生成可视化图表
    print("\n4. 生成可视化图表...")
    plot_risk_performance(stats, output_dir)
    plot_risk_correlation(risk_stats, output_dir)
    
    # 5. 保存结果
    print("\n5. 保存分析结果...")
    generate_summary_table(stats, output_dir)
    
    # 6. 生成报告
    print("\n6. 生成总结报告...")
    generate_summary_report(stats, risk_stats, output_dir)
    
    print("\n" + "="*60)
    print("✓ B3 实验完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
