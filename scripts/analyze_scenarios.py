#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1 实验：Scenario-level 分层结果分析

基于已有评估数据，按乳腺癌临床场景分层分析模型表现
"""

import json
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 定义场景分类规则
SCENARIO_KEYWORDS = {
    'Recovery': ['康复', '恢复', '锻炼', '运动', '淋巴水肿', '手臂肿胀', '关节僵硬', '功能恢复'],
    'Surgery-related': ['手术', '伤口', '切口', '红肿', '渗液', '疼痛', '引流管', '拆线'],
    'Pain': ['疼痛', '酸痛', '胀痛', '关节痛', '头痛', '肌肉痛'],
    'Endocrine therapy': ['内分泌', '他莫昔芬', '来曲唑', '阿那曲唑', '依西美坦', '潮热', '关节痛', '骨密度'],
    'Chemotherapy': ['化疗', '恶心', '呕吐', '脱发', '乏力', '白细胞', '免疫', '靶向治疗'],
    'Follow-up': ['复查', '随访', '检查', '超声', '钼靶', '肿瘤标志物', '骨密度'],
    'Medication side effects': ['副作用', '不良反应', '恶心', '潮热', '疲劳', '关节痛'],
    'Psychological support': ['焦虑', '担心', '害怕', '抑郁', '情绪', '压力', '心理'],
    'Nutrition': ['饮食', '营养', '食物', '忌口', '发物', '蛋白质', '维生素'],
    'Work & Life': ['工作', '生活', '家务', '社交', '运动', '旅行'],
    'Family planning': ['生育', '怀孕', '避孕', '月经', '绝经'],
    'Genetic counseling': ['基因', '遗传', 'BRCA', '家族史'],
    'Complementary therapy': ['中医', '针灸', '按摩', '保健品', '替代疗法']
}


def classify_scenario(ehr_data, conversation=None):
    """
    基于 EHR 数据和对话内容分类场景
    返回最匹配的场景类别
    """
    # 收集所有文本信息
    texts = []
    
    # 从 EHR 提取（确保是字符串）
    if 'current_symptoms' in ehr_data:
        symptoms = ehr_data['current_symptoms']
        if isinstance(symptoms, list):
            texts.extend([str(s) for s in symptoms if s])
        elif symptoms:
            texts.append(str(symptoms))
    
    if 'concerns' in ehr_data:
        concerns = ehr_data['concerns']
        if isinstance(concerns, list):
            texts.extend([str(c) for c in concerns if c])
        elif concerns:
            texts.append(str(concerns))
    
    if 'diagnosis' in ehr_data:
        diagnosis = ehr_data['diagnosis']
        if isinstance(diagnosis, list):
            texts.extend([str(d) for d in diagnosis if d])
        elif diagnosis:
            texts.append(str(diagnosis))
    
    if 'pathology_type' in ehr_data:
        pathology = ehr_data['pathology_type']
        if isinstance(pathology, list):
            texts.extend([str(p) for p in pathology if p])
        elif pathology:
            texts.append(str(pathology))
    
    # 从对话第一轮提取（患者主诉）
    if conversation and len(conversation) > 0:
        first_patient_msg = conversation[0].get('content', '')
        if first_patient_msg:
            texts.append(str(first_patient_msg))
    
    all_text = ' '.join(texts) if texts else ''
    
    # 匹配场景关键词
    scenario_scores = {}
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in all_text)
        scenario_scores[scenario] = score
    
    # 返回得分最高的场景
    if max(scenario_scores.values()) > 0:
        return max(scenario_scores, key=scenario_scores.get)
    else:
        # 默认场景
        return 'Other'


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


def analyze_by_scenario(all_data):
    """按场景分析模型表现"""
    scenario_stats = defaultdict(lambda: defaultdict(list))
    
    for model, results in all_data.items():
        for case in results:
            # 分类场景
            scenario = classify_scenario(
                case.get('ehr_data', {}),
                case.get('dialogue_history', [])
            )
            
            # 提取评分
            evaluation = case.get('evaluation', {})
            overall_score = evaluation.get('overall_score', 0)
            is_passed = evaluation.get('is_passed', False)
            
            # 记录统计
            scenario_stats[scenario][model].append({
                'score': overall_score,
                'passed': is_passed
            })
    
    return scenario_stats


def generate_summary_table(scenario_stats, output_dir):
    """生成场景 - 模型汇总表"""
    models = ['gpt-4o', 'qwen3-0.6b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b']
    scenarios = sorted(scenario_stats.keys())
    
    # 准备数据
    data = []
    for scenario in scenarios:
        row = {'Scenario': scenario}
        for model in models:
            if model in scenario_stats[scenario]:
                scores = [item['score'] for item in scenario_stats[scenario][model]]
                passed = sum(1 for item in scenario_stats[scenario][model] if item['passed'])
                total = len(scores)
                avg_score = sum(scores) / len(scores) if scores else 0
                pass_rate = passed / total * 100 if total > 0 else 0
                row[f'{model}_score'] = f"{avg_score:.2f}"
                row[f'{model}_pass'] = f"{pass_rate:.0f}%"
            else:
                row[f'{model}_score'] = 'N/A'
                row[f'{model}_pass'] = 'N/A'
        data.append(row)
    
    # 保存为 JSON
    summary_file = os.path.join(output_dir, 'scenario_results.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({'scenarios': scenarios, 'models': models, 'data': data}, 
                  f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存场景分析结果到：{summary_file}")
    
    return data, scenarios, models


def create_heatmap(scenario_stats, output_dir):
    """生成 model × scenario heatmap"""
    models = ['gpt-4o', 'qwen3-0.6b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b']
    scenarios = sorted(scenario_stats.keys())
    
    # 构建热力图数据矩阵
    heatmap_data = np.zeros((len(scenarios), len(models)))
    
    for i, scenario in enumerate(scenarios):
        for j, model in enumerate(models):
            if model in scenario_stats[scenario]:
                scores = [item['score'] for item in scenario_stats[scenario][model]]
                if scores:
                    heatmap_data[i, j] = np.mean(scores)
                else:
                    heatmap_data[i, j] = np.nan
            else:
                heatmap_data[i, j] = np.nan
    
    # 创建热力图
    plt.figure(figsize=(14, 10))
    
    # 处理 NaN 值
    mask = np.isnan(heatmap_data)
    heatmap_data_filled = np.ma.masked_array(heatmap_data, mask)
    
    # 绘制热力图
    ax = sns.heatmap(heatmap_data_filled, 
                     annot=True, 
                     fmt='.2f',
                     cmap='RdYlGn',
                     vmin=1, 
                     vmax=5,
                     center=3,
                     xticklabels=[m.replace('qwen3-', 'Qwen3-').upper() for m in models],
                     yticklabels=scenarios,
                     cbar_kws={'label': 'Average Score'},
                     linewidths=0.5)
    
    plt.title('Model Performance Across Breast Cancer Consultation Scenarios', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Scenario', fontsize=12)
    
    # 旋转 y 轴标签
    plt.yticks(rotation=0, fontsize=9)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(output_dir, 'scenario_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存热力图到：{output_file}")
    
    plt.close()


def generate_summary_report(scenario_stats, output_dir):
    """生成文字总结报告"""
    models = ['gpt-4o', 'qwen3-0.6b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b']
    scenarios = sorted(scenario_stats.keys())
    
    report_lines = []
    report_lines.append("# A1 实验：Scenario-level 分层结果分析报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**数据源**: 6 个模型 × 50 cases\n")
    report_lines.append(f"**场景分类数**: {len(scenarios)}\n")
    report_lines.append("---\n")
    
    # 1. 总体统计
    report_lines.append("## 1. 场景分布统计\n")
    for scenario in scenarios:
        total_cases = sum(len(scenario_stats[scenario][model]) for model in models if model in scenario_stats[scenario])
        report_lines.append(f"- **{scenario}**: {total_cases} cases")
    report_lines.append("")
    
    # 2. 各场景最佳/最差模型
    report_lines.append("## 2. 各场景表现最佳模型\n")
    for scenario in scenarios:
        best_model = None
        best_score = 0
        worst_model = None
        worst_score = 5
        
        for model in models:
            if model in scenario_stats[scenario]:
                scores = [item['score'] for item in scenario_stats[scenario][model]]
                if scores:
                    avg = np.mean(scores)
                    if avg > best_score:
                        best_score = avg
                        best_model = model
                    if avg < worst_score:
                        worst_score = avg
                        worst_model = model
        
        if best_model:
            report_lines.append(f"- **{scenario}**: 最佳 {best_model} ({best_score:.2f}), 最差 {worst_model} ({worst_score:.2f})")
    report_lines.append("")
    
    # 3. 模型跨场景稳定性
    report_lines.append("## 3. 模型跨场景稳定性分析\n")
    model_stability = {}
    for model in models:
        all_scores = []
        for scenario in scenarios:
            if model in scenario_stats[scenario]:
                scores = [item['score'] for item in scenario_stats[scenario][model]]
                if scores:
                    all_scores.append(np.mean(scores))
        
        if all_scores:
            model_stability[model] = {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'min': np.min(all_scores),
                'max': np.max(all_scores)
            }
    
    # 按平均分排序
    sorted_models = sorted(model_stability.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for model, stats in sorted_models:
        report_lines.append(f"- **{model}**: 平均分 {stats['mean']:.2f} ± {stats['std']:.2f} (范围：{stats['min']:.2f}-{stats['max']:.2f})")
    report_lines.append("")
    
    # 4. 关键发现
    report_lines.append("## 4. 关键发现\n")
    
    # 找出最难场景
    scenario_difficulty = []
    for scenario in scenarios:
        all_scores = []
        for model in models:
            if model in scenario_stats[scenario]:
                scores = [item['score'] for item in scenario_stats[scenario][model]]
                all_scores.extend(scores)
        if all_scores:
            scenario_difficulty.append((scenario, np.mean(all_scores)))
    
    scenario_difficulty.sort(key=lambda x: x[1])
    
    if scenario_difficulty:
        easiest = scenario_difficulty[-1]
        hardest = scenario_difficulty[0]
        report_lines.append(f"- **最容易场景**: {easiest[0]} (平均分 {easiest[1]:.2f})")
        report_lines.append(f"- **最难场景**: {hardest[0]} (平均分 {hardest[1]:.2f})")
    
    # 找出表现最稳定的模型
    if model_stability:
        most_stable = min(model_stability.items(), key=lambda x: x[1]['std'])
        report_lines.append(f"- **最稳定模型**: {most_stable[0]} (标准差 {most_stable[1]['std']:.2f})")
    
    report_lines.append("")
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'scenario_summary.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 保存总结报告到：{report_file}")


def main():
    results_dir = 'outputs/model_evaluation_50cases'
    output_dir = 'outputs/experiments/A1_scenario_analysis'
    
    print("="*60)
    print("A1 实验：Scenario-level 分层结果分析")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载所有模型结果...")
    all_data = load_all_results(results_dir)
    
    if not all_data:
        print("错误：未找到任何模型结果文件")
        return
    
    # 2. 场景分类和统计
    print("\n2. 进行场景分类和统计...")
    scenario_stats = analyze_by_scenario(all_data)
    print(f"识别到 {len(scenario_stats)} 个场景类别")
    
    # 3. 生成汇总表
    print("\n3. 生成场景 - 模型汇总表...")
    data, scenarios, models = generate_summary_table(scenario_stats, output_dir)
    
    # 4. 生成热力图
    print("\n4. 生成 model × scenario heatmap...")
    create_heatmap(scenario_stats, output_dir)
    
    # 5. 生成总结报告
    print("\n5. 生成总结报告...")
    generate_summary_report(scenario_stats, output_dir)
    
    print("\n" + "="*60)
    print("✓ A1 实验完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
