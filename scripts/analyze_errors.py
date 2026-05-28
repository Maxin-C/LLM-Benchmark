#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B4 实验：Error taxonomy / failure mode analysis

分析低分案例的错误类型和失败模式
"""

import json
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 定义错误类型分类体系
ERROR_TYPES = {
    'factual_error': {
        'name': '事实错误',
        'description': '提供了错误的医学事实或信息',
        'keywords': ['错误', '不正确', '不是', '没有', '错误地', '误']
    },
    'missing_info': {
        'name': '信息缺失',
        'description': '遗漏了关键信息或未回答问题',
        'keywords': ['不知道', '不清楚', '无法回答', '未提及', '缺少', '遗漏']
    },
    'unsafe_recommendation': {
        'name': '不安全建议',
        'description': '提供了可能有害的建议',
        'keywords': ['建议', '应该', '可以', '不要', '禁止', '避免']
    },
    'inconsistency': {
        'name': '前后矛盾',
        'description': '回答内部存在矛盾或不一致',
        'keywords': ['但是', '然而', '矛盾', '不一致', '相反', '却']
    },
    'overconfidence': {
        'name': '过度自信',
        'description': '在不确定的情况下给出肯定的结论',
        'keywords': ['肯定', '一定', '绝对', '毫无疑问', '显然', '必然']
    },
    'underconfidence': {
        'name': '过度保守',
        'description': '过于谨慎，未能提供足够的信息',
        'keywords': ['可能', '也许', '或许', '不确定', '建议咨询']
    },
    'communication_issue': {
        'name': '沟通问题',
        'description': '语言表达不清、生硬或缺乏共情',
        'keywords': ['抱歉', '不好意思', '简单来说', '直白地说', '老实说']
    },
    'irrelevant_response': {
        'name': '答非所问',
        'description': '回答与问题无关或偏离主题',
        'keywords': ['另外', '顺便说', '补充一下', '关于']
    },
    'medical_knowledge_gap': {
        'name': '医学知识缺失',
        'description': '缺乏必要的医学知识',
        'keywords': ['根据我的知识', '据我所知', '研究表明', '医学上']
    },
    'treatment_error': {
        'name': '治疗建议错误',
        'description': '提供了错误的治疗建议',
        'keywords': ['治疗', '药物', '手术', '化疗', '放疗', '内分泌']
    }
}


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


def identify_low_score_cases(all_data, threshold=3):
    """识别低分案例"""
    low_score_cases = {}
    
    for model, results in all_data.items():
        model_low_cases = []
        
        for case in results:
            evaluation = case.get('evaluation', {})
            overall_score = evaluation.get('overall_score', 5)
            
            if overall_score < threshold:
                model_low_cases.append(case)
        
        low_score_cases[model] = model_low_cases
        print(f"  {model}: {len(model_low_cases)} 个低分案例")
    
    return low_score_cases


def analyze_error_types(case):
    """分析单个案例的错误类型"""
    errors = []
    
    # 获取医生回答
    dialogue_history = case.get('dialogue_history', [])
    doctor_response = ""
    for turn in dialogue_history:
        role = turn.get('role', '').lower()
        if role == 'assistant' or role == 'doctor':
            doctor_response += turn.get('content', '') + " "
    
    # 获取评估评论（如果有）
    evaluation = case.get('evaluation', {})
    comments = evaluation.get('comments', '')
    
    # 分析错误类型
    for error_key, error_info in ERROR_TYPES.items():
        keywords = error_info['keywords']
        found = False
        
        # 在医生回答中查找关键词
        for keyword in keywords:
            if keyword in doctor_response:
                found = True
                break
        
        # 在评论中查找关键词
        if not found and comments:
            for keyword in keywords:
                if keyword in comments:
                    found = True
                    break
        
        if found:
            errors.append({
                'type': error_key,
                'name': error_info['name'],
                'description': error_info['description']
            })
    
    # 如果没有识别到错误类型，添加默认类型
    if not errors:
        errors.append({
            'type': 'other',
            'name': '其他错误',
            'description': '未能明确分类的错误'
        })
    
    return errors


def analyze_all_low_cases(low_score_cases):
    """分析所有低分案例"""
    results = {}
    error_stats = defaultdict(lambda: defaultdict(int))
    
    for model, cases in low_score_cases.items():
        model_results = []
        
        for case in cases:
            errors = analyze_error_types(case)
            
            # 记录错误类型
            for error in errors:
                error_stats[model][error['type']] += 1
            
            model_results.append({
                'case_id': case.get('case_id', ''),
                'overall_score': case.get('evaluation', {}).get('overall_score', 0),
                'errors': errors,
                'dialogue_history': case.get('dialogue_history', []),
                'ehr_data': case.get('ehr_data', {})
            })
        
        results[model] = model_results
    
    return results, error_stats


def calculate_statistics(error_stats, low_score_cases):
    """计算统计指标"""
    stats = {}
    
    for model, error_counts in error_stats.items():
        total_errors = sum(error_counts.values())
        total_cases = len(low_score_cases.get(model, []))
        
        stats[model] = {
            'total_low_cases': total_cases,
            'total_errors': total_errors,
            'errors_per_case': total_errors / max(total_cases, 1),
            'error_distribution': dict(error_counts),
            'top_errors': sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    return stats


def plot_error_distribution(error_stats, output_dir):
    """绘制错误类型分布"""
    models = list(error_stats.keys())
    error_types = list(ERROR_TYPES.keys())
    
    # 准备数据
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(error_types))
    width = 0.13
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    for i, model in enumerate(models):
        counts = []
        for error_type in error_types:
            counts.append(error_stats[model].get(error_type, 0))
        
        ax.bar(x + i * width, counts, width, 
                label=model.replace('qwen3-', 'Qwen3-').upper(),
                color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([ERROR_TYPES[e]['name'] for e in error_types], 
                       fontsize=10, fontweight='bold', rotation=45)
    ax.set_ylabel('Error Count', fontsize=12, fontweight='bold')
    ax.set_title('Error Type Distribution by Model', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 保存错误类型分布图到：{output_file}")
    
    plt.close()


def generate_error_examples(results, output_dir, max_examples=3):
    """生成错误案例示例"""
    examples = []
    
    for model, cases in results.items():
        # 按错误数量排序
        sorted_cases = sorted(cases, key=lambda x: len(x['errors']), reverse=True)
        
        for case in sorted_cases[:max_examples]:
            examples.append({
                'model': model,
                'case_id': case['case_id'],
                'score': case['overall_score'],
                'errors': case['errors'],
                'dialogue_history': case['dialogue_history']
            })
    
    # 保存示例
    examples_file = os.path.join(output_dir, 'error_examples.md')
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write("# B4 实验：错误类型案例示例\n\n")
        f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**示例数量**: {len(examples)}\n\n")
        
        for i, example in enumerate(examples, 1):
            f.write(f"## 示例 {i}\n\n")
            f.write(f"- **模型**: {example['model'].replace('qwen3-', 'Qwen3-').upper()}\n")
            f.write(f"- **案例 ID**: {example['case_id']}\n")
            f.write(f"- **评分**: {example['score']}\n")
            f.write(f"- **错误数量**: {len(example['errors'])}\n\n")
            
            f.write("### 错误类型\n")
            for j, error in enumerate(example['errors'], 1):
                f.write(f"{j}. **{error['name']}**: {error['description']}\n")
            
            f.write("\n### 对话摘要\n")
            # 提取医生回答
            for turn in example['dialogue_history']:
                role = turn.get('role', '')
                content = turn.get('content', '')[:200] + "..." if len(turn.get('content', '')) > 200 else turn.get('content', '')
                f.write(f"- **{role}**: {content}\n")
            
            f.write("\n---\n\n")
    
    print(f"✓ 保存错误案例示例到：{examples_file}")


def generate_summary_report(stats, error_stats, output_dir):
    """生成总结报告"""
    models = list(stats.keys())
    
    report_lines = []
    report_lines.append("# B4 实验：Error Taxonomy / Failure Mode Analysis 报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**数据源**: 6 个模型的低分案例\n")
    report_lines.append("---\n")
    
    # 1. 低分案例统计
    report_lines.append("## 1. 低分案例统计\n")
    
    total_low_cases = sum(stats[m]['total_low_cases'] for m in models)
    report_lines.append(f"- **总低分案例数**: {total_low_cases}\n\n")
    
    for model in models:
        s = stats[model]
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                          f"{s['total_low_cases']} 个低分案例, "
                          f"平均每案例 {s['errors_per_case']:.2f} 个错误")
    report_lines.append("")
    
    # 2. 错误类型分布
    report_lines.append("## 2. 错误类型分布\n")
    
    # 统计每种错误类型的总数
    total_by_error = defaultdict(int)
    for model, errors in error_stats.items():
        for error_type, count in errors.items():
            total_by_error[error_type] += count
    
    sorted_errors = sorted(total_by_error.items(), key=lambda x: x[1], reverse=True)
    
    for error_type, count in sorted_errors:
        report_lines.append(f"- **{ERROR_TYPES.get(error_type, {'name': error_type})['name']}**: {count} 次")
    report_lines.append("")
    
    # 3. 各模型主要错误类型
    report_lines.append("## 3. 各模型主要错误类型\n")
    
    for model in models:
        top_errors = stats[model]['top_errors']
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**:")
        for error_type, count in top_errors:
            report_lines.append(f"  - {ERROR_TYPES.get(error_type, {'name': error_type})['name']}: {count} 次")
    report_lines.append("")
    
    # 4. 错误类型定义
    report_lines.append("## 4. 错误类型定义\n")
    
    for error_key, error_info in ERROR_TYPES.items():
        report_lines.append(f"### {error_info['name']}")
        report_lines.append(f"- **描述**: {error_info['description']}")
        report_lines.append(f"- **关键词**: {', '.join(error_info['keywords'])}")
        report_lines.append("")
    
    # 5. 关键发现
    report_lines.append("## 5. 关键发现\n")
    
    # 最常见错误类型
    most_common = sorted_errors[0]
    report_lines.append(f"- **最常见错误类型**: {ERROR_TYPES.get(most_common[0], {'name': most_common[0]})['name']} "
                       f"({most_common[1]} 次, "
                       f"{most_common[1]/sum(total_by_error.values())*100:.1f}%)")
    
    # 错误最多的模型
    most_errors = max(models, key=lambda m: stats[m]['total_errors'])
    report_lines.append(f"- **错误最多的模型**: {most_errors.replace('qwen3-', 'Qwen3-').upper()} "
                       f"({stats[most_errors]['total_errors']} 个错误)")
    
    # 错误最少的模型
    fewest_errors = min(models, key=lambda m: stats[m]['total_errors'])
    report_lines.append(f"- **错误最少的模型**: {fewest_errors.replace('qwen3-', 'Qwen3-').upper()} "
                       f"({stats[fewest_errors]['total_errors']} 个错误)")
    
    report_lines.append("")
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'error_summary.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 保存总结报告到：{report_file}")


def save_results(results, stats, error_stats, output_dir):
    """保存所有结果"""
    output = {
        'results': results,
        'statistics': stats,
        'error_stats': dict(error_stats)
    }
    
    output_file = os.path.join(output_dir, 'error_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存错误分析结果到：{output_file}")


def main():
    results_dir = 'outputs/model_evaluation_50cases'
    output_dir = 'outputs/experiments/B4_error_taxonomy'
    
    print("="*60)
    print("B4 实验：Error Taxonomy / Failure Mode Analysis")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载所有模型结果...")
    all_data = load_all_results(results_dir)
    
    if not all_data:
        print("错误：未找到任何模型结果文件")
        return
    
    # 2. 识别低分案例
    print("\n2. 识别低分案例（评分 < 3）...")
    low_score_cases = identify_low_score_cases(all_data, threshold=3)
    
    # 3. 分析错误类型
    print("\n3. 分析错误类型...")
    results, error_stats = analyze_all_low_cases(low_score_cases)
    
    # 4. 计算统计指标
    print("\n4. 计算统计指标...")
    stats = calculate_statistics(error_stats, low_score_cases)
    
    # 5. 生成可视化图表
    print("\n5. 生成可视化图表...")
    plot_error_distribution(error_stats, output_dir)
    
    # 6. 生成错误案例示例
    print("\n6. 生成错误案例示例...")
    generate_error_examples(results, output_dir)
    
    # 7. 保存结果
    print("\n7. 保存分析结果...")
    save_results(results, stats, error_stats, output_dir)
    
    # 8. 生成报告
    print("\n8. 生成总结报告...")
    generate_summary_report(stats, error_stats, output_dir)
    
    print("\n" + "="*60)
    print("✓ B4 实验完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
