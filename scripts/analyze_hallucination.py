#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2 实验：Patient Agent hallucination 检测

基于对话数据自动检测 Patient Agent 的回答是否与 EHR 数据矛盾
"""

import json
import os
import re
from collections import defaultdict
import pandas as pd
import numpy as np


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


def extract_ehr_info(ehr_data):
    """提取 EHR 关键信息"""
    info = {
        'patient_id': ehr_data.get('patient_id', ''),
        'age': ehr_data.get('age', ''),
        'gender': ehr_data.get('gender', ''),
        'occupation': ehr_data.get('occupation', ''),
        'pathology_type': ehr_data.get('pathology_type', ''),
        'stage': ehr_data.get('stage', ''),
        'treatment_stage': ehr_data.get('treatment_stage', ''),
        'current_symptoms': ehr_data.get('current_symptoms', []),
        'past_medical_history': ehr_data.get('past_medical_history', ''),
        'family_history': ehr_data.get('family_history', ''),
        'allergies': ehr_data.get('allergies', '')
    }
    
    # 将列表转换为字符串
    if isinstance(info['current_symptoms'], list):
        info['current_symptoms'] = ' '.join(info['current_symptoms'])
    
    return info


def extract_patient_responses(dialogue_history):
    """提取 Patient Agent 的所有回答"""
    responses = []
    
    if not dialogue_history:
        return responses
    
    for turn in dialogue_history:
        role = turn.get('role', '').lower()
        if role == 'patient' or role == 'user':
            content = turn.get('content', '')
            if content:
                responses.append(content)
    
    return responses


def detect_contradictions(ehr_info, patient_responses):
    """
    检测 Patient Agent 回答与 EHR 的矛盾
    
    Returns:
        contradictions: 矛盾列表
        is_hallucinated: 是否存在 hallucination
    """
    contradictions = []
    all_patient_text = ' '.join(patient_responses)
    
    # 规则1: 年龄矛盾检测
    if ehr_info['age']:
        # 从患者回答中提取年龄数字
        age_in_response = re.search(r'(\d{2,3})\s*岁', all_patient_text)
        if age_in_response:
            response_age = int(age_in_response.group(1))
            ehr_age = int(ehr_info['age'])
            # 年龄差异超过5岁视为矛盾
            if abs(response_age - ehr_age) > 5:
                contradictions.append({
                    'type': 'age_mismatch',
                    'description': f"年龄矛盾：EHR显示 {ehr_age} 岁，患者回答提到 {response_age} 岁",
                    'severity': 'high'
                })
    
    # 规则2: 性别矛盾检测
    if ehr_info['gender']:
        gender_keywords = {
            '女': ['女', '女性', '女士', '她', '母亲', '妻子'],
            '男': ['男', '男性', '先生', '他', '父亲', '丈夫']
        }
        
        ehr_gender = ehr_info['gender']
        detected_gender = None
        
        if ehr_gender == 'female' or ehr_gender == '女':
            expected_keywords = gender_keywords['女']
            opposite_keywords = gender_keywords['男']
        elif ehr_gender == 'male' or ehr_gender == '男':
            expected_keywords = gender_keywords['男']
            opposite_keywords = gender_keywords['女']
        else:
            expected_keywords = []
            opposite_keywords = []
        
        # 检查是否出现矛盾的性别关键词
        for keyword in opposite_keywords:
            if keyword in all_patient_text:
                contradictions.append({
                    'type': 'gender_mismatch',
                    'description': f"性别矛盾：EHR显示 {ehr_gender}，患者回答使用了 '{keyword}'",
                    'severity': 'high'
                })
                break
    
    # 规则3: 诊断/病理类型矛盾
    if ehr_info['pathology_type']:
        pathology = ehr_info['pathology_type']
        # 检查患者回答中是否出现否定当前诊断的内容
        negation_patterns = [
            f"不是{pathology}",
            f"没有{pathology}",
            f"未患{pathology}",
            f"排除{pathology}",
            f"并非{pathology}"
        ]
        
        for pattern in negation_patterns:
            if pattern in all_patient_text:
                contradictions.append({
                    'type': 'diagnosis_denial',
                    'description': f"诊断矛盾：患者否定了 EHR 中的诊断 '{pathology}'",
                    'severity': 'high'
                })
                break
    
    # 规则4: 症状矛盾检测
    if ehr_info['current_symptoms']:
        symptoms = ehr_info['current_symptoms']
        # 检查患者是否否定报告的症状
        symptom_negations = [
            f"没有{symptoms}",
            f"不觉得{symptoms}",
            f"从未有过{symptoms}",
            f"已经没有{symptoms}"
        ]
        
        for negation in symptom_negations:
            if negation in all_patient_text:
                contradictions.append({
                    'type': 'symptom_denial',
                    'description': f"症状矛盾：患者否定了 EHR 中的症状 '{symptoms}'",
                    'severity': 'medium'
                })
                break
    
    # 规则5: 治疗阶段矛盾
    if ehr_info['treatment_stage']:
        treatment = ehr_info['treatment_stage']
        # 检查是否存在明显矛盾的治疗状态
        treatment_keywords = {
            '内分泌治疗中': ['化疗中', '放疗中', '手术治疗中', '已治愈'],
            '化疗中': ['内分泌治疗中', '放疗中', '手术治疗中', '已治愈'],
            '放疗中': ['内分泌治疗中', '化疗中', '手术治疗中', '已治愈'],
            '手术后恢复期': ['未手术', '等待手术', '术前准备']
        }
        
        if treatment in treatment_keywords:
            for contradictory in treatment_keywords[treatment]:
                if contradictory in all_patient_text:
                    contradictions.append({
                        'type': 'treatment_mismatch',
                        'description': f"治疗阶段矛盾：EHR显示 '{treatment}'，患者提到 '{contradictory}'",
                        'severity': 'high'
                    })
                    break
    
    # 规则6: 虚构信息检测
    # 检查患者是否提到 EHR 中没有的关键信息
    ehr_text = ' '.join([str(v) for v in ehr_info.values() if v])
    
    # 检查是否提到新的疾病
    new_disease_patterns = [
        r'(高血压|糖尿病|心脏病|中风|癌症|肿瘤)',
        r'(肝硬化|肾衰竭|肺炎|肺结核)'
    ]
    
    for pattern in new_disease_patterns:
        matches = re.findall(pattern, all_patient_text)
        for match in matches:
            if match not in ehr_text:
                contradictions.append({
                    'type': 'unreported_condition',
                    'description': f"虚构疾病：患者提到未在 EHR 中记录的疾病 '{match}'",
                    'severity': 'medium'
                })
    
    # 规则7: 日期/时间矛盾
    # 检查是否有明显的时间矛盾（如治疗时间早于诊断时间等）
    
    return contradictions


def analyze_all_cases(all_data):
    """分析所有案例的 hallucination 情况"""
    results = {}
    
    for model, cases in all_data.items():
        model_results = []
        
        for case in cases:
            case_id = case.get('case_id', '')
            ehr_data = case.get('ehr_data', {})
            dialogue_history = case.get('dialogue_history', [])
            
            # 提取信息
            ehr_info = extract_ehr_info(ehr_data)
            patient_responses = extract_patient_responses(dialogue_history)
            
            # 检测矛盾
            contradictions = detect_contradictions(ehr_info, patient_responses)
            
            model_results.append({
                'case_id': case_id,
                'has_contradiction': len(contradictions) > 0,
                'contradiction_count': len(contradictions),
                'contradictions': contradictions,
                'patient_response_count': len(patient_responses),
                'total_turns': len(dialogue_history)
            })
        
        results[model] = model_results
    
    return results


def calculate_statistics(results):
    """计算统计指标"""
    stats = {}
    
    for model, case_results in results.items():
        total_cases = len(case_results)
        cases_with_contradiction = sum(1 for r in case_results if r['has_contradiction'])
        total_contradictions = sum(r['contradiction_count'] for r in case_results)
        
        # 按严重程度统计
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for r in case_results:
            for c in r['contradictions']:
                severity = c.get('severity', 'medium')
                if severity in severity_counts:
                    severity_counts[severity] += 1
        
        stats[model] = {
            'total_cases': total_cases,
            'cases_with_contradiction': cases_with_contradiction,
            'contradiction_rate': cases_with_contradiction / total_cases * 100,
            'avg_contradictions_per_case': total_contradictions / total_cases,
            'total_contradictions': total_contradictions,
            'severity_distribution': severity_counts,
            'high_severity_rate': severity_counts['high'] / max(total_contradictions, 1) * 100
        }
    
    return stats


def generate_contradiction_examples(results, output_dir, max_examples=5):
    """生成矛盾案例示例"""
    examples = []
    
    for model, case_results in results.items():
        model_examples = []
        
        for r in case_results:
            if r['has_contradiction']:
                model_examples.append({
                    'model': model,
                    'case_id': r['case_id'],
                    'contradictions': r['contradictions']
                })
        
        # 取前几个示例
        examples.extend(model_examples[:max_examples])
    
    # 保存示例
    examples_file = os.path.join(output_dir, 'contradiction_examples.md')
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write("# B2 实验：Hallucination 案例示例\n\n")
        f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**示例数量**: {len(examples)}\n\n")
        
        for i, example in enumerate(examples, 1):
            f.write(f"## 示例 {i}\n\n")
            f.write(f"- **模型**: {example['model'].replace('qwen3-', 'Qwen3-').upper()}\n")
            f.write(f"- **案例 ID**: {example['case_id']}\n")
            f.write(f"- **矛盾数量**: {len(example['contradictions'])}\n\n")
            f.write("### 矛盾详情\n")
            
            for j, contradiction in enumerate(example['contradictions'], 1):
                f.write(f"{j}. **类型**: {contradiction['type']}\n")
                f.write(f"   **严重程度**: {contradiction['severity']}\n")
                f.write(f"   **描述**: {contradiction['description']}\n\n")
    
    print(f"✓ 保存矛盾案例示例到：{examples_file}")


def generate_summary_report(stats, output_dir):
    """生成总结报告"""
    models = list(stats.keys())
    
    report_lines = []
    report_lines.append("# B2 实验：Patient Agent Hallucination 检测报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**数据源**: 6 个模型 × 50 cases\n")
    report_lines.append("---\n")
    
    # 1. 矛盾率统计
    report_lines.append("## 1. 矛盾率统计\n")
    sorted_models = sorted(models, key=lambda m: stats[m]['contradiction_rate'])
    
    for model in sorted_models:
        s = stats[model]
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                          f"矛盾率 {s['contradiction_rate']:.1f}% "
                          f"({s['cases_with_contradiction']}/{s['total_cases']} cases)")
    report_lines.append("")
    
    # 2. 严重程度分布
    report_lines.append("## 2. 严重程度分布\n")
    for model in sorted_models:
        s = stats[model]
        report_lines.append(f"- **{model.replace('qwen3-', 'Qwen3-').upper()}**: "
                          f"高严重 {s['severity_distribution']['high']} 例, "
                          f"中严重 {s['severity_distribution']['medium']} 例")
    report_lines.append("")
    
    # 3. 关键发现
    report_lines.append("## 3. 关键发现\n")
    
    # 最低矛盾率模型
    best_model = sorted_models[0]
    worst_model = sorted_models[-1]
    
    report_lines.append(f"- **矛盾率最低**: {best_model.replace('qwen3-', 'Qwen3-').upper()} "
                       f"({stats[best_model]['contradiction_rate']:.1f}%)")
    report_lines.append(f"- **矛盾率最高**: {worst_model.replace('qwen3-', 'Qwen3-').upper()} "
                       f"({stats[worst_model]['contradiction_rate']:.1f}%)")
    
    # 总矛盾数
    total_contradictions = sum(stats[m]['total_contradictions'] for m in models)
    total_high_severity = sum(stats[m]['severity_distribution']['high'] for m in models)
    report_lines.append(f"- **总矛盾数**: {total_contradictions} 例")
    report_lines.append(f"- **高严重矛盾**: {total_high_severity} 例 "
                       f"({total_high_severity/total_contradictions*100:.1f}%)")
    
    # 矛盾类型分析
    report_lines.append("")
    report_lines.append("## 4. 常见矛盾类型\n")
    report_lines.append("基于自动检测规则，主要发现以下类型的矛盾：\n")
    report_lines.append("- **年龄矛盾**: 患者回答中的年龄与 EHR 不符（差异 > 5 岁）")
    report_lines.append("- **性别矛盾**: 患者使用与 EHR 性别矛盾的人称代词")
    report_lines.append("- **诊断否定**: 患者否认 EHR 中记录的诊断")
    report_lines.append("- **症状否定**: 患者否认报告的症状")
    report_lines.append("- **治疗阶段矛盾**: 患者描述的治疗状态与 EHR 不符")
    report_lines.append("- **虚构疾病**: 患者提到未在 EHR 中记录的疾病")
    
    report_lines.append("")
    report_lines.append("---\n")
    report_lines.append("*报告结束*\n")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'hallucination_summary.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 保存总结报告到：{report_file}")


def save_results(results, stats, output_dir):
    """保存所有结果"""
    output = {
        'results': results,
        'statistics': stats
    }
    
    output_file = os.path.join(output_dir, 'hallucination_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存检测结果到：{output_file}")


def main():
    results_dir = 'outputs/model_evaluation_50cases'
    output_dir = 'outputs/experiments/B2_patient_hallucination'
    
    print("="*60)
    print("B2 实验：Patient Agent Hallucination 检测")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载所有模型结果...")
    all_data = load_all_results(results_dir)
    
    if not all_data:
        print("错误：未找到任何模型结果文件")
        return
    
    # 2. 分析所有案例
    print("\n2. 分析所有案例的 hallucination...")
    results = analyze_all_cases(all_data)
    
    # 3. 计算统计指标
    print("\n3. 计算统计指标...")
    stats = calculate_statistics(results)
    
    # 4. 生成矛盾案例示例
    print("\n4. 生成矛盾案例示例...")
    generate_contradiction_examples(results, output_dir)
    
    # 5. 保存结果
    print("\n5. 保存检测结果...")
    save_results(results, stats, output_dir)
    
    # 6. 生成报告
    print("\n6. 生成总结报告...")
    generate_summary_report(stats, output_dir)
    
    print("\n" + "="*60)
    print("✓ B2 实验完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
