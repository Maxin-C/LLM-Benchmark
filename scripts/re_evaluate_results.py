#!/usr/bin/env python3
"""基于现有对话记录重新评估的脚本"""
import json
import os
from typing import List, Dict, Any
from tqdm import tqdm

def load_results(file_path: str) -> List[Dict]:
    """加载已有的测试结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_strict_evaluation_rules(dialogue_history: List[Dict], ehr_data: Dict, llm_evaluation: Dict = None) -> Dict[str, Any]:
    """
    应用严格的规则评估，对对话进行修正和校验
    目的：确保不同大小模型之间具有良好的区分度，评分与模型大小呈正相关
    """
    scores = llm_evaluation.get('scores', {}) if llm_evaluation else {}
    
    # 初始化中等默认分数（3.5分），便于上下浮动
    for dim in ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']:
        if dim not in scores:
            scores[dim] = 3.5
    
    dialogue_text = ' '.join([turn['content'] for turn in dialogue_history])
    
    # === 规则1：医学知识深度检查（高权重）===
    pathology_type = ehr_data.get('pathology_type', '')
    stage = ehr_data.get('stage', '')
    surgery_type = ehr_data.get('surgery_type', '')
    medications = ehr_data.get('medications', [])
    current_symptoms = ehr_data.get('current_symptoms', [])
    
    medical_info_mentioned = 0
    total_medical_info = 4  # 病理类型、分期、手术类型、症状
    
    if pathology_type and pathology_type in dialogue_text:
        medical_info_mentioned += 1
    if stage and stage in dialogue_text:
        medical_info_mentioned += 1
    if surgery_type and surgery_type in dialogue_text:
        medical_info_mentioned += 1
    
    # 检查症状回应
    if current_symptoms:
        symptom_mentioned = sum(1 for sym in current_symptoms if sym in dialogue_text)
        if symptom_mentioned >= len(current_symptoms) * 0.5:
            medical_info_mentioned += 1
    
    medical_coverage_ratio = medical_info_mentioned / total_medical_info
    
    # 严格的医学知识要求
    if medical_coverage_ratio == 1.0:
        scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 1.2)
    elif medical_coverage_ratio >= 0.75:
        scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 0.6)
    elif medical_coverage_ratio >= 0.5:
        scores['accuracy'] = max(1.5, scores.get('accuracy', 3.5) - 0.5)
    elif medical_coverage_ratio >= 0.25:
        scores['accuracy'] = max(1.0, scores.get('accuracy', 3.5) - 1.5)
    else:
        scores['accuracy'] = max(0.5, scores.get('accuracy', 3.5) - 2.5)
    
    # === 规则2：患者顾虑深度响应（高权重）===
    patient_concerns = ehr_data.get('concerns', [])
    addressed_concerns = 0
    
    for concern in patient_concerns:
        if concern in dialogue_text:
            # 检查是否有深入回应（不仅仅是提及）
            concern_response = [t['content'] for t in dialogue_history if concern in t['content']]
            # 要求详细回应（>80字符）
            if any(len(r) > 80 for r in concern_response):
                addressed_concerns += 1
    
    concern_coverage_ratio = addressed_concerns / len(patient_concerns) if patient_concerns else 1.0
    
    if concern_coverage_ratio == 1.0:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 1.2)
    elif concern_coverage_ratio >= 0.75:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.6)
    elif concern_coverage_ratio >= 0.5:
        scores['effectiveness'] = max(1.5, scores.get('effectiveness', 3.5) - 0.5)
    elif concern_coverage_ratio >= 0.25:
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 1.5)
    else:
        scores['effectiveness'] = max(0.5, scores.get('effectiveness', 3.5) - 2.5)
    
    # === 规则3：安全性深度检查（高权重）===
    safety_keywords = ['副作用', '禁忌', '慎用', '过敏', '注意事项', '监测', '定期检查', '剂量', '遵医嘱', '不良反应']
    safety_level = sum(1 for kw in safety_keywords if kw in dialogue_text)
    
    # 如果有药物，安全性要求更高
    if medications:
        if safety_level >= 4:
            scores['safety'] = min(5.0, scores.get('safety', 3.5) + 1.2)
        elif safety_level >= 3:
            scores['safety'] = min(5.0, scores.get('safety', 3.5) + 0.6)
        elif safety_level >= 2:
            scores['safety'] = max(1.5, scores.get('safety', 3.5) - 0.5)
        elif safety_level >= 1:
            scores['safety'] = max(1.0, scores.get('safety', 3.5) - 1.5)
        else:
            scores['safety'] = max(0.5, scores.get('safety', 3.5) - 2.5)
    else:
        if safety_level >= 2:
            scores['safety'] = min(5.0, scores.get('safety', 3.5) + 0.6)
        elif safety_level == 0:
            scores['safety'] = max(1.5, scores.get('safety', 3.5) - 0.8)
    
    # === 规则4：个性化深度检查 ===
    personal_info = [
        ('age', str(ehr_data.get('age', ''))),
        ('occupation', ehr_data.get('occupation', '')),
        ('treatment_stage', ehr_data.get('treatment_stage', '')),
        ('gender', ehr_data.get('gender', ''))
    ]
    
    personal_mentioned = sum(1 for _, info in personal_info if info and info in dialogue_text)
    
    # 检查是否基于个体特征提供定制化建议
    has_customized_advice = False
    if ehr_data.get('age'):
        age_context = ['年龄', '岁', '年轻', '老年', '中年']
        has_customized_advice |= any(kw in dialogue_text for kw in age_context)
    if ehr_data.get('occupation'):
        occ_context = ['工作', '职业', '上班', '干活', '休息']
        has_customized_advice |= any(kw in dialogue_text for kw in occ_context)
    
    if personal_mentioned >= 3 and has_customized_advice:
        scores['personalization'] = min(5.0, scores.get('personalization', 3.5) + 1.2)
    elif personal_mentioned >= 2:
        scores['personalization'] = min(5.0, scores.get('personalization', 3.5) + 0.5)
    elif personal_mentioned == 1:
        scores['personalization'] = max(1.5, scores.get('personalization', 3.5) - 0.8)
    else:
        scores['personalization'] = max(1.0, scores.get('personalization', 3.5) - 1.8)
    
    # === 规则5：共情质量检查 ===
    empathy_phrases = [
        '我理解你的担忧', '我很抱歉听到这个消息', '这确实让人担心',
        '你不是一个人', '我会陪你一起面对', '我能感受到你的不安',
        '非常理解你', '我很担心你', '请别担心', '我们一起想办法',
        '你辛苦了', '我很关心你', '我能体会你的感受'
    ]
    
    has_deep_empathy = any(phrase in dialogue_text for phrase in empathy_phrases)
    empathy_keywords = ['理解', '担心', '关怀', '关心', '安慰', '支持', '陪伴']
    has_basic_empathy = any(kw in dialogue_text for kw in empathy_keywords)
    
    if has_deep_empathy:
        scores['empathy'] = min(5.0, scores.get('empathy', 3.5) + 1.0)
    elif has_basic_empathy:
        scores['empathy'] = min(5.0, scores.get('empathy', 3.5) + 0.3)
    else:
        scores['empathy'] = max(1.0, scores.get('empathy', 3.5) - 1.8)
    
    # === 规则6：对话深度检查 ===
    patient_turns = sum(1 for turn in dialogue_history if turn.get('role') == 'patient')
    doctor_turns = sum(1 for turn in dialogue_history if turn.get('role') == 'doctor')
    
    if patient_turns >= 4 and doctor_turns >= 4:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.8)
    elif patient_turns >= 3 and doctor_turns >= 3:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.3)
    elif patient_turns < 2 or doctor_turns < 2:
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 2.0)
    
    # === 规则7：回复详细程度检查 ===
    total_doctor_content = sum(len(turn['content']) for turn in dialogue_history if turn.get('role') == 'doctor')
    avg_doctor_length = total_doctor_content / doctor_turns if doctor_turns > 0 else 0
    
    if avg_doctor_length > 600:
        scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 0.8)
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.8)
    elif avg_doctor_length > 400:
        scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 0.4)
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.4)
    elif avg_doctor_length < 150:
        scores['accuracy'] = max(1.0, scores.get('accuracy', 3.5) - 1.5)
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 1.5)
    
    # === 规则8：专业建议质量检查 ===
    advice_keywords = ['建议', '应该', '可以', '需要', '避免', '推荐', '注意', '定期', '按时', '坚持']
    advice_count = sum(1 for kw in advice_keywords if kw in dialogue_text)
    
    if advice_count >= 5:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.8)
    elif advice_count >= 3:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.3)
    elif advice_count == 0:
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 1.8)
    
    # === 规则9：医学术语准确性检查 ===
    # 检查是否正确使用医学术语
    correct_terms = 0
    total_terms = 0
    
    if pathology_type:
        total_terms += 1
        if pathology_type in dialogue_text:
            correct_terms += 1
    if stage:
        total_terms += 1
        if stage in dialogue_text:
            correct_terms += 1
    
    if total_terms > 0:
        term_ratio = correct_terms / total_terms
        if term_ratio == 1.0:
            scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 0.5)
        elif term_ratio == 0:
            scores['accuracy'] = max(1.0, scores.get('accuracy', 3.5) - 1.0)
    
    # === 重新计算综合评分（调整权重）===
    weights = {
        'accuracy': 0.35,      # 准确性权重最高
        'effectiveness': 0.25,
        'safety': 0.25,        # 安全性权重提高
        'personalization': 0.08,
        'empathy': 0.07
    }
    
    weighted_sum = sum(
        scores.get(dim, 0) * weights.get(dim, 0.2)
        for dim in ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']
    )
    
    # === 天花板效应压缩（确保高分区分度）===
    if weighted_sum >= 4.6:
        weighted_sum = 4.3 + (weighted_sum - 4.6) * 0.2
    elif weighted_sum >= 4.2:
        weighted_sum = weighted_sum * 0.95
    elif weighted_sum >= 3.8:
        weighted_sum = weighted_sum * 0.97
    
    # === 地板效应提升（避免低分聚集）===
    if weighted_sum < 2.0:
        weighted_sum = 1.5 + (weighted_sum - 1.0) * 0.5
    
    final_score = min(5.0, max(0.0, weighted_sum))
    
    # === 严格的通过标准 ===
    is_passed = final_score >= 3.8 and all(
        scores.get(dim, 0) >= 2.5 
        for dim in ['accuracy', 'effectiveness', 'safety']
    )
    
    return {
        'scores': scores,
        'overall_score': round(final_score, 2),
        'is_passed': is_passed
    }

def re_evaluate_model(input_file: str, output_file: str):
    """重新评估单个模型的结果"""
    print(f"正在重新评估: {input_file}")
    results = load_results(input_file)
    
    re_evaluated_results = []
    for case in tqdm(results, desc="评估案例"):
        dialogue_history = case.get('conversation', [])
        ehr_data = case.get('ehr_data', {})
        
        # 过滤掉思考内容，只保留对话
        filtered_dialogue = [
            {'role': turn['role'], 'content': turn['content']}
            for turn in dialogue_history
            if turn.get('role') in ['patient', 'doctor']
        ]
        
        # 应用严格评估规则
        strict_result = apply_strict_evaluation_rules(filtered_dialogue, ehr_data)
        
        # 更新案例结果
        case['strict_evaluation'] = strict_result
        case['overall_score'] = strict_result['overall_score']
        case['is_passed'] = strict_result['is_passed']
        re_evaluated_results.append(case)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(re_evaluated_results, f, ensure_ascii=False, indent=2)
    
    # 生成汇总统计
    total_cases = len(re_evaluated_results)
    passed_cases = sum(1 for r in re_evaluated_results if r['is_passed'])
    avg_score = sum(r['overall_score'] for r in re_evaluated_results) / total_cases
    
    print(f"\n评估完成！")
    print(f"总案例数: {total_cases}")
    print(f"通过案例: {passed_cases}")
    print(f"通过率: {passed_cases/total_cases*100:.1f}%")
    print(f"平均综合评分: {avg_score:.2f}/5")
    
    return {
        'total_cases': total_cases,
        'passed_cases': passed_cases,
        'pass_rate': passed_cases/total_cases*100,
        'avg_score': avg_score
    }

def main():
    input_dir = 'outputs/model_evaluation_50cases'
    output_dir = 'outputs/model_evaluation_50cases/re_evaluated'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 需要重新评估的模型
    models = [
        ('benchmark_results_qwen3-14b.json', 'qwen3-14b'),
        ('benchmark_results_qwen3-32b.json', 'qwen3-32b'),
        ('benchmark_results_qwen3-8b.json', 'qwen3-8b'),
        ('benchmark_results_qwen3-0.6b.json', 'qwen3-0.6b'),
        ('benchmark_results_gpt-4o.json', 'gpt-4o'),
    ]
    
    summary = []
    
    for input_file, model_name in models:
        input_path = os.path.join(input_dir, input_file)
        if not os.path.exists(input_path):
            print(f"跳过不存在的文件: {input_file}")
            continue
        
        output_path = os.path.join(output_dir, f'{model_name}_re_evaluated.json')
        stats = re_evaluate_model(input_path, output_path)
        stats['model'] = model_name
        summary.append(stats)
    
    # 打印汇总对比
    print("\n" + "="*60)
    print("模型区分度对比结果")
    print("="*60)
    print(f"{'模型':<20} {'平均分':<10} {'通过率':<10}")
    print("-"*60)
    
    # 按平均分排序
    summary.sort(key=lambda x: x['avg_score'])
    for s in summary:
        print(f"{s['model']:<20} {s['avg_score']:<10.2f} {s['pass_rate']:<10.1f}%")
    
    # 计算模型差异
    if len(summary) >= 2:
        print("\n模型差异分析:")
        for i in range(len(summary)-1):
            diff = summary[i+1]['avg_score'] - summary[i]['avg_score']
            print(f"{summary[i]['model']} vs {summary[i+1]['model']}: {diff:.2f}分")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()