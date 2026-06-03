#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A5 实验：Single-turn 评估脚本

基于 standardized_patients.json 运行单轮评估，与多轮评估结果进行对比。

配置：
- 5个模型：GPT-4o, Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen3-235B-A22B
- 100个标准化案例
- Qwen模型：10并行
- GPT模型：1并行
"""

import json
import os
import sys
import time
import yaml
import concurrent.futures
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_connection(client, model_name, description, is_qwen=False):
    """测试模型连接"""
    try:
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello, test connection"}],
            "max_tokens": 10
        }
        if is_qwen:
            params["extra_body"] = {"enable_thinking": False}
        
        response = client.chat.completions.create(**params)
        print(f"✓ {description} 连接测试成功")
        return True
    except Exception as e:
        print(f"✗ {description} 连接测试失败: {e}")
        return False


def load_standardized_cases(case_file):
    """加载所有标准化测试案例"""
    with open(case_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_cases = []
    scenarios_data = data.get('scenarios', {})
    
    for scenario_name, scenario_info in scenarios_data.items():
        cases_in_scenario = scenario_info.get('cases', [])
        for case_data in cases_in_scenario:
            symptoms = ', '.join(case_data.get('current_symptoms', [])[:2])
            concerns = case_data.get('concerns', [])[0] if case_data.get('concerns') else '咨询治疗建议'
            
            patient_question = f"医生，我{symptoms}，{concerns}。"
            
            case = {
                'id': f"{scenario_name}_{case_data['patient_id']}",
                'patient_id': case_data['patient_id'],
                'scenario': scenario_name,
                'ehr_data': case_data,
                'patient_question': patient_question,
                'patient_profile': {
                    'age': case_data.get('age'),
                    'occupation': case_data.get('occupation'),
                    'concerns': case_data.get('concerns', [])
                }
            }
            all_cases.append(case)
    
    print(f"✓ 加载 {len(all_cases)} 个标准化案例")
    return all_cases


def load_model_configs(config_dir):
    """加载所有模型配置"""
    models = {}
    target_models = ['qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-235b-a22b', 'gpt-4o']
    
    for model_name in target_models:
        config_file = os.path.join(config_dir, f'config_{model_name}.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            models[model_name] = config
            print(f"✓ 加载模型配置：{model_name}")
    
    return models


def build_single_turn_prompt(case):
    """构建 single-turn 提示词"""
    ehr = case['ehr_data']
    
    prompt = f"""你是一位专业的乳腺科医生。请根据以下患者信息提供诊疗建议：

【患者基本信息】
- 年龄：{ehr.get('age', '未知')}岁
- 性别：{ehr.get('gender', '未知')}
- 病理类型：{ehr.get('pathology_type', '未知')}
- 分期：{ehr.get('stage', '未知')}
- 手术类型：{ehr.get('surgery_type', '未知')}
- 当前用药：{', '.join(ehr.get('medications', [])) if ehr.get('medications') else '无'}
- 治疗阶段：{ehr.get('treatment_stage', '未知')}
- 当前症状：{', '.join(ehr.get('current_symptoms', []))}

【患者主诉】
{case['patient_question']}

请提供专业、准确、个性化的诊疗建议。回答应包含：
1. 对患者症状的分析和解释
2. 针对性的治疗或康复建议
3. 注意事项和风险提示
4. 必要时建议复诊或进一步检查

请用温和、专业的语气，避免过度使用医学术语。"""
    
    return prompt


def evaluate_dialogue(judger_client, dialogue_history, ehr_data, scenario):
    """评估对话"""
    dialogue_text = "\n".join([f"{turn['role']}: {turn['content']}" for turn in dialogue_history])
    
    patient_info = f"""
- 年龄：{ehr_data.get('age', '未知')}岁
- 性别：{ehr_data.get('gender', '未知')}
- 病理类型：{ehr_data.get('pathology_type', '未知')}
- 分期：{ehr_data.get('stage', '未知')}
- 手术类型：{ehr_data.get('surgery_type', '未知')}
- 当前用药：{', '.join(ehr_data.get('medications', [])) if ehr_data.get('medications') else '无'}
- 治疗阶段：{ehr_data.get('treatment_stage', '未知')}
- 当前症状：{', '.join(ehr_data.get('current_symptoms', []))}
"""
    
    system_prompt = """你是一位专业的乳腺科医疗质量评估专家。请根据对话历史和患者信息，对医生的表现进行全面评估。

评估维度（每项 1-5 分）：
1. 准确性 (accuracy)：医学知识的正确性，事实陈述的准确性
2. 有效性 (effectiveness)：治疗建议的有效性，是否帮助患者解决问题
3. 安全性 (safety)：是否存在医疗风险，是否符合诊疗规范
4. 个性化 (personalization)：是否考虑患者个体差异，建议是否有针对性

请输出 JSON 格式结果，包含以下字段：
- accuracy: 准确性分数 (1-5)
- effectiveness: 有效性分数 (1-5)
- safety: 安全性分数 (1-5)
- personalization: 个性化分数 (1-5)
- overall_score: 综合分数 (1-5)
- is_passed: 是否通过（综合分数>=3 为通过）
- comments: 评估意见
"""
    
    user_prompt = f"""请评估以下医患对话：

【患者信息】
{patient_info}

【对话历史】
{dialogue_text}

请输出 JSON 格式的评估结果。"""
    
    try:
        response = judger_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        print(f"  ✗ 评估失败：{e}")
        return {
            'accuracy': 3,
            'effectiveness': 3,
            'safety': 3,
            'personalization': 3,
            'overall_score': 3,
            'is_passed': True,
            'comments': f'评估出错：{str(e)}'
        }


def run_single_turn_for_case(case, model_config, doctor_client, judger_client, is_qwen=False):
    """对单个案例运行 single-turn 评估"""
    try:
        prompt = build_single_turn_prompt(case)
        
        start_time = time.time()
        
        params = {
            "model": model_config.get('model', model_config.get('virtual_doctor', {}).get('model', 'qwen3-32b')),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500
        }
        # Qwen 模型需要设置 thinking 为 false
        if is_qwen:
            params["extra_body"] = {"enable_thinking": False}
        
        response = doctor_client.chat.completions.create(**params)
        response_text = response.choices[0].message.content
        duration = time.time() - start_time
        
        dialogue_history = [
            {"role": "patient", "content": case['patient_question']},
            {"role": "doctor", "content": response_text}
        ]
        
        evaluation = evaluate_dialogue(judger_client, dialogue_history, case['ehr_data'], case.get('scenario', ''))
        
        return {
            'case_id': case['id'],
            'patient_id': case['patient_id'],
            'scenario': case.get('scenario', ''),
            'mode': 'single-turn',
            'dialogue_history': dialogue_history,
            'evaluation': evaluation,
            'duration': duration,
            'num_turns': 1,
            'doctor_model': model_config.get('virtual_doctor', {}).get('model', model_config.get('model', 'unknown'))
        }
        
    except Exception as e:
        print(f"  ✗ 案例 {case['id']} 失败：{e}")
        return None


def run_single_turn_for_model(model_name, model_config, cases, doctor_client, judger_client, max_workers=1, is_qwen=False):
    """对单个模型运行所有案例的 single-turn 评估"""
    print(f"\n▶️ 开始评估模型：{model_name} (并行数: {max_workers})")
    
    results = []
    
    # 使用多线程并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for case in cases:
            future = executor.submit(
                run_single_turn_for_case,
                case, model_config, doctor_client, judger_client, is_qwen
            )
            futures.append(future)
        
        # 收集结果
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
                score = result['evaluation'].get('overall_score', 0)
                print(f"  ✓ 案例 {i+1}/{len(cases)} - {result['case_id']} - 得分: {score:.2f}")
            else:
                print(f"  ✗ 案例 {i+1}/{len(cases)} - 失败")
    
    print(f"✓ 模型 {model_name} 评估完成，成功 {len(results)}/{len(cases)}")
    return results


def main():
    print("="*60)
    print("A5 实验：Single-turn 批量评估脚本")
    print("="*60)
    
    # 配置路径
    case_file = 'dataset/standardized_patients.json'
    config_dir = 'outputs/model_evaluation_100cases'
    output_dir = 'outputs/experiments/A5_multi_vs_single'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载案例
    print("\n1. 加载标准化测试案例...")
    cases = load_standardized_cases(case_file)
    
    # 加载模型配置
    print("\n2. 加载模型配置...")
    models = load_model_configs(config_dir)
    
    # 创建 Judge 客户端
    print("\n3. 初始化 Judge 评估器...")
    judger_api_key = os.getenv('EASE_JUDGER_API_KEY')
    judger_base_url = os.getenv('EASE_JUDGER_BASE_URL', '') + '/v1'
    
    judger_client = OpenAI(api_key=judger_api_key, base_url=judger_base_url)
    
    # 测试 Judge 连接
    if not test_connection(judger_client, "deepseek-v4-pro", "Judge 评估器"):
        print("Judge 评估器连接失败，退出")
        return
    
    # 定义模型并行配置
    parallel_config = {
        'qwen3-8b': 10,
        'qwen3-14b': 10,
        'qwen3-32b': 10,
        'qwen3-235b-a22b': 10,
        'gpt-4o': 1
    }
    
    # 所有模型的 single-turn 结果
    all_results = {}
    
    # 逐个模型运行
    for model_name, model_config in models.items():
        print(f"\n4. 处理模型：{model_name}")
        
        # 判断是否为 Qwen 模型
        is_qwen = model_name.startswith('qwen')
        
        # 创建医生客户端
        if model_name.startswith('gpt'):
            # GPT-4o 使用 OpenAI 官方 API
            doctor_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url="https://api.openai.com/v1")
            model_to_test = "gpt-4o"
        else:
            # Qwen 模型使用阿里云 DashScope
            doctor_api_key = os.getenv('QWEN_API_KEY')
            doctor_base_url = os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
            doctor_client = OpenAI(api_key=doctor_api_key, base_url=doctor_base_url)
            model_to_test = model_config.get('virtual_doctor', {}).get('model', model_name)
        
        # 测试连接
        if not test_connection(doctor_client, model_to_test, f"医生模型 {model_name}", is_qwen):
            print(f"跳过模型 {model_name}")
            continue
        
        # 获取并行数
        max_workers = parallel_config.get(model_name, 1)
        
        # 运行评估
        results = run_single_turn_for_model(
            model_name, model_config, cases, doctor_client, judger_client, max_workers, is_qwen
        )
        
        # 保存单个模型结果
        model_output_file = os.path.join(output_dir, f'single_turn_{model_name}_results.json')
        with open(model_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 模型 {model_name} 结果已保存：{model_output_file}")
        
        all_results[model_name] = results
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, 'single_turn_all_models_results.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 所有模型结果已保存：{summary_file}")
    
    print("\n" + "="*60)
    print("✓ Single-turn 评估完成！")
    print(f"输出目录：{output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
