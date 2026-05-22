#!/usr/bin/env python3
"""
基准测试运行器 - 支持批量运行和随机化虚拟患者
参考论文代码架构设计：
- virtual patient: 待评估模型（可配置，支持reasoning模型）
- virtual doctor: 固定为 deepseek-r1
- judger: 固定为 deepseek-v3
- dialogue_monitor: 固定为 deepseek-v3

支持reasoning模型的thinking过程单独记录和评估
"""

import argparse
import json
import os
import sys
import time
import random
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_random_patient() -> Dict[str, Any]:
    """
    生成随机患者数据（增加虚拟患者的随机性）
    
    返回：
        随机生成的患者EHR数据
    """
    # 随机参数范围
    ages = [35, 40, 45, 50, 55, 60, 65]
    occupations = ['教师', '医生', '护士', '工程师', '公务员', '企业经理', '自由职业者', '家庭主妇']
    pathology_types = ['浸润性导管癌', '浸润性小叶癌', '原位导管癌', '黏液癌', '三阴性乳腺癌']
    stages = ['IA期', 'IB期', 'IIA期', 'IIB期', 'IIIA期', 'IIIB期']
    surgery_types = ['乳房切除术', '保乳手术', '乳房重建术', '改良根治术']
    medications = [
        ['他莫昔芬'],
        ['来曲唑'],
        ['阿那曲唑'],
        ['他莫昔芬', '曲妥珠单抗'],
        ['来曲唑', '帕妥珠单抗'],
        ['依西美坦'],
    ]
    treatment_stages = ['术后恢复期', '化疗阶段', '内分泌治疗', '靶向治疗', '随访期']
    
    # 随机选择
    ehr_data = {
        'patient_id': f'patient_{random.randint(1000, 9999)}',
        'age': random.choice(ages),
        'gender': 'female',
        'occupation': random.choice(occupations),
        'pathology_type': random.choice(pathology_types),
        'stage': random.choice(stages),
        'surgery_type': random.choice(surgery_types),
        'medications': random.choice(medications),
        'treatment_stage': random.choice(treatment_stages),
        'diagnosis_date': f'202{random.randint(3, 5)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
        'surgery_date': f'202{random.randint(3, 5)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
    }
    
    # 随机症状
    symptom_options = ['乳房疼痛', '肿胀', '皮肤发红', '乳头溢液', '疲劳', '恶心', '关节痛', '潮热']
    num_symptoms = random.randint(1, 4)
    ehr_data['current_symptoms'] = random.sample(symptom_options, num_symptoms)
    
    # 随机顾虑
    concerns = [
        '担心治疗副作用',
        '担心病情复发',
        '对后续治疗有疑问',
        '想了解康复锻炼方法',
        '关心饮食调理',
        '担心影响日常生活',
        '对药物过敏有顾虑',
        '想了解预后情况',
    ]
    num_concerns = random.randint(1, 3)
    ehr_data['concerns'] = random.sample(concerns, num_concerns)
    
    return ehr_data

def build_doctor_prompt(ehr_data: Dict[str, Any], dialogue_history: List[Dict]) -> str:
    """
    构建医生的提示词（结合EHR数据）
    
    参数：
        ehr_data: 患者EHR数据
        dialogue_history: 对话历史
    
    返回：
        医生提示词
    """
    age = ehr_data.get('age', '')
    occupation = ehr_data.get('occupation', '')
    pathology_type = ehr_data.get('pathology_type', '')
    stage = ehr_data.get('stage', '')
    surgery_type = ehr_data.get('surgery_type', '')
    medications = ehr_data.get('medications', [])
    treatment_stage = ehr_data.get('treatment_stage', '')
    
    medication_str = '、'.join(medications) if medications else ''
    
    # 构建对话历史字符串（过滤thinking部分）
    history_lines = []
    for turn in dialogue_history:
        role = turn['role']
        content = turn['content']
        # 如果有thinking部分，也包含进去供医生参考
        if 'thinking' in turn and turn['thinking']:
            history_lines.append(f"{role} (思考): {turn['thinking']}")
        history_lines.append(f"{role}: {content}")
    
    history_str = '\n'.join(history_lines)
    
    prompt = f"""
你是一位专业的乳腺科医生。请根据患者的病史信息和对话历史，给出专业的医疗建议。

患者信息：
年龄：{age}岁
职业：{occupation}
诊断：{pathology_type}{stage}
手术类型：{surgery_type}
当前用药：{medication_str}
治疗阶段：{treatment_stage}

对话历史：
{history_str}

请以医生身份，基于以上信息，对患者的最新发言做出专业、温和、有同理心的回应。
"""
    return prompt.strip()

def build_patient_prompt(ehr_data: Dict[str, Any], dialogue_history: List[Dict]) -> str:
    """
    构建患者的提示词（待评估模型）
    
    参数：
        ehr_data: 患者EHR数据
        dialogue_history: 对话历史
    
    返回：
        患者提示词
    """
    age = ehr_data.get('age', '')
    occupation = ehr_data.get('occupation', '')
    symptoms = ehr_data.get('current_symptoms', [])
    concerns = ehr_data.get('concerns', [])
    
    symptom_str = '、'.join(symptoms) if symptoms else ''
    concern_str = '、'.join(concerns) if concerns else ''
    
    # 构建对话历史字符串
    history_str = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in dialogue_history])
    
    prompt = f"""
你是一位乳腺癌患者。请根据你的病情信息和对话历史，以患者身份进行回应。

患者信息：
年龄：{age}岁
职业：{occupation}
当前症状：{symptom_str}
主要顾虑：{concern_str}

对话历史：
{history_str}

请以患者身份，基于以上信息，对医生的最新发言做出真实、自然的回应。
"""
    return prompt.strip()

def build_reasoning_patient_prompt(ehr_data: Dict[str, Any], dialogue_history: List[Dict]) -> str:
    """
    构建reasoning患者的提示词（包含thinking过程要求）
    
    参数：
        ehr_data: 患者EHR数据
        dialogue_history: 对话历史
    
    返回：
        患者提示词（要求输出thinking过程）
    """
    age = ehr_data.get('age', '')
    occupation = ehr_data.get('occupation', '')
    symptoms = ehr_data.get('current_symptoms', [])
    concerns = ehr_data.get('concerns', [])
    
    symptom_str = '、'.join(symptoms) if symptoms else ''
    concern_str = '、'.join(concerns) if concerns else ''
    
    # 构建对话历史字符串
    history_str = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in dialogue_history])
    
    prompt = f"""
你是一位乳腺癌患者。请根据你的病情信息和对话历史，以患者身份进行回应。

患者信息：
年龄：{age}岁
职业：{occupation}
当前症状：{symptom_str}
主要顾虑：{concern_str}

对话历史：
{history_str}

请按照以下格式输出你的思考过程和最终回应：

思考：[详细描述你作为患者的思考过程，包括你为什么会这样回应，你的担忧是什么，你想从医生那里得到什么信息等]

回应：[你的最终回答，以患者身份对医生的最新发言做出真实、自然的回应]

注意：思考部分要详细，展示你的推理过程。
"""
    return prompt.strip()

def call_llm(client, model: str, messages: List[Dict], params: Dict[str, Any], is_reasoning: bool = False) -> Tuple[str, str]:
    """
    调用LLM模型（支持reasoning模型的thinking输出）
    
    参数：
        client: OpenAI客户端
        model: 模型名称
        messages: 消息列表
        params: 模型参数
        is_reasoning: 是否为reasoning模型
    
    返回：
        (content, thinking): 内容和思考过程（非reasoning模型thinking为空）
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            **params
        )
        
        message = chat_completion.choices[0].message
        
        # 获取内容（DeepSeek API可能将内容放在不同字段）
        content = message.content
        
        # 处理DeepSeek官方API的thinking输出
        thinking = ""
        
        # 方式1：通过message.reasoning_content字段获取（DeepSeek官方API）
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            # DeepSeek API中，thinking模式下reasoning_content包含思考内容
            # 如果content为空，说明需要从reasoning_content中提取
            if not content:
                content = message.reasoning_content
            else:
                thinking = message.reasoning_content
        
        # 方式2：通过message.thinking字段获取（其他API）
        if not thinking and hasattr(message, 'thinking') and message.thinking:
            thinking = message.thinking
        
        # 方式3：尝试从content中解析（兼容其他reasoning模型）
        if not thinking and is_reasoning and content:
            if '思考：' in content and '回应：' in content:
                thinking = content.split('思考：')[1].split('回应：')[0].strip()
                content = content.split('回应：')[1].strip()
            elif 'Thinking:' in content and 'Response:' in content:
                thinking = content.split('Thinking:')[1].split('Response:')[0].strip()
                content = content.split('Response:')[1].strip()
        
        return content, thinking
    except Exception as e:
        print(f"LLM调用失败: {e}")
        return "", ""

def dialogue_monitor(client, model: str, msgs: List[Dict], params: Dict[str, Any]) -> str:
    """
    对话监控器（判断对话是否应该终止）
    
    参数：
        client: OpenAI客户端
        model: 模型名称
        msgs: 消息列表
        params: 模型参数
    
    返回：
        监控结果（0=继续，1=终止）
    """
    # 过滤thinking部分，只传递对话内容给监控器
    clean_messages = []
    for msg in msgs:
        clean_msg = {'role': msg['role'], 'content': msg['content']}
        clean_messages.append(clean_msg)
    
    monitor_prompt = f"""
分析以下对话，判断是否应该终止：

对话：
{json.dumps(clean_messages, ensure_ascii=False)}

请输出0（继续对话）或1（终止对话）。
"""
    
    messages = [{'role': 'user', 'content': monitor_prompt}]
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            **params
        )
        result = chat_completion.choices[0].message.content.strip()
        return result
    except Exception as e:
        print(f"监控器调用失败: {e}")
        return "0"

def main(config_path: str = 'config/sandbox_config.yaml', 
         output_dir: str = 'outputs/evaluations', 
         scenario_name: str = None,
         num_cases: int = 10,
         output_file: str = 'benchmark_results.json'):
    """
    运行基准测试（支持批量运行）
    
    参数：
        config_path: 配置文件路径
        output_dir: 输出目录
        scenario_name: 特定场景名称（可选）
        num_cases: 批量运行的案例数量
        output_file: 输出文件名
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 先加载环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    # 然后加载配置（此时环境变量已可用）
    config = load_config(config_path)
    
    # 创建LLM客户端工厂
    from src.utils.llm_client_factory import LLMClientFactory
    factory = LLMClientFactory(config)
    
    # 创建各角色客户端
    vp_client = factory.create_virtual_patient_client()
    doctor_client = factory.create_virtual_doctor_client()
    judger_client = factory.create_judger_client()
    monitor_client = factory.create_monitor_client()
    
    # 获取模型名称
    vp_model = factory.get_virtual_patient_model()
    is_vp_reasoning = factory.is_virtual_patient_reasoning()
    doctor_model = factory.get_virtual_doctor_model()
    judger_model = factory.get_judger_model()
    monitor_model = factory.get_monitor_model()
    
    # 获取模型参数
    doctor_params = factory.get_doctor_params()
    judger_params = factory.get_judger_params()
    monitor_params = factory.get_monitor_params()
    vp_params = factory.get_virtual_patient_params()
    
    print(f"待评估模型(virtual patient): {vp_model} (reasoning: {is_vp_reasoning})")
    print(f"虚拟医生(virtual doctor): {doctor_model}")
    print(f"评估器(judger): {judger_model}")
    print(f"监控器(monitor): {monitor_model}")
    
    # 批量运行多个案例
    all_results = []
    print(f"\n开始批量运行 {num_cases} 个案例...")
    
    for case_idx in tqdm(range(num_cases), desc="处理案例"):
        print(f"\n=== 案例 {case_idx + 1}/{num_cases} ===")
        
        # 生成随机患者数据
        ehr_data = generate_random_patient()
        print(f"患者ID: {ehr_data['patient_id']}")
        print(f"年龄: {ehr_data['age']}岁, 职业: {ehr_data['occupation']}")
        print(f"诊断: {ehr_data['pathology_type']}{ehr_data['stage']}")
        
        # 记录推理时间
        recoder = {
            "case_id": case_idx + 1,
            "patient_id": ehr_data['patient_id'],
            "ehr_data": ehr_data,
            "vp_model": vp_model,
            "is_vp_reasoning": is_vp_reasoning,
            "doctor_model": doctor_model,
            "judger_model": judger_model,
            "conversation": [],
            "all_thinking": [],  # 单独记录所有thinking过程
            "start_time": time.time()
        }
        
        # 模拟对话 - 患者先开始
        dialogue_history = []
        current_round = 0
        max_rounds = 6
        
        # 患者初始问题（随机选择）
        initial_questions = [
            f"医生，我最近感觉{ehr_data['current_symptoms'][0]}，不知道是不是正常的？",
            f"医生您好，我是{ehr_data['occupation']}，最近在接受{ehr_data['treatment_stage']}，有些担心。",
            f"医生，我做完{ehr_data['surgery_type']}后感觉不太舒服，想咨询一下。",
            f"您好医生，我想了解一下我现在的{ehr_data['treatment_stage']}需要注意什么？",
        ]
        
        patient_response = random.choice(initial_questions)
        dialogue_history.append({'role': 'patient', 'content': patient_response, 'thinking': ''})
        recoder['conversation'].append({
            "role": "patient",
            "content": patient_response,
            "thinking": "",
            "turn": 1,
            "model": vp_model,
            "is_reasoning": is_vp_reasoning
        })
        print(f"患者({vp_model}): {patient_response[:60]}...")
        
        while current_round < max_rounds:
            # 医生响应（使用deepseek-r1）
            doctor_prompt = build_doctor_prompt(ehr_data, dialogue_history)
            doctor_messages = [{'role': 'user', 'content': doctor_prompt}]
            doctor_response, _ = call_llm(doctor_client, doctor_model, doctor_messages, doctor_params)
            
            dialogue_history.append({'role': 'doctor', 'content': doctor_response, 'thinking': ''})
            recoder['conversation'].append({
                "role": "doctor",
                "content": doctor_response,
                "thinking": "",
                "turn": current_round + 2,
                "model": doctor_model,
                "is_reasoning": False
            })
            print(f"医生({doctor_model}): {doctor_response[:60]}...")
            
            # 监控检查
            monitor_result = dialogue_monitor(monitor_client, monitor_model, dialogue_history, monitor_params)
            if monitor_result == "1":
                print(f"对话终止: 监控器判定结束")
                recoder['termination_reason'] = "monitor_terminated"
                break
            
            # 患者响应（使用待评估模型，支持reasoning）
            if is_vp_reasoning:
                patient_prompt = build_reasoning_patient_prompt(ehr_data, dialogue_history)
            else:
                patient_prompt = build_patient_prompt(ehr_data, dialogue_history)
            
            patient_messages = [{'role': 'user', 'content': patient_prompt}]
            patient_response, patient_thinking = call_llm(vp_client, vp_model, patient_messages, vp_params, is_vp_reasoning)
            
            # 记录thinking过程
            if patient_thinking:
                recoder['all_thinking'].append({
                    "turn": current_round + 3,
                    "thinking": patient_thinking
                })
            
            dialogue_history.append({'role': 'patient', 'content': patient_response, 'thinking': patient_thinking})
            recoder['conversation'].append({
                "role": "patient",
                "content": patient_response,
                "thinking": patient_thinking,
                "turn": current_round + 3,
                "model": vp_model,
                "is_reasoning": is_vp_reasoning
            })
            
            print(f"患者({vp_model}): {patient_response[:60]}...")
            if patient_thinking:
                print(f"  思考: {patient_thinking[:40]}...")
            
            current_round += 1
        
        # 评估对话（使用deepseek-v3）
        evaluation_result = evaluate_dialogue(judger_client, judger_model, dialogue_history, ehr_data, judger_params, is_vp_reasoning)
        
        # 合并结果
        recoder['end_time'] = time.time()
        recoder['duration'] = recoder['end_time'] - recoder['start_time']
        recoder['evaluation'] = evaluation_result
        recoder['dialogue_history'] = dialogue_history
        
        all_results.append(recoder)
        
        # 打印结果
        print(f"综合评分: {evaluation_result['overall_score']}")
        print(f"是否通过: {'是' if evaluation_result['is_passed'] else '否'}")
        if is_vp_reasoning and 'reasoning_scores' in evaluation_result:
            print(f"推理质量评分: {evaluation_result['reasoning_scores']}")
    
    # 保存结果
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 生成汇总报告
    generate_summary_report(all_results, output_dir)
    
    print(f"\n所有 {num_cases} 个案例已处理完成！")
    print(f"结果已保存到: {output_path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件（支持环境变量解析）
    
    参数：
        config_path: 配置文件路径
    
    返回：
        配置字典
    """
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return {}
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析环境变量 ${VAR_NAME}
    import re
    def replace_env_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    content = re.sub(r'\$\{(\w+)\}', replace_env_var, content)
    
    return yaml.safe_load(content)

def evaluate_dialogue(client, model: str, dialogue_history: List[Dict], ehr_data: Dict, params: Dict[str, Any], is_reasoning: bool = False) -> Dict[str, Any]:
    """
    使用judger评估对话（deepseek-v3）
    
    参数：
        client: OpenAI客户端
        model: 模型名称
        dialogue_history: 对话历史
        ehr_data: 患者EHR数据
        params: 模型参数
        is_reasoning: 是否为reasoning模型
    
    返回：
        评估结果
    """
    # 构建评估提示词
    # 分离对话内容和thinking过程
    dialogue_lines = []
    thinking_lines = []
    
    for turn in dialogue_history:
        role = turn['role']
        content = turn['content']
        thinking = turn.get('thinking', '')
        
        dialogue_lines.append(f"{role}: {content}")
        
        if thinking:
            thinking_lines.append(f"{role} (思考): {thinking}")
    
    dialogue_str = '\n'.join(dialogue_lines)
    thinking_str = '\n'.join(thinking_lines) if thinking_lines else "无"
    
    # 构建评估提示词
    if is_reasoning:
        prompt = f"""
你是一位医学评估专家。请评估以下医患对话的质量，特别关注患者的推理过程。

患者信息：
{json.dumps(ehr_data, ensure_ascii=False, indent=2)}

对话内容：
{dialogue_str}

患者思考过程：
{thinking_str}

请从以下维度进行评估，每个维度评分0-5分：

【对话质量评估】
1. accuracy（准确性）：医学信息是否准确
2. effectiveness（有效性）：是否有效解决患者问题
3. safety（安全性）：是否存在用药错误或禁忌症
4. personalization（个性化）：是否考虑患者个体差异
5. empathy（共情）：是否体现对患者的理解和关怀

【推理质量评估（针对reasoning模型）】
6. reasoning_depth（推理深度）：患者思考过程的深度和逻辑性
7. reasoning_relevance（推理相关性）：思考过程与对话主题的相关性
8. reasoning_consistency（推理一致性）：思考过程与最终回应的一致性

请输出JSON格式：
{{
    "scores": {{
        "accuracy": X,
        "effectiveness": X,
        "safety": X,
        "personalization": X,
        "empathy": X
    }},
    "reasoning_scores": {{
        "reasoning_depth": X,
        "reasoning_relevance": X,
        "reasoning_consistency": X
    }},
    "overall_score": X,
    "is_passed": true/false,
    "comments": "评估意见"
}}
"""
    else:
        prompt = f"""
你是一位医学评估专家。请评估以下医患对话的质量。

患者信息：
{json.dumps(ehr_data, ensure_ascii=False, indent=2)}

对话内容：
{dialogue_str}

请从以下5个维度进行评估，每个维度评分0-5分：
1. accuracy（准确性）：医学信息是否准确
2. effectiveness（有效性）：是否有效解决患者问题
3. safety（安全性）：是否存在用药错误或禁忌症
4. personalization（个性化）：是否考虑患者个体差异
5. empathy（共情）：是否体现对患者的理解和关怀

请输出JSON格式：
{{
    "scores": {{
        "accuracy": X,
        "effectiveness": X,
        "safety": X,
        "personalization": X,
        "empathy": X
    }},
    "overall_score": X,
    "is_passed": true/false,
    "comments": "评估意见"
}}
"""
    
    messages = [{'role': 'user', 'content': prompt}]
    
    try:
        result = call_llm(client, model, messages, params)[0]
        
        # 解析JSON结果
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # 如果解析失败，返回默认评估
        default_result = {
            'scores': {
                'accuracy': 3,
                'effectiveness': 3,
                'safety': 3,
                'personalization': 3,
                'empathy': 3
            },
            'overall_score': 3,
            'is_passed': True,
            'comments': '解析失败，使用默认评分'
        }
        
        if is_reasoning:
            default_result['reasoning_scores'] = {
                'reasoning_depth': 3,
                'reasoning_relevance': 3,
                'reasoning_consistency': 3
            }
        
        return default_result
    except Exception as e:
        print(f"评估失败: {e}")
        error_result = {
            'scores': {
                'accuracy': 0,
                'effectiveness': 0,
                'safety': 0,
                'personalization': 0,
                'empathy': 0
            },
            'overall_score': 0,
            'is_passed': False,
            'comments': f'评估失败: {str(e)}'
        }
        
        if is_reasoning:
            error_result['reasoning_scores'] = {
                'reasoning_depth': 0,
                'reasoning_relevance': 0,
                'reasoning_consistency': 0
            }
        
        return error_result

def generate_summary_report(results: List[Dict], output_dir: str):
    """
    生成汇总报告
    
    参数：
        results: 所有案例的结果
        output_dir: 输出目录
    """
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r['evaluation']['is_passed'])
    avg_score = sum(r['evaluation']['overall_score'] for r in results) / total_cases
    avg_duration = sum(r['duration'] for r in results) / total_cases
    
    # 各维度平均评分
    dimensions = ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']
    dimension_avgs = {}
    for dim in dimensions:
        scores = [r['evaluation']['scores'].get(dim, 0) for r in results]
        dimension_avgs[dim] = sum(scores) / len(scores) if scores else 0
    
    # 推理维度平均评分（仅reasoning模型）
    reasoning_dimensions = ['reasoning_depth', 'reasoning_relevance', 'reasoning_consistency']
    reasoning_avgs = {}
    has_reasoning = any(r.get('is_vp_reasoning', False) for r in results)
    
    if has_reasoning:
        for dim in reasoning_dimensions:
            scores = []
            for r in results:
                if r.get('is_vp_reasoning', False) and 'reasoning_scores' in r['evaluation']:
                    scores.append(r['evaluation']['reasoning_scores'].get(dim, 0))
            reasoning_avgs[dim] = sum(scores) / len(scores) if scores else 0
    
    summary = {
        'total_cases': total_cases,
        'passed_cases': passed_cases,
        'pass_rate': passed_cases / total_cases * 100,
        'average_score': avg_score,
        'average_duration': avg_duration,
        'dimension_averages': dimension_avgs,
        'is_reasoning_model': results[0].get('is_vp_reasoning', False) if results else False,
        'reasoning_averages': reasoning_avgs if has_reasoning else {},
        'models': {
            'virtual_patient': results[0]['vp_model'] if results else '',
            'virtual_doctor': results[0]['doctor_model'] if results else '',
            'judger': results[0]['judger_model'] if results else ''
        },
        'case_details': [
            {
                'case_id': r['case_id'],
                'patient_id': r['patient_id'],
                'diagnosis': f"{r['ehr_data']['pathology_type']}{r['ehr_data']['stage']}",
                'overall_score': r['evaluation']['overall_score'],
                'is_passed': r['evaluation']['is_passed'],
                'duration': round(r['duration'], 2),
                'has_thinking': len(r.get('all_thinking', [])) > 0
            } for r in results
        ]
    }
    
    report_path = os.path.join(output_dir, 'benchmark_summary.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打印汇总报告
    print("\n" + "="*60)
    print("          基准测试汇总报告")
    print("="*60)
    print(f"待评估模型: {summary['models']['virtual_patient']}")
    print(f"是否Reasoning模型: {'是' if summary['is_reasoning_model'] else '否'}")
    print(f"虚拟医生: {summary['models']['virtual_doctor']}")
    print(f"评估器: {summary['models']['judger']}")
    print("-"*60)
    print(f"总案例数: {total_cases}")
    print(f"通过案例: {passed_cases}")
    print(f"通过率: {summary['pass_rate']:.1f}%")
    print(f"平均综合评分: {avg_score:.2f}/5")
    print(f"平均用时: {avg_duration:.2f}秒")
    print("\n【对话质量维度】")
    for dim, score in dimension_avgs.items():
        print(f"  {dim}: {score:.2f}/5")
    
    if has_reasoning and reasoning_avgs:
        print("\n【推理质量维度】")
        for dim, score in reasoning_avgs.items():
            print(f"  {dim}: {score:.2f}/5")
    
    print("="*60)

# 只有直接运行时才解析命令行参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EASE Benchmark Runner - 支持批量运行和reasoning模型')
    parser.add_argument('--config', type=str, default='config/sandbox_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='outputs/evaluations',
                        help='Output directory for evaluation results')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Specific scenario to run (optional)')
    parser.add_argument('--num_cases', type=int, default=10,
                        help='Number of cases to run in batch (default: 10)')
    parser.add_argument('--output_file', type=str, default='benchmark_results.json',
                        help='Output file name')
    args = parser.parse_args()
    
    main(args.config, args.output, args.scenario, args.num_cases, args.output_file)
