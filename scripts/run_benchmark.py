#!/usr/bin/env python3
"""
基准测试运行器 - 支持批量运行和随机化虚拟患者
参考论文代码架构设计：
- virtual patient: 待评估模型（可配置，支持reasoning模型）
- virtual doctor: 固定为 deepseek-r1
- judger: 固定为 deepseek-v3
- dialogue_monitor: 固定为 deepseek-v3

支持reasoning模型的thinking过程单独记录和评估
支持API压力测试和并行处理
"""

import argparse
import json
import os
import sys
import time
import random
import threading
import concurrent.futures
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

# Bad case示例库 - 用于提醒模型避免不符合临床场景的对话
BAD_CASE_EXAMPLES = [
    {
        "description": "避免讨论与医疗无关的技术话题",
        "bad_example": "医生：您可以用Excel做一个双轴图表来记录心率和体温。患者：好的医生，我回去立刻设个双轴图表模板。",
        "good_example": "医生：建议您记录每天的体温和心率变化，这有助于我们监测治疗效果。患者：好的医生，我会认真记录的。"
    },
    {
        "description": "避免过度的情感表达",
        "bad_example": "医生：您这份将治疗转化为系统工程的智慧，让我看到了医学与科学精神最美好的融合。患者：您这番话让我鼻子都有点酸了。",
        "good_example": "医生：您的配合和理解对治疗非常重要。患者：谢谢医生，我会继续努力的。"
    },
    {
        "description": "避免使用专业术语讨论非医疗话题",
        "bad_example": "医生：我们可以建立一个心率-体温同步曲线来分析。患者：智能手表刚好可以采集这些传感器数据。",
        "good_example": "医生：请留意自己的身体感受，如有异常及时告诉我。患者：好的，我会注意观察的。"
    },
    {
        "description": "避免闲聊日常话题",
        "bad_example": "医生：今天天气真好啊。患者：是啊，春天来了。",
        "good_example": "医生：今天感觉怎么样？患者：还是有点乏力。"
    }
]

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
    
    # 构建bad case示例提醒
    bad_case_reminder = "\n".join([
        f"- {case['description']}" for case in BAD_CASE_EXAMPLES
    ])
    
    prompt = f"""
你是一位专业的乳腺科医生。请根据患者的病史信息和对话历史，给出专业的医疗建议。

**身份要求：**
- 保持专业医生的身份，使用医学专业知识回答
- 语言要符合患者的年龄和文化水平，避免过于技术化的术语
- 表达要温和、有同理心，但不要过度情绪化
- 专注于医疗相关话题，不要讨论与病情无关的内容
- 回复长度控制在150字左右，保持简洁清晰

**避免以下不符合临床场景的对话：**
{bad_case_reminder}

患者信息：
年龄：{age}岁
职业：{occupation}
诊断：{pathology_type}{stage}
手术类型：{surgery_type}
当前用药：{medication_str}
治疗阶段：{treatment_stage}

对话历史：
{history_str}

请以医生身份，基于以上信息，对患者的最新发言做出专业、温和、有同理心的医疗回应。
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
    
    # 构建bad case示例提醒
    bad_case_reminder = "\n".join([
        f"- {case['description']}" for case in BAD_CASE_EXAMPLES
    ])
    
    prompt = f"""
你是一位乳腺癌患者。请根据你的病情信息和对话历史，以患者身份进行回应。

**身份要求：**
- 保持患者身份，符合你的年龄和职业背景
- 语言要自然、真实，符合普通人的表达习惯
- 关注自己的病情和治疗，提出与健康相关的问题
- 不要讨论与医疗无关的话题（如技术细节、日常闲聊等）

**避免以下不符合临床场景的对话：**
{bad_case_reminder}

患者信息：
年龄：{age}岁
职业：{occupation}
当前症状：{symptom_str}
主要顾虑：{concern_str}

对话历史：
{history_str}

请以患者身份，基于以上信息，对医生的最新发言做出真实、自然的医疗相关回应。
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
    
    # 构建bad case示例提醒
    bad_case_reminder = "\n".join([
        f"- {case['description']}" for case in BAD_CASE_EXAMPLES
    ])
    
    prompt = f"""
你是一位乳腺癌患者。请根据你的病情信息和对话历史，以患者身份进行回应。

**身份要求：**
- 保持患者身份，符合你的年龄和职业背景
- 语言要自然、真实，符合普通人的表达习惯
- 关注自己的病情和治疗，提出与健康相关的问题
- 不要讨论与医疗无关的话题（如技术细节、日常闲聊等）

**避免以下不符合临床场景的对话：**
{bad_case_reminder}

患者信息：
年龄：{age}岁
职业：{occupation}
当前症状：{symptom_str}
主要顾虑：{concern_str}

对话历史：
{history_str}

请按照以下格式输出你的思考过程和最终回应：

思考：[详细描述你作为患者的思考过程，包括你为什么会这样回应，你的担忧是什么，你想从医生那里得到什么信息等]

回应：[你的最终回答，以患者身份对医生的最新发言做出真实、自然的医疗相关回应]

注意：思考部分要详细，展示你的推理过程。
"""
    return prompt.strip()

def call_llm(client, model: str, messages: List[Dict], params: Dict[str, Any], is_reasoning: bool = False, max_retries: int = 3) -> Tuple[str, str]:
    """
    调用LLM模型（支持reasoning模型的thinking输出，支持重试机制）
    
    参数：
        client: OpenAI客户端
        model: 模型名称
        messages: 消息列表
        params: 模型参数
        is_reasoning: 是否为reasoning模型
        max_retries: 最大重试次数
    
    返回：
        (content, thinking): 内容和思考过程（非reasoning模型thinking为空）
    """
    for attempt in range(max_retries):
        try:
            # Qwen API需要使用流式调用来避免enable_thinking错误
            is_qwen = 'qwen' in model.lower()
            
            if is_qwen:
                # 使用流式调用
                stream_response = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=True,
                    **params
                )
                content = ''
                for chunk in stream_response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                thinking = ""
            else:
                # 非流式调用
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
                # 只有当is_reasoning=True时才解析thinking
                if not thinking and is_reasoning and content:
                    if '思考：' in content and '回应：' in content:
                        thinking = content.split('思考：')[1].split('回应：')[0].strip()
                        content = content.split('回应：')[1].strip()
                    elif 'Thinking:' in content and 'Response:' in content:
                        thinking = content.split('Thinking:')[1].split('Response:')[0].strip()
                        content = content.split('Response:')[1].strip()
                # 如果is_reasoning=False，清空thinking内容
                elif not is_reasoning:
                    thinking = ""
            
            # 添加延迟避免API限流（根据模型类型调整延迟）
            if 'deepseek' in model.lower():
                time.sleep(1.5)  # DeepSeek API需要更长延迟
            else:
                time.sleep(0.5)
            
            return content, thinking
        except Exception as e:
            print(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # 重试前等待更长时间
                wait_time = (attempt + 1) * 3
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("已达到最大重试次数，返回空结果")
                return "", ""

def dialogue_monitor(client, model: str, msgs: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    对话监控器（判断对话是否应该终止）
    
    参数：
        client: OpenAI客户端
        model: 模型名称
        msgs: 消息列表
        params: 模型参数
    
    返回：
        监控结果字典：
        - should_terminate: 是否终止（True/False）
        - violation_type: 违规类型（'none'/'off_topic'/'role_play_failure'/'goal_achieved'/'deadlock'）
        - warning_message: 警告消息（用于注入到prompt中）
    """
    # 过滤thinking部分，只传递对话内容给监控器
    clean_messages = []
    for msg in msgs:
        clean_msg = {'role': msg['role'], 'content': msg['content']}
        clean_messages.append(clean_msg)
    
    # 构建监控提示词，增强临床场景检查
    # 简化提示词以确保模型返回有效JSON
    monitor_prompt = f"""
分析对话，输出JSON。

对话：{json.dumps(clean_messages, ensure_ascii=False)}

规则：
- 符合医疗场景、需继续：should_terminate=false, violation_type="none"
- 偏离主题：should_terminate=false, violation_type="off_topic", warning_message="请专注于医疗话题"
- 角色失败：should_terminate=false, violation_type="role_play_failure", warning_message="请保持角色"
- 目标达成：should_terminate=true, violation_type="goal_achieved"
- 陷入僵局：should_terminate=true, violation_type="deadlock"

直接输出JSON，不要解释。
"""
    
    messages = [{'role': 'user', 'content': monitor_prompt}]
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            **params
        )
        result = chat_completion.choices[0].message.content.strip()
        
        # 尝试解析JSON
        try:
            result_dict = json.loads(result)
            return {
                'should_terminate': result_dict.get('should_terminate', False),
                'violation_type': result_dict.get('violation_type', 'none'),
                'warning_message': result_dict.get('warning_message', '')
            }
        except json.JSONDecodeError:
            print(f"监控器返回非JSON格式: {result}")
            return {'should_terminate': False, 'violation_type': 'none', 'warning_message': ''}
            
    except Exception as e:
        print(f"监控器调用失败: {e}")
        return {'should_terminate': False, 'violation_type': 'none', 'warning_message': ''}

def main(config_path: str = 'config/sandbox_config.yaml', 
         output_dir: str = 'outputs/evaluations', 
         scenario_name: str = None,
         num_cases: int = 10,
         output_file: str = 'benchmark_results.json',
         parallel: int = 1):
    """
    运行基准测试（支持批量运行）
    
    参数：
        config_path: 配置文件路径
        output_dir: 输出目录
        scenario_name: 特定场景名称（可选）
        num_cases: 批量运行的案例数量
        output_file: 输出文件名
        parallel: 并行线程数（默认1，表示顺序执行）
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
    bad_cases = []  # 收集不符合临床场景的bad case
    max_retries_per_case = 3  # 每个案例最大重试次数
    
    print(f"\n开始批量运行 {num_cases} 个案例...")
    print(f"并行模式: {'并行' if parallel > 1 else '顺序'}, 线程数: {parallel}")
    
    # 生成所有患者数据
    patient_data_list = [generate_random_patient() for _ in range(num_cases)]
    
    if parallel > 1:
        # 并行处理模式
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            for case_idx, ehr_data in enumerate(patient_data_list):
                print(f"\n提交案例 {case_idx + 1}/{num_cases}")
                print(f"患者ID: {ehr_data['patient_id']}")
                print(f"年龄: {ehr_data['age']}岁, 职业: {ehr_data['occupation']}")
                print(f"诊断: {ehr_data['pathology_type']}{ehr_data['stage']}")
                
                future = executor.submit(
                    run_single_case,
                    case_idx, ehr_data,
                    vp_client, vp_model, is_vp_reasoning,
                    doctor_client, doctor_model, doctor_params,
                    judger_client, judger_model, judger_params,
                    monitor_client, monitor_model, monitor_params,
                    vp_params, max_retries_per_case
                )
                futures.append(future)
            
            print("\n等待所有案例完成...")
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_cases, desc="处理案例"):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"案例执行失败: {e}")
    
    else:
        # 顺序处理模式
        for case_idx, ehr_data in enumerate(tqdm(patient_data_list, desc="处理案例")):
            print(f"\n=== 案例 {case_idx + 1}/{num_cases} ===")
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
                "start_time": time.time(),
                "retry_count": 0  # 记录重试次数
            }
            
            # 案例重试循环
            case_success = False
            retry_count = 0
            
            while not case_success and retry_count < max_retries_per_case:
                # 重置对话历史
                dialogue_history = []
                current_round = 0
                max_rounds = 8
                
                # 患者初始问题（随机选择）
                initial_questions = [
                    f"医生，我最近感觉{ehr_data['current_symptoms'][0]}，不知道是不是正常的？",
                    f"医生您好，我是{ehr_data['occupation']}，最近在接受{ehr_data['treatment_stage']}，有些担心。",
                    f"医生，我做完{ehr_data['surgery_type']}后感觉不太舒服，想咨询一下。",
                    f"您好医生，我想了解一下我现在的{ehr_data['treatment_stage']}需要注意什么？",
                ]
                
                patient_response = random.choice(initial_questions)
                dialogue_history.append({'role': 'patient', 'content': patient_response, 'thinking': ''})
                
                # 重置recoder的conversation和thinking
                recoder['conversation'] = [{
                    "role": "patient",
                    "content": patient_response,
                    "thinking": "",
                    "turn": 1,
                    "model": vp_model,
                    "is_reasoning": is_vp_reasoning
                }]
                recoder['all_thinking'] = []
                
                print(f"患者({vp_model}): {patient_response[:60]}...")
            
                # 标记是否因不符合场景被终止
                scene_violation = False
                # 记录当前轮是否已经发出警告
                has_warned = False
                
                while current_round < max_rounds and not scene_violation:
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
                
                # 监控检查 - 每3轮检查一次（优化性能）
                if (current_round + 1) % 3 == 0 or current_round == max_rounds - 1:
                    monitor_result = dialogue_monitor(monitor_client, monitor_model, dialogue_history, monitor_params)
                else:
                    monitor_result = {'should_terminate': False, 'violation_type': 'none', 'warning_message': ''}
                
                if monitor_result['should_terminate']:
                    print(f"监控器检测到问题: {monitor_result['violation_type']}")
                    
                    # 如果是对话目标达成或陷入僵局，直接终止
                    if monitor_result['violation_type'] in ['goal_achieved', 'deadlock']:
                        print(f"对话终止: {monitor_result['violation_type']}")
                        recoder['termination_reason'] = monitor_result['violation_type']
                        scene_violation = True
                        break
                    
                    # 如果已经警告过但仍然违规，则中断重启
                    if has_warned:
                        print(f"警告后仍不符合场景，中断重启")
                        
                        # 记录bad case
                        bad_case = {
                            "case_id": case_idx + 1,
                            "retry_attempt": retry_count + 1,
                            "ehr_data": ehr_data,
                            "dialogue_history": dialogue_history.copy(),
                            "termination_reason": f"warning_failed_{monitor_result['violation_type']}",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "warning_message": monitor_result.get('warning_message', '')
                        }
                        bad_cases.append(bad_case)
                        
                        scene_violation = True
                        retry_count += 1
                        print(f"第 {retry_count}/{max_retries_per_case} 次重试...")
                        break
                    else:
                        # 第一次检测到错误，注入警告到prompt中
                        warning_msg = monitor_result.get('warning_message', '请回到医疗相关话题')
                        print(f"注入警告: {warning_msg}")
                        has_warned = True
                        
                        # 将警告注入到对话历史中，后续轮次的prompt会包含这个警告
                        dialogue_history.append({
                            'role': 'system', 
                            'content': f"【警告】{warning_msg}。请保持专业的医患对话，专注于医疗相关话题。"
                        })
                
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
            
            # 如果正常结束对话（没有场景违规）
            if not scene_violation:
                case_success = True
        
        # 更新重试次数
        recoder['retry_count'] = retry_count
        
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
    
    # 保存bad case到文件
    if bad_cases:
        bad_cases_path = os.path.join(output_dir, 'bad_cases.json')
        with open(bad_cases_path, 'w', encoding='utf-8') as f:
            json.dump(bad_cases, f, ensure_ascii=False, indent=2)
        print(f"\nBad cases已保存到: {bad_cases_path}")
        print(f"共收集到 {len(bad_cases)} 个不符合临床场景的案例")
    
    # 保存结果
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 生成汇总报告（包含bad case统计）
    generate_summary_report(all_results, output_dir, bad_cases)
    
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

def generate_summary_report(results: List[Dict], output_dir: str, bad_cases: List[Dict] = None):
    """
    生成汇总报告
    
    参数：
        results: 所有案例的结果
        output_dir: 输出目录
        bad_cases: bad case列表（用于统计）
    """
    if bad_cases is None:
        bad_cases = []
    
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
    
    # Bad case统计分析
    total_bad_cases = len(bad_cases)
    bad_case_rate = total_bad_cases / (total_cases + total_bad_cases) * 100 if (total_cases + total_bad_cases) > 0 else 0
    
    # 按错误类型统计bad case
    error_type_counts = {}
    for bad_case in bad_cases:
        # 分析bad case的错误类型
        dialogue_history = bad_case.get('dialogue_history', [])
        if dialogue_history:
            last_turn = dialogue_history[-1]
            content = last_turn.get('content', '')
            
            # 根据内容判断错误类型
            error_type = "unknown"
            if any(keyword in content for keyword in ['图表', '双轴', '曲线', 'Excel', '数据', '传感器']):
                error_type = "technical_topic"  # 技术话题
            elif any(keyword in content for keyword in ['酸了', '感动', '美好的融合', '智慧', '工程师思维']):
                error_type = "emotional_excess"  # 过度情感表达
            elif any(keyword in content for keyword in ['天气', '日常', '闲聊', '吃饭', '休息']):
                error_type = "casual_chitchat"  # 日常闲聊
            elif any(keyword in content for keyword in ['谢谢', '感谢', '不客气']):
                error_type = "excessive_politeness"  # 过度客套
            
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
    
    # 重试次数统计
    total_retries = sum(r.get('retry_count', 0) for r in results)
    avg_retries_per_case = total_retries / total_cases if total_cases > 0 else 0
    
    summary = {
        'total_cases': total_cases,
        'passed_cases': passed_cases,
        'pass_rate': passed_cases / total_cases * 100,
        'average_score': avg_score,
        'average_duration': avg_duration,
        'dimension_averages': dimension_avgs,
        'is_reasoning_model': results[0].get('is_vp_reasoning', False) if results else False,
        'reasoning_averages': reasoning_avgs if has_reasoning else {},
        # Bad case统计指标
        'bad_case_statistics': {
            'total_bad_cases': total_bad_cases,
            'bad_case_rate': bad_case_rate,
            'error_type_distribution': error_type_counts,
            'total_retries': total_retries,
            'average_retries_per_case': avg_retries_per_case
        },
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
                'retry_count': r.get('retry_count', 0),
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
    
    # 打印Bad case统计
    print("\n【Bad Case统计】")
    print(f"  总bad case数: {total_bad_cases}")
    print(f"  Bad case发生率: {bad_case_rate:.2f}%")
    print(f"  总重试次数: {total_retries}")
    print(f"  平均每案例重试: {avg_retries_per_case:.2f}次")
    if error_type_counts:
        print("  错误类型分布:")
        error_type_labels = {
            "technical_topic": "技术话题偏离",
            "emotional_excess": "过度情感表达",
            "casual_chitchat": "日常闲聊",
            "excessive_politeness": "过度客套",
            "unknown": "未知类型"
        }
        for error_type, count in error_type_counts.items():
            label = error_type_labels.get(error_type, error_type)
            percentage = count / total_bad_cases * 100 if total_bad_cases > 0 else 0
            print(f"    - {label}: {count}次 ({percentage:.1f}%)")
    
    print("="*60)

def run_single_case(case_idx: int, ehr_data: Dict[str, Any], vp_client, vp_model, is_vp_reasoning, 
                    doctor_client, doctor_model, doctor_params, 
                    judger_client, judger_model, judger_params,
                    monitor_client, monitor_model, monitor_params,
                    vp_params, max_retries_per_case: int = 3) -> Dict[str, Any]:
    """
    运行单个案例（用于并行处理）
    
    参数：
        case_idx: 案例索引
        ehr_data: 患者EHR数据
        其他参数：各客户端和模型配置
    
    返回：
        案例结果
    """
    print(f"\n=== 案例 {case_idx + 1} ===")
    print(f"患者ID: {ehr_data['patient_id']}")
    print(f"年龄: {ehr_data['age']}岁, 职业: {ehr_data['occupation']}")
    print(f"诊断: {ehr_data['pathology_type']}{ehr_data['stage']}")
    
    recoder = {
        "case_id": case_idx + 1,
        "patient_id": ehr_data['patient_id'],
        "ehr_data": ehr_data,
        "vp_model": vp_model,
        "is_vp_reasoning": is_vp_reasoning,
        "doctor_model": doctor_model,
        "judger_model": judger_model,
        "conversation": [],
        "all_thinking": [],
        "start_time": time.time(),
        "retry_count": 0
    }
    
    case_success = False
    retry_count = 0
    
    while not case_success and retry_count < max_retries_per_case:
        dialogue_history = []
        current_round = 0
        max_rounds = 8
        
        initial_questions = [
            f"医生，我最近感觉{ehr_data['current_symptoms'][0]}，不知道是不是正常的？",
            f"医生您好，我是{ehr_data['occupation']}，最近在接受{ehr_data['treatment_stage']}，有些担心。",
            f"医生，我做完{ehr_data['surgery_type']}后感觉不太舒服，想咨询一下。",
            f"您好医生，我想了解一下我现在的{ehr_data['treatment_stage']}需要注意什么？",
        ]
        
        patient_response = random.choice(initial_questions)
        dialogue_history.append({'role': 'patient', 'content': patient_response, 'thinking': ''})
        
        recoder['conversation'] = [{
            "role": "patient",
            "content": patient_response,
            "thinking": "",
            "turn": 1,
            "model": vp_model,
            "is_reasoning": is_vp_reasoning
        }]
        recoder['all_thinking'] = []
        
        print(f"患者({vp_model}): {patient_response[:60]}...")
        
        scene_violation = False
        has_warned = False
        
        while current_round < max_rounds and not scene_violation:
            # 医生响应
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
            
            # 监控检查 - 每3轮检查一次
            if (current_round + 1) % 3 == 0 or current_round == max_rounds - 1:
                monitor_result = dialogue_monitor(monitor_client, monitor_model, dialogue_history, monitor_params)
            else:
                monitor_result = {'should_terminate': False, 'violation_type': 'none', 'warning_message': ''}
            
            if monitor_result['should_terminate']:
                print(f"监控器检测到问题: {monitor_result['violation_type']}")
                
                if monitor_result['violation_type'] in ['goal_achieved', 'deadlock']:
                    print(f"对话终止: {monitor_result['violation_type']}")
                    recoder['termination_reason'] = monitor_result['violation_type']
                    scene_violation = True
                    break
                
                if has_warned:
                    print(f"警告后仍不符合场景，中断重启")
                    scene_violation = True
                    retry_count += 1
                    print(f"第 {retry_count}/{max_retries_per_case} 次重试...")
                    break
                else:
                    warning_msg = monitor_result.get('warning_message', '请回到医疗相关话题')
                    print(f"注入警告: {warning_msg}")
                    has_warned = True
                    dialogue_history.append({
                        'role': 'system', 
                        'content': f"【警告】{warning_msg}。请保持专业的医患对话，专注于医疗相关话题。"
                    })
            
            # 患者响应
            if is_vp_reasoning:
                patient_prompt = build_reasoning_patient_prompt(ehr_data, dialogue_history)
            else:
                patient_prompt = build_patient_prompt(ehr_data, dialogue_history)
            
            patient_messages = [{'role': 'user', 'content': patient_prompt}]
            patient_response, patient_thinking = call_llm(vp_client, vp_model, patient_messages, vp_params, is_vp_reasoning)
            
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
        
        if not scene_violation:
            case_success = True
    
    recoder['retry_count'] = retry_count
    
    # 评估对话
    evaluation_result = evaluate_dialogue(judger_client, judger_model, dialogue_history, ehr_data, judger_params, is_vp_reasoning)
    
    recoder['end_time'] = time.time()
    recoder['duration'] = recoder['end_time'] - recoder['start_time']
    recoder['evaluation'] = evaluation_result
    recoder['dialogue_history'] = dialogue_history
    
    print(f"综合评分: {evaluation_result['overall_score']}")
    print(f"是否通过: {'是' if evaluation_result['is_passed'] else '否'}")
    
    return recoder

def api_stress_test(client, model: str, test_prompts: List[str], concurrent_requests: int = 5, iterations: int = 10):
    """
    API压力测试函数
    
    参数：
        client: LLM客户端
        model: 模型名称
        test_prompts: 测试提示词列表
        concurrent_requests: 并发请求数
        iterations: 迭代次数
    
    返回：
        测试结果统计
    """
    print(f"\n=== 开始API压力测试 ===")
    print(f"模型: {model}")
    print(f"并发数: {concurrent_requests}")
    print(f"迭代次数: {iterations}")
    
    results = {
        'total_requests': 0,
        'success_count': 0,
        'failure_count': 0,
        'total_time': 0,
        'min_time': float('inf'),
        'max_time': 0,
        'avg_time': 0,
        'error_messages': []
    }
    
    def single_request(prompt: str):
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                model=model,
                max_tokens=50,
                temperature=0.1
            )
            duration = time.time() - start_time
            return {'success': True, 'duration': duration, 'error': None}
        except Exception as e:
            duration = time.time() - start_time
            return {'success': False, 'duration': duration, 'error': str(e)}
    
    for iteration in range(iterations):
        print(f"\n迭代 {iteration + 1}/{iterations}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(single_request, random.choice(test_prompts)) for _ in range(concurrent_requests)]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results['total_requests'] += 1
                
                if result['success']:
                    results['success_count'] += 1
                    results['total_time'] += result['duration']
                    results['min_time'] = min(results['min_time'], result['duration'])
                    results['max_time'] = max(results['max_time'], result['duration'])
                else:
                    results['failure_count'] += 1
                    if result['error'] not in results['error_messages']:
                        results['error_messages'].append(result['error'])
    
    if results['success_count'] > 0:
        results['avg_time'] = results['total_time'] / results['success_count']
    
    print("\n=== 压力测试结果 ===")
    print(f"总请求数: {results['total_requests']}")
    print(f"成功数: {results['success_count']}")
    print(f"失败数: {results['failure_count']}")
    print(f"成功率: {results['success_count']/results['total_requests']*100:.1f}%")
    print(f"平均响应时间: {results['avg_time']:.2f}秒")
    print(f"最小响应时间: {results['min_time']:.2f}秒")
    print(f"最大响应时间: {results['max_time']:.2f}秒")
    
    if results['error_messages']:
        print("\n错误信息:")
        for error in results['error_messages'][:5]:
            print(f"  - {error}")
    
    return results

# 只有直接运行时才解析命令行参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EASE Benchmark Runner - 支持批量运行、reasoning模型和并行处理')
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
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel threads (default: 1, use 1 for sequential)')
    parser.add_argument('--stress_test', action='store_true',
                        help='Run API stress test instead of benchmark')
    parser.add_argument('--stress_concurrent', type=int, default=5,
                        help='Number of concurrent requests for stress test (default: 5)')
    parser.add_argument('--stress_iterations', type=int, default=10,
                        help='Number of iterations for stress test (default: 10)')
    args = parser.parse_args()
    
    if args.stress_test:
        # 运行API压力测试
        from src.utils.llm_client_factory import LLMClientFactory
        from dotenv import load_dotenv
        load_dotenv()
        
        config = load_config(args.config)
        factory = LLMClientFactory(config)
        
        # 使用虚拟患者客户端进行测试
        vp_client = factory.create_virtual_patient_client()
        vp_model = factory.get_virtual_patient_model()
        
        # 测试提示词
        test_prompts = [
            "你好",
            "请介绍一下乳腺癌的治疗方法",
            "什么是化疗",
            "如何缓解化疗副作用",
            "乳腺癌患者需要注意什么"
        ]
        
        results = api_stress_test(vp_client, vp_model, test_prompts, args.stress_concurrent, args.stress_iterations)
        
        # 保存测试结果
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'stress_test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n测试结果已保存到: {os.path.join(args.output, 'stress_test_results.json')}")
    else:
        # 运行基准测试
        main(args.config, args.output, args.scenario, args.num_cases, args.output_file, args.parallel)
