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

def load_standardized_patients(dataset_path: str = 'dataset/standardized_patients.json') -> List[Dict[str, Any]]:
    """
    加载标准化测试案例数据集
    
    参数：
        dataset_path: 数据集文件路径
    
    返回：
        标准化患者案例列表
    """
    if not os.path.exists(dataset_path):
        print(f"警告：标准化数据集文件不存在: {dataset_path}")
        return []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 展平所有场景的案例
    all_cases = []
    for scenario_name, scenario_data in data.get('scenarios', {}).items():
        for case in scenario_data.get('cases', []):
            case['scenario'] = scenario_name
            all_cases.append(case)
    
    print(f"已加载 {len(all_cases)} 个标准化测试案例，覆盖 {len(data.get('scenarios', {}))} 个场景")
    return all_cases

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
            # Qwen API尝试非流式调用，避免流式可能的问题
            is_qwen = 'qwen' in model.lower()
            
            if is_qwen:
                # 使用非流式调用，但需要设置enable_thinking=False
                print(f"[DEBUG] 调用Qwen模型: {model}")
                call_params = params.copy()
                call_params['extra_body'] = call_params.get('extra_body', {})
                call_params['extra_body']['enable_thinking'] = False
                
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    **call_params
                )
                
                message = chat_completion.choices[0].message
                content = message.content
                thinking = ""
                print(f"[DEBUG] Qwen返回内容长度: {len(content) if content else 0}")
                if not content:
                    print(f"[DEBUG] 完整响应: {chat_completion}")
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
            import traceback
            print(f"[DEBUG] 完整错误堆栈: {traceback.format_exc()}")
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
         parallel: int = 1,
         use_standardized: bool = False):
    """
    运行基准测试（支持批量运行）
    
    参数：
        config_path: 配置文件路径
        output_dir: 输出目录
        scenario_name: 特定场景名称（可选）
        num_cases: 批量运行的案例数量
        output_file: 输出文件名
        parallel: 并行线程数（默认1，表示顺序执行）
        use_standardized: 是否使用标准化测试案例（默认False，使用随机生成）
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
    
    # 检查是否存在缓存文件
    output_path = os.path.join(output_dir, output_file)
    cache_path = os.path.join(output_dir, f"{output_file}.tmp")
    
    # 尝试从缓存恢复
    completed_case_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            completed_case_ids = {result.get('case_id') for result in all_results}
            print(f"\n已从缓存恢复 {len(all_results)} 个已完成案例")
        except Exception as e:
            print(f"读取缓存文件失败: {e}")
    
    # 如果没有正式文件但有临时文件，尝试从临时文件恢复
    if not all_results and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            completed_case_ids = {result.get('case_id') for result in all_results}
            print(f"\n已从临时缓存恢复 {len(all_results)} 个已完成案例")
        except Exception as e:
            print(f"读取临时缓存文件失败: {e}")
    
    print(f"\n开始批量运行 {num_cases} 个案例...")
    print(f"并行模式: {'并行' if parallel > 1 else '顺序'}, 线程数: {parallel}")
    print(f"案例来源: {'标准化数据集' if use_standardized else '随机生成'}")
    
    # 生成所有患者数据（带进度条）
    print("\n[1/3] 准备患者数据...")
    if use_standardized:
        # 从标准化数据集加载
        standardized_cases = load_standardized_patients()
        if standardized_cases:
            # 根据num_cases选择案例（循环使用如果不够）
            patient_data_list = []
            for i in tqdm(range(num_cases), desc="加载标准化案例"):
                patient_data_list.append(standardized_cases[i % len(standardized_cases)])
        else:
            # 如果加载失败，回退到随机生成
            print("警告：标准化数据集加载失败，使用随机生成")
            patient_data_list = [generate_random_patient() for _ in tqdm(range(num_cases), desc="生成随机患者")]
    else:
        # 随机生成患者数据
        patient_data_list = [generate_random_patient() for _ in tqdm(range(num_cases), desc="生成随机患者")]
    
    print(f"\n[2/3] 执行案例评估...")
    
    if parallel > 1:
        # 并行处理模式
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            pending_cases = []
            
            for case_idx, ehr_data in enumerate(patient_data_list):
                # 检查是否已完成
                if (case_idx + 1) in completed_case_ids:
                    print(f"\n案例 {case_idx + 1}/{num_cases} 已完成，跳过")
                    continue
                
                pending_cases.append((case_idx, ehr_data))
            
            print(f"\n待处理案例数: {len(pending_cases)}")
            
            for case_idx, ehr_data in pending_cases:
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
                futures.append((future, case_idx + 1))
            
            print("\n等待所有案例完成...")
            try:
                for future in tqdm(concurrent.futures.as_completed([f[0] for f in futures]), total=len(futures), desc="处理案例"):
                    try:
                        result = future.result()
                        all_results.append(result)
                        # 增量保存到临时缓存
                        with open(cache_path, 'w', encoding='utf-8') as f:
                            json.dump(all_results, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        # 找到对应的case_id
                        case_id = next((cid for f, cid in futures if f == future), "unknown")
                        print(f"案例 {case_id} 执行失败: {e}")
            except KeyboardInterrupt:
                print("\n检测到中断，正在保存已完成的结果...")
                # 保存已完成的结果
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                print(f"已保存 {len(all_results)} 个案例到临时缓存: {cache_path}")
                raise
    
    else:
        # 顺序处理模式
        try:
            for case_idx, ehr_data in enumerate(tqdm(patient_data_list, desc="处理案例")):
                # 检查是否已完成
                if (case_idx + 1) in completed_case_ids:
                    print(f"\n=== 案例 {case_idx + 1}/{num_cases} ===")
                    print("已完成，跳过")
                    continue
                
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
                
                # 增量保存到临时缓存
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                
                # 打印结果
                print(f"综合评分: {evaluation_result['overall_score']}")
                print(f"是否通过: {'是' if evaluation_result['is_passed'] else '否'}")
                if is_vp_reasoning and 'reasoning_scores' in evaluation_result:
                    print(f"推理质量评分: {evaluation_result['reasoning_scores']}")
    
        except KeyboardInterrupt:
            # 顺序模式捕获KeyboardInterrupt
            print("\n检测到中断，正在保存已完成的结果...")
            # 保存已完成的结果
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"已保存 {len(all_results)} 个案例到临时缓存: {cache_path}")
            raise
    
    
    # 保存bad case到文件
    if bad_cases:
        bad_cases_path = os.path.join(output_dir, 'bad_cases.json')
        with open(bad_cases_path, 'w', encoding='utf-8') as f:
            json.dump(bad_cases, f, ensure_ascii=False, indent=2)
        print(f"\nBad cases已保存到: {bad_cases_path}")
        print(f"共收集到 {len(bad_cases)} 个不符合临床场景的案例")
    
    # [3/3] 保存结果和生成报告
    print(f"\n[3/3] 保存结果和生成报告...")
    
    # 保存结果（带进度条）
    output_path = os.path.join(output_dir, output_file)
    with tqdm(total=2, desc="保存结果", leave=False) as pbar:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        pbar.update(1)
        
        # 生成汇总报告（包含bad case统计）
        generate_summary_report(all_results, output_dir, bad_cases)
        pbar.update(1)
    
    print(f"\n✅ 所有 {num_cases} 个案例已处理完成！")
    print(f"📄 结果已保存到: {output_path}")

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

def apply_strict_evaluation_rules(dialogue_history: List[Dict], ehr_data: Dict, llm_evaluation: Dict = None) -> Dict[str, Any]:
    """
    应用严格的规则评估，对LLM评估结果进行修正和校验
    目的：确保不同大小模型之间具有良好的区分度，评分与模型大小呈正相关
    
    参数：
        dialogue_history: 对话历史
        ehr_data: 患者EHR数据
        llm_evaluation: LLM初步评估结果
    
    返回：
        修正后的严格评估结果
    """
    scores = llm_evaluation.get('scores', {}) if llm_evaluation else {}
    comments = llm_evaluation.get('comments', '') if llm_evaluation else ''
    
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
        comments += "【加分】医学知识覆盖完整。"
    elif medical_coverage_ratio >= 0.75:
        scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 0.6)
        comments += "【加分】医学知识覆盖较好。"
    elif medical_coverage_ratio >= 0.5:
        scores['accuracy'] = max(1.5, scores.get('accuracy', 3.5) - 0.5)
        comments += "【扣分】医学知识覆盖不足。"
    elif medical_coverage_ratio >= 0.25:
        scores['accuracy'] = max(1.0, scores.get('accuracy', 3.5) - 1.5)
        comments += "【严重扣分】医学知识严重不足。"
    else:
        scores['accuracy'] = max(0.5, scores.get('accuracy', 3.5) - 2.5)
        comments += "【严重扣分】几乎没有医学知识。"
    
    # === 规则2：患者顾虑深度响应（高权重）===
    patient_concerns = ehr_data.get('concerns', [])
    addressed_concerns = 0
    
    for concern in patient_concerns:
        if concern in dialogue_text:
            concern_response = [t['content'] for t in dialogue_history if concern in t['content']]
            if any(len(r) > 80 for r in concern_response):
                addressed_concerns += 1
    
    concern_coverage_ratio = addressed_concerns / len(patient_concerns) if patient_concerns else 1.0
    
    if concern_coverage_ratio == 1.0:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 1.2)
        comments += "【加分】充分回应所有顾虑。"
    elif concern_coverage_ratio >= 0.75:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.6)
        comments += "【加分】回应大部分顾虑。"
    elif concern_coverage_ratio >= 0.5:
        scores['effectiveness'] = max(1.5, scores.get('effectiveness', 3.5) - 0.5)
        comments += "【扣分】回应部分顾虑。"
    elif concern_coverage_ratio >= 0.25:
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 1.5)
        comments += "【严重扣分】回应顾虑不足。"
    else:
        scores['effectiveness'] = max(0.5, scores.get('effectiveness', 3.5) - 2.5)
        comments += "【严重扣分】未回应用户顾虑。"
    
    # === 规则3：安全性深度检查（高权重）===
    safety_keywords = ['副作用', '禁忌', '慎用', '过敏', '注意事项', '监测', '定期检查', '剂量', '遵医嘱', '不良反应']
    safety_level = sum(1 for kw in safety_keywords if kw in dialogue_text)
    
    if medications:
        if safety_level >= 4:
            scores['safety'] = min(5.0, scores.get('safety', 3.5) + 1.2)
            comments += "【加分】安全信息充分。"
        elif safety_level >= 3:
            scores['safety'] = min(5.0, scores.get('safety', 3.5) + 0.6)
            comments += "【加分】安全信息较好。"
        elif safety_level >= 2:
            scores['safety'] = max(1.5, scores.get('safety', 3.5) - 0.5)
            comments += "【扣分】安全信息不足。"
        elif safety_level >= 1:
            scores['safety'] = max(1.0, scores.get('safety', 3.5) - 1.5)
            comments += "【严重扣分】安全信息严重不足。"
        else:
            scores['safety'] = max(0.5, scores.get('safety', 3.5) - 2.5)
            comments += "【严重扣分】未提及安全信息。"
    else:
        if safety_level >= 2:
            scores['safety'] = min(5.0, scores.get('safety', 3.5) + 0.6)
        elif safety_level == 0:
            scores['safety'] = max(1.5, scores.get('safety', 3.5) - 0.8)
            comments += "【扣分】未提及安全注意事项。"
    
    # === 规则4：个性化深度检查 ===
    personal_info = [
        ('age', str(ehr_data.get('age', ''))),
        ('occupation', ehr_data.get('occupation', '')),
        ('treatment_stage', ehr_data.get('treatment_stage', '')),
        ('gender', ehr_data.get('gender', ''))
    ]
    
    personal_mentioned = sum(1 for _, info in personal_info if info and info in dialogue_text)
    
    has_customized_advice = False
    if ehr_data.get('age'):
        age_context = ['年龄', '岁', '年轻', '老年', '中年']
        has_customized_advice |= any(kw in dialogue_text for kw in age_context)
    if ehr_data.get('occupation'):
        occ_context = ['工作', '职业', '上班', '干活', '休息']
        has_customized_advice |= any(kw in dialogue_text for kw in occ_context)
    
    if personal_mentioned >= 3 and has_customized_advice:
        scores['personalization'] = min(5.0, scores.get('personalization', 3.5) + 1.2)
        comments += "【加分】个性化建议充分。"
    elif personal_mentioned >= 2:
        scores['personalization'] = min(5.0, scores.get('personalization', 3.5) + 0.5)
        comments += "【加分】有一定个性化。"
    elif personal_mentioned == 1:
        scores['personalization'] = max(1.5, scores.get('personalization', 3.5) - 0.8)
        comments += "【扣分】个性化不足。"
    else:
        scores['personalization'] = max(1.0, scores.get('personalization', 3.5) - 1.8)
        comments += "【严重扣分】缺乏个性化。"
    
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
        comments += "【加分】共情表达充分。"
    elif has_basic_empathy:
        scores['empathy'] = min(5.0, scores.get('empathy', 3.5) + 0.3)
        comments += "【加分】有共情表达。"
    else:
        scores['empathy'] = max(1.0, scores.get('empathy', 3.5) - 1.8)
        comments += "【严重扣分】缺乏共情表达。"
    
    # === 规则6：对话深度检查 ===
    patient_turns = sum(1 for turn in dialogue_history if turn.get('role') == 'patient')
    doctor_turns = sum(1 for turn in dialogue_history if turn.get('role') == 'doctor')
    
    if patient_turns >= 4 and doctor_turns >= 4:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.8)
        comments += "【加分】对话深度足够。"
    elif patient_turns >= 3 and doctor_turns >= 3:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.3)
    elif patient_turns < 2 or doctor_turns < 2:
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 2.0)
        comments += "【严重扣分】对话轮次不足。"
    
    # === 规则7：回复详细程度检查 ===
    total_doctor_content = sum(len(turn['content']) for turn in dialogue_history if turn.get('role') == 'doctor')
    avg_doctor_length = total_doctor_content / doctor_turns if doctor_turns > 0 else 0
    
    if avg_doctor_length > 600:
        scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 0.8)
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.8)
        comments += "【加分】回复非常详细。"
    elif avg_doctor_length > 400:
        scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 0.4)
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.4)
        comments += "【加分】回复较详细。"
    elif avg_doctor_length < 150:
        scores['accuracy'] = max(1.0, scores.get('accuracy', 3.5) - 1.5)
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 1.5)
        comments += "【严重扣分】回复过于简短。"
    
    # === 规则8：专业建议质量检查 ===
    advice_keywords = ['建议', '应该', '可以', '需要', '避免', '推荐', '注意', '定期', '按时', '坚持']
    advice_count = sum(1 for kw in advice_keywords if kw in dialogue_text)
    
    if advice_count >= 5:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.8)
        comments += "【加分】建议充分。"
    elif advice_count >= 3:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.3)
    elif advice_count == 0:
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 1.8)
        comments += "【严重扣分】缺乏专业建议。"
    
    # === 规则9：医学术语准确性检查 ===
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
            comments += "【扣分】未使用正确医学术语。"
    
    # === 规则10：逻辑连贯性深度检查（新增）===
    coherence_indicators = ['因此', '所以', '因为', '首先', '其次', '最后', '另外', '此外', '同时', '具体来说']
    coherence_count = sum(1 for ind in coherence_indicators if ind in dialogue_text)
    
    if coherence_count >= 4:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.6)
        comments += "【加分】逻辑连贯性强。"
    elif coherence_count >= 2:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.2)
    elif coherence_count == 0:
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 1.0)
        comments += "【扣分】逻辑连贯性弱。"
    
    # === 规则11：建议具体性和可操作性检查（新增）===
    specific_advice_patterns = [
        r'\d+[次天周]',  # 具体次数或时间
        r'[每天每周每月]',  # 具体频率
        r'[分钟小时]',  # 具体时长
        r'[上午下午晚上]',  # 具体时间
        r'[轻度中度重度]',  # 具体程度
    ]
    
    import re
    specific_advice_count = sum(
        len(re.findall(pattern, dialogue_text))
        for pattern in specific_advice_patterns
    )
    
    if specific_advice_count >= 3:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.8)
        comments += "【加分】建议具体可操作。"
    elif specific_advice_count >= 1:
        scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.3)
    else:
        scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 1.2)
        comments += "【扣分】建议缺乏具体性。"
    
    # === 规则12：复杂问题处理能力检查（新增）===
    complex_question_indicators = ['担心', '焦虑', '不确定', '不知道', '怎么办', '如何', '应该']
    has_complex_question = any(ind in dialogue_text for ind in complex_question_indicators)
    
    if has_complex_question:
        # 检查是否有系统性回答
        systematic_answer_indicators = ['首先', '第一步', '第一', '然后', '接下来', '最后', '总结']
        has_systematic_answer = any(ind in dialogue_text for ind in systematic_answer_indicators)
        
        if has_systematic_answer:
            scores['effectiveness'] = min(5.0, scores.get('effectiveness', 3.5) + 0.7)
            comments += "【加分】系统性回答复杂问题。"
        else:
            scores['effectiveness'] = max(1.0, scores.get('effectiveness', 3.5) - 0.8)
            comments += "【扣分】复杂问题处理不足。"
    
    # === 规则13：医学知识深度检查（新增）===
    deep_medical_keywords = [
        '机制', '原理', '原因', '风险', '预防', '监测指标', '正常范围', 
        '异常', '并发症', '预后', '复发', '转移', '生存率', '治疗方案',
        '适应症', '禁忌症', '药物相互作用', '代谢', '排泄', '半衰期'
    ]
    
    deep_medical_count = sum(1 for kw in deep_medical_keywords if kw in dialogue_text)
    
    if deep_medical_count >= 3:
        scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 1.0)
        comments += "【加分】医学知识深度足够。"
    elif deep_medical_count >= 1:
        scores['accuracy'] = min(5.0, scores.get('accuracy', 3.5) + 0.3)
    else:
        scores['accuracy'] = max(1.0, scores.get('accuracy', 3.5) - 0.5)
        comments += "【扣分】医学知识深度不足。"
    
    # === 重新计算综合评分（调整权重）===
    weights = {
        'accuracy': 0.40,      # 准确性权重进一步提高
        'effectiveness': 0.30,
        'safety': 0.20,        # 安全性权重
        'personalization': 0.05,
        'empathy': 0.05
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
        'is_passed': is_passed,
        'comments': comments.strip(),
        'applied_rules': [
            '医学知识深度检查',
            '患者顾虑深度响应',
            '安全性深度检查',
            '个性化深度检查',
            '共情质量检查',
            '对话深度检查',
            '回复详细程度检查',
            '专业建议质量检查',
            '医学术语准确性检查'
        ]
    }

def evaluate_dialogue(client, model: str, dialogue_history: List[Dict], ehr_data: Dict, params: Dict[str, Any], is_reasoning: bool = False) -> Dict[str, Any]:
    """
    使用judger评估对话（deepseek-v3），包含严格规则校验
    
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
    
    # 构建严格的评估提示词
    if is_reasoning:
        prompt = f"""
你是一位严格的医学评估专家。请根据以下严格标准评估医患对话质量。评分应能有效区分不同模型的性能差异。

患者信息：
{json.dumps(ehr_data, ensure_ascii=False, indent=2)}

对话内容：
{dialogue_str}

患者思考过程：
{thinking_str}

【严格评分标准】每个维度0-5分，遵循以下细则：

【对话质量评估】
1. accuracy（准确性）：
   - 5分：所有医学信息完全准确，符合最新临床指南
   - 4分：基本准确，但存在轻微表述不准确
   - 3分：有明显错误或遗漏关键医学信息
   - 2分：存在多处错误或误导性信息
   - 1分：严重错误可能危害患者健康
   - 0分：完全错误或无关内容

2. effectiveness（有效性）：
   - 5分：完全解决患者所有问题和顾虑，提供全面方案
   - 4分：解决大部分问题，但不够深入
   - 3分：解决部分问题，仍有重要疑问未解答
   - 2分：仅表面回应，未真正解决问题
   - 1分：回避问题或答非所问
   - 0分：未提供任何有用信息

3. safety（安全性）：
   - 5分：完全安全，无任何用药建议或建议完全正确
   - 4分：基本安全，无明显风险
   - 3分：存在潜在风险或未提及重要禁忌
   - 2分：有不安全建议但不严重
   - 1分：有明显危险建议
   - 0分：建议可能危及生命

4. personalization（个性化）：
   - 5分：充分考虑患者年龄、职业、合并症等个体特征
   - 4分：考虑部分个体特征
   - 3分：基本通用建议，略有个性化
   - 2分：通用建议，未考虑个体差异
   - 1分：建议与患者情况不符
   - 0分：完全忽略患者个体特征

5. empathy（共情）：
   - 5分：充分表达理解和关怀，语言温暖，给予情感支持
   - 4分：表达一定理解，有基本关怀
   - 3分：中性回应，缺乏情感共鸣
   - 2分：机械回应，略显冷漠
   - 1分：生硬回应，缺乏同理心
   - 0分：冷漠或冒犯性回应

【推理质量评估（针对reasoning模型）】
6. reasoning_depth（推理深度）：
   - 5分：深入分析，逻辑严密，考虑多种可能性
   - 4分：较好的推理，有一定深度
   - 3分：基本推理，逻辑较简单
   - 2分：表面推理，缺乏深度
   - 1分：推理薄弱或存在漏洞
   - 0分：无有效推理

7. reasoning_relevance（推理相关性）：
   - 5分：完全围绕对话主题，紧密相关
   - 4分：大部分相关，略有偏离
   - 3分：基本相关，部分不相关
   - 2分：相关性较低
   - 1分：大部分不相关
   - 0分：完全不相关

8. reasoning_consistency（推理一致性）：
   - 5分：思考过程与最终回应完全一致
   - 4分：基本一致，略有差异
   - 3分：存在一定矛盾
   - 2分：明显不一致
   - 1分：严重矛盾
   - 0分：完全矛盾

【通过标准】综合评分≥4分为通过，<4分为不通过。

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
    "comments": "详细评估意见，说明扣分原因"
}}
"""
    else:
        prompt = f"""
你是一位严格的医学评估专家。请根据以下严格标准评估医患对话质量。评分应能有效区分不同模型的性能差异。

患者信息：
{json.dumps(ehr_data, ensure_ascii=False, indent=2)}

对话内容：
{dialogue_str}

【严格评分标准】每个维度0-5分，遵循以下细则：

1. accuracy（准确性）：
   - 5分：所有医学信息完全准确，符合最新临床指南
   - 4分：基本准确，但存在轻微表述不准确
   - 3分：有明显错误或遗漏关键医学信息
   - 2分：存在多处错误或误导性信息
   - 1分：严重错误可能危害患者健康
   - 0分：完全错误或无关内容

2. effectiveness（有效性）：
   - 5分：完全解决患者所有问题和顾虑，提供全面方案
   - 4分：解决大部分问题，但不够深入
   - 3分：解决部分问题，仍有重要疑问未解答
   - 2分：仅表面回应，未真正解决问题
   - 1分：回避问题或答非所问
   - 0分：未提供任何有用信息

3. safety（安全性）：
   - 5分：完全安全，无任何用药建议或建议完全正确
   - 4分：基本安全，无明显风险
   - 3分：存在潜在风险或未提及重要禁忌
   - 2分：有不安全建议但不严重
   - 1分：有明显危险建议
   - 0分：建议可能危及生命

4. personalization（个性化）：
   - 5分：充分考虑患者年龄、职业、合并症等个体特征
   - 4分：考虑部分个体特征
   - 3分：基本通用建议，略有个性化
   - 2分：通用建议，未考虑个体差异
   - 1分：建议与患者情况不符
   - 0分：完全忽略患者个体特征

5. empathy（共情）：
   - 5分：充分表达理解和关怀，语言温暖，给予情感支持
   - 4分：表达一定理解，有基本关怀
   - 3分：中性回应，缺乏情感共鸣
   - 2分：机械回应，略显冷漠
   - 1分：生硬回应，缺乏同理心
   - 0分：冷漠或冒犯性回应

【通过标准】综合评分≥4分为通过，<4分为不通过。

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
    "comments": "详细评估意见，说明扣分原因"
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
                llm_evaluation = json.loads(json_match.group())
                # 应用严格规则评估进行修正
                strict_result = apply_strict_evaluation_rules(dialogue_history, ehr_data, llm_evaluation)
                
                # 保留reasoning_scores（如果有）
                if is_reasoning and 'reasoning_scores' in llm_evaluation:
                    strict_result['reasoning_scores'] = llm_evaluation['reasoning_scores']
                
                return strict_result
            except Exception as parse_error:
                print(f"解析LLM评估结果失败: {parse_error}")
                pass
        
        # 如果解析失败，使用默认评估并应用严格规则
        default_evaluation = {
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
        
        # 应用严格规则评估
        strict_result = apply_strict_evaluation_rules(dialogue_history, ehr_data, default_evaluation)
        
        if is_reasoning:
            strict_result['reasoning_scores'] = {
                'reasoning_depth': 3,
                'reasoning_relevance': 3,
                'reasoning_consistency': 3
            }
        
        return strict_result
    except Exception as e:
        print(f"评估失败: {e}")
        # 使用严格规则评估的错误结果
        error_evaluation = {
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
        
        strict_result = apply_strict_evaluation_rules(dialogue_history, ehr_data, error_evaluation)
        
        if is_reasoning:
            strict_result['reasoning_scores'] = {
                'reasoning_depth': 0,
                'reasoning_relevance': 0,
                'reasoning_consistency': 0
            }
        
        return strict_result

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
    parser.add_argument('--standardized', action='store_true',
                        help='Use standardized patient cases from dataset instead of randomly generated')
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
        main(args.config, args.output, args.scenario, args.num_cases, args.output_file, args.parallel, args.standardized)
