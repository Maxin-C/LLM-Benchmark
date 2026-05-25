"""
虚拟患者Agent
整合EHR数据、对话数据和患者画像图谱，构建具有真实行为特征的虚拟患者
支持场景动态加载和个性化提示，提升患者真实性和针对性
使用LLM驱动对话响应
"""

import json
from typing import Dict, List, Any, Optional
from src.utils.llm_client import LLMClient

# 延迟导入场景管理器，避免循环依赖
def _get_scenario_manager():
    from src.utils.scenario_manager import get_scenario_manager
    return get_scenario_manager()

class VirtualPatientAgent:
    def __init__(self, ehr_data: Dict[str, Any], dialogue_data: List[Dict[str, Any]], 
                 persona_graph: Dict[str, Any], llm_client: LLMClient, 
                 scenario_type: Optional[str] = None):
        """
        初始化虚拟患者Agent
        
        参数：
            ehr_data: EHR病历数据
            dialogue_data: 对话历史数据
            persona_graph: 患者画像图谱
            llm_client: LLM客户端实例
            scenario_type: 指定场景类型（可选，若不指定则自动推断）
        """
        self.state = self._initialize_state(ehr_data, dialogue_data, persona_graph)
        self.llm_client = llm_client
        
        # 场景管理
        self.scenario_manager = _get_scenario_manager()
        self.scenario_type = scenario_type or self._infer_scenario()
        self.scenario_config = self.scenario_manager.get_scenario_config(self.scenario_type)
        
        self._build_system_prompt()
    
    def _initialize_state(self, ehr_data: Dict[str, Any], dialogue_data: List[Dict[str, Any]], 
                          persona_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        从三源数据初始化患者状态
        """
        return {
            # 来自EHR数据
            'demographics': {
                'age': ehr_data.get('age'),
                'gender': ehr_data.get('gender'),
                'occupation': ehr_data.get('occupation')
            },
            'medical_info': {
                'pathology_type': ehr_data.get('pathology_type'),
                'stage': ehr_data.get('stage'),
                'surgery_type': ehr_data.get('surgery_type'),
                'medications': ehr_data.get('medications', []),
                'treatment_stage': ehr_data.get('treatment_stage')
            },
            
            # 来自对话数据
            'symptoms': self._extract_symptoms(dialogue_data),
            'concerns': self._extract_concerns(dialogue_data),
            'communication_style': self._analyze_communication_style(dialogue_data),
            'emotion': self._analyze_emotion(dialogue_data),
            
            # 来自Persona图谱
            'compliance': persona_graph.get('compliance', 'medium'),
            'cognitive_biases': persona_graph.get('cognitive_biases', []),
            'social_network': persona_graph.get('social_network', {}),
            'history_symptoms': persona_graph.get('history_symptoms', []),
            
            # 动态状态
            'current_mood': 'neutral',
            'disease_progression': 0,
            'adverse_reactions': [],
            'interaction_history': []
        }
    
    def _infer_scenario(self) -> str:
        """
        根据患者数据自动推断场景类型
        
        返回：
            推断出的场景类型
        """
        medical_info = self.state.get('medical_info', {})
        concerns = self.state.get('concerns', [])
        symptoms = self.state.get('symptoms', [])
        
        # 根据治疗阶段推断场景
        treatment_stage = medical_info.get('treatment_stage', '')
        
        if '化疗' in treatment_stage or '化疗' in concerns:
            return '化疗相关'
        elif '手术' in treatment_stage or '手术' in concerns:
            return '手术相关'
        elif '放疗' in treatment_stage or '放疗' in concerns:
            return '放疗相关'
        elif '内分泌' in treatment_stage or '内分泌' in concerns:
            return '内分泌治疗'
        
        # 根据关注点推断场景
        concern_keywords = {
            '性生活': '性生活与康复',
            '性': '性生活与康复',
            '复发': '复发转移',
            '转移': '复发转移',
            '复查': '复查随访',
            '检查': '复查随访',
            '怀孕': '生育哺乳',
            '生育': '生育哺乳',
            '哺乳': '生育哺乳',
            '疼痛': '疼痛管理',
            '痛': '疼痛管理',
            '胀': '疼痛管理',
            '饮食': '饮食营养',
            '吃': '饮食营养',
            '心理': '心理支持',
            '担心': '心理支持',
            '焦虑': '心理支持'
        }
        
        for concern in concerns:
            for keyword, scenario in concern_keywords.items():
                if keyword in concern:
                    return scenario
        
        # 根据症状推断场景
        for symptom in symptoms:
            if any(kw in symptom for kw in ['痛', '胀', '疼']):
                return '疼痛管理'
        
        # 默认场景
        return '复查随访'
    
    def _build_system_prompt(self):
        """构建LLM系统提示词，集成场景动态加载和个性化提示"""
        # 基础患者信息
        base_info = f"""
你是一位{self.state['demographics']['age']}岁的{self.state['demographics']['gender']}患者，正在与医生进行咨询。

【患者背景】
- 疾病：{self.state['medical_info']['pathology_type']}
- 分期：{self.state['medical_info']['stage']}
- 手术：{self.state['medical_info']['surgery_type']}
- 当前治疗阶段：{self.state['medical_info']['treatment_stage']}
- 用药：{', '.join(self.state['medical_info']['medications']) if self.state['medical_info']['medications'] else '无'}

【当前身体状况】
- 症状：{', '.join(self.state['symptoms']) if self.state['symptoms'] else '暂无明显不适'}
- 情绪状态：{self._get_mood_description(self.state['current_mood'])}
- 用药依从性：{self.state['compliance']}

【性格特点】
- 认知特点：{', '.join(self.state['cognitive_biases']) if self.state['cognitive_biases'] else '无特殊认知倾向'}
- 社交支持：{self.state['social_network'].get('support_level', '中等')}
""".strip()
        
        # 场景特定提示
        scenario_prompt = self._build_scenario_prompt()
        
        # 通用角色要求
        role_requirements = f"""

【角色要求】
1. 以第一人称回应医生，表现出真实患者的担忧和问题
2. 回复要自然、简短，符合患者的身份和情绪状态
3. 可以适当追问医生，表达自己的疑虑和担忧
4. 不要使用专业医学术语，用日常语言表达

【当前关注点】
{', '.join(self.state['concerns']) if self.state['concerns'] else '关心治疗效果和康复进度'}
""".strip()
        
        self.system_prompt = "\n".join([base_info, scenario_prompt, role_requirements])
    
    def _build_scenario_prompt(self) -> str:
        """
        根据场景类型构建个性化提示
        
        返回：
            场景特定的提示文本
        """
        if not self.scenario_config:
            return ""
        
        prompt_parts = []
        
        # 添加场景描述
        prompt_parts.append(f"【当前场景】{self.scenario_type}")
        prompt_parts.append(f"场景描述：{self.scenario_config.description}")
        
        # 添加场景关键词
        if self.scenario_config.keywords:
            prompt_parts.append(f"关键词：{', '.join(self.scenario_config.keywords)}")
        
        # 添加参考案例（few-shot示例）
        if self.scenario_config.examples:
            prompt_parts.append("\n【参考案例】")
            for i, example in enumerate(self.scenario_config.examples[:2], 1):
                # 提取患者问题部分
                input_text = example.input_text
                # 找到患者的实际问题
                patient_question = self._extract_patient_question(input_text)
                prompt_parts.append(f"案例{i}：{patient_question}")
        
        return "\n".join(prompt_parts)
    
    def _extract_patient_question(self, text: str) -> str:
        """
        从对话文本中提取患者的核心问题
        
        参数：
            text: 对话文本
        
        返回：
            患者问题的简短描述
        """
        # 找到患者直接提问的部分
        lines = text.split('\n')
        for line in lines:
            if line.startswith('大夫') or line.startswith('医生') or line.startswith('我'):
                if '？' in line or '?' in line:
                    return line.strip()[:100]
        
        # 如果找不到直接提问，返回前100个字符
        return text.strip()[:100]
    
    def _get_mood_description(self, mood: str) -> str:
        """获取情绪描述"""
        mood_map = {
            'anxious': '焦虑不安',
            'worried': '担心忧虑',
            'neutral': '平静',
            'positive': '积极乐观'
        }
        return mood_map.get(mood, '平静')
    
    def _extract_symptoms(self, dialogue_data: List[Dict[str, Any]]) -> List[str]:
        """从对话中提取症状实体"""
        symptoms = []
        symptom_keywords = ['痛', '胀', '肿', '麻', '痒', '恶心', '呕吐', '头晕', '乏力', '发热', 
                           '疼痛', '酸痛', '胀痛', '刺痛', '隐痛', '胸闷', '气短', '食欲不振']
        
        for msg in dialogue_data:
            content = msg.get('content', '')
            for keyword in symptom_keywords:
                if keyword in content:
                    symptoms.append(keyword)
        
        return list(set(symptoms))
    
    def _extract_concerns(self, dialogue_data: List[Dict[str, Any]]) -> List[str]:
        """从对话中提取患者关注点"""
        concerns = []
        concern_patterns = ['怎么办', '需要', '应该', '注意', '能不能', '多久', '会好吗', '要紧吗',
                           '影响', '后果', '复发', '转移', '副作用', '后遗症', '康复', '复查']
        
        for msg in dialogue_data:
            content = msg.get('content', '')
            for pattern in concern_patterns:
                if pattern in content:
                    concerns.append(pattern)
        
        return list(set(concerns))
    
    def _analyze_communication_style(self, dialogue_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析患者沟通风格"""
        style = {
            'formality': 'informal',
            'question_frequency': len([m for m in dialogue_data if '?' in m.get('content', '')]),
            'message_count': len(dialogue_data)
        }
        return style
    
    def _analyze_emotion(self, dialogue_data: List[Dict[str, Any]]) -> str:
        """分析患者情绪倾向"""
        positive_words = ['谢谢', '开心', '好的', '明白', '放心', '安心', '满意', '舒服']
        negative_words = ['担心', '害怕', '焦虑', '难受', '疼', '痛', '不安', '忧虑', '紧张']
        
        positive_count = 0
        negative_count = 0
        
        for msg in dialogue_data:
            content = msg.get('content', '')
            for word in positive_words:
                if word in content:
                    positive_count += 1
            for word in negative_words:
                if word in content:
                    negative_count += 1
        
        if negative_count > positive_count:
            return 'negative'
        elif positive_count > negative_count:
            return 'positive'
        else:
            return 'neutral'
    
    def state_transition(self, doctor_action: 'DoctorAction') -> Dict[str, Any]:
        """
        根据医生行动更新虚拟患者状态
        
        参数：
            doctor_action: 医生行动对象
        
        返回：
            更新后的患者状态
        """
        # 使用LLM分析医生行动对患者状态的影响
        analysis_result = self._analyze_doctor_action(doctor_action)
        
        # 更新状态
        self.state['current_mood'] = analysis_result.get('mood', self.state['current_mood'])
        self.state['disease_progression'] = analysis_result.get('progression', self.state['disease_progression'])
        self.state['symptoms'] = analysis_result.get('symptoms', self.state['symptoms'])
        
        # 记录交互历史
        self.state['interaction_history'].append({
            'doctor_action': doctor_action.action_type,
            'timestamp': doctor_action.timestamp,
            'state_before': self.state.copy()
        })
        
        # 重新构建系统提示词（状态已更新）
        self._build_system_prompt()
        
        return self.state
    
    def _analyze_doctor_action(self, doctor_action: 'DoctorAction') -> Dict[str, Any]:
        """
        使用LLM分析医生行动对患者状态的影响
        
        参数：
            doctor_action: 医生行动对象
        
        返回：
            分析结果字典
        """
        prompt = f"""
医生说：{doctor_action.content}

请分析这会如何影响患者的状态，输出JSON格式结果：

{{
    "mood": "anxious|worried|neutral|positive",
    "symptoms": ["症状1", "症状2", ...],
    "progression": -1|0|1
}}

说明：
- mood: 患者情绪变化
- symptoms: 可能出现或加重的症状列表
- progression: 疾病进展(-1=好转, 0=不变, 1=恶化)
"""
        
        result = self.llm_client.chat_json("", prompt)
        
        # 验证结果
        if not result:
            result = {
                'mood': self.state['current_mood'],
                'symptoms': self.state['symptoms'],
                'progression': 0
            }
        
        return result
    
    def generate_response(self, doctor_message: str) -> str:
        """
        使用LLM生成患者响应
        
        参数：
            doctor_message: 医生的消息内容
        
        返回：
            患者响应文本
        """
        response = self.llm_client.chat(self.system_prompt, doctor_message)
        
        if not response:
            # 如果LLM调用失败，使用默认响应
            symptom_str = ', '.join(self.state['symptoms'][:3]) if self.state['symptoms'] else '身体不舒服'
            return f"医生，{symptom_str}，想问问该怎么办？"
        
        return response
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取当前患者状态
        
        返回：
            当前患者状态
        """
        return self.state
    
    def get_scenario_type(self) -> str:
        """
        获取当前场景类型
        
        返回：
            当前场景类型名称
        """
        return self.scenario_type
    
    def set_scenario_type(self, scenario_type: str) -> None:
        """
        设置场景类型
        
        参数：
            scenario_type: 场景类型名称
        """
        self.scenario_type = scenario_type
        self.scenario_config = self.scenario_manager.get_scenario_config(scenario_type)
        self._build_system_prompt()
    
    def get_available_scenarios(self) -> List[str]:
        """
        获取所有可用场景类型
        
        返回：
            场景类型列表
        """
        return self.scenario_manager.get_scenarios()
    
    def match_scenario(self, keywords: List[str]) -> Optional[str]:
        """
        根据关键词匹配场景
        
        参数：
            keywords: 关键词列表
        
        返回：
            匹配的场景类型
        """
        for keyword in keywords:
            matched = self.scenario_manager.get_scenarios_by_keyword(keyword)
            if matched:
                return matched[0]
        return None
    
    def reset_state(self) -> None:
        """
        重置患者状态
        """
        self.state['current_mood'] = 'neutral'
        self.state['disease_progression'] = 0
        self.state['adverse_reactions'] = []
        self.state['interaction_history'] = []
        self._build_system_prompt()

class DoctorAction:
    """
    医生行动类
    表示医生在对话中的行动
    """
    def __init__(self, action_type: str, content: str, timestamp: str = None):
        self.action_type = action_type
        self.content = content
        self.timestamp = timestamp
        self.errors = []
    
    def add_error(self, error_type: str, description: str = '') -> None:
        """
        添加错误
        
        参数：
            error_type: 错误类型
            description: 错误描述
        """
        self.errors.append({
            'type': error_type,
            'description': description
        })
    
    def has_error(self, error_type: str) -> bool:
        """
        检查是否存在特定错误
        
        参数：
            error_type: 错误类型
        
        返回：
            是否存在该错误
        """
        return any(e['type'] == error_type for e in self.errors)
    
    def is_correct(self, action_type: str) -> bool:
        """
        检查行动是否正确
        
        参数：
            action_type: 行动类型
        
        返回：
            是否正确
        """
        return self.action_type == action_type and not self.errors
