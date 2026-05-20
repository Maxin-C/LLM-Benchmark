"""
虚拟患者Agent
整合EHR数据、对话数据和患者画像图谱，构建具有真实行为特征的虚拟患者
"""

import random
from typing import Dict, List, Any

class VirtualPatientAgent:
    def __init__(self, ehr_data: Dict[str, Any], dialogue_data: List[Dict[str, Any]], persona_graph: Dict[str, Any]):
        """
        初始化虚拟患者Agent
        
        参数：
            ehr_data: EHR病历数据
            dialogue_data: 对话历史数据
            persona_graph: 患者画像图谱
        """
        self.state = self._initialize_state(ehr_data, dialogue_data, persona_graph)
    
    def _initialize_state(self, ehr_data: Dict[str, Any], dialogue_data: List[Dict[str, Any]], persona_graph: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def _extract_symptoms(self, dialogue_data: List[Dict[str, Any]]) -> List[str]:
        """从对话中提取症状实体"""
        symptoms = []
        symptom_keywords = ['痛', '胀', '肿', '麻', '痒', '恶心', '呕吐', '头晕', '乏力', '发热']
        
        for msg in dialogue_data:
            content = msg.get('content', '')
            for keyword in symptom_keywords:
                if keyword in content:
                    symptoms.append(keyword)
        
        return list(set(symptoms))
    
    def _extract_concerns(self, dialogue_data: List[Dict[str, Any]]) -> List[str]:
        """从对话中提取患者关注点"""
        concerns = []
        concern_keywords = ['怎么办', '需要', '应该', '注意', '能不能', '多久', '会好吗', '要紧吗']
        
        for msg in dialogue_data:
            content = msg.get('content', '')
            for keyword in concern_keywords:
                if keyword in content:
                    concerns.append(keyword)
        
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
        positive_words = ['谢谢', '开心', '好的', '明白', '放心']
        negative_words = ['担心', '害怕', '焦虑', '难受', '疼', '痛']
        
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
        # 药物错误处理
        if doctor_action.has_error('wrong_medication'):
            self.state['adverse_reactions'].append('药物不良反应')
            self.state['current_mood'] = 'anxious'
            if '恶心' not in self.state['symptoms']:
                self.state['symptoms'].append('恶心')
        
        # 随访遗漏处理
        if doctor_action.has_error('missed_followup'):
            self.state['disease_progression'] += 1
            self.state['current_mood'] = 'worried'
        
        # 正确治疗建议处理
        if doctor_action.is_correct('treatment_plan'):
            self.state['disease_progression'] = max(0, self.state['disease_progression'] - 1)
            if self.state['current_mood'] in ['anxious', 'worried']:
                self.state['current_mood'] = 'neutral'
        
        # 依从性影响
        if self.state['compliance'] == 'low':
            if random.random() < 0.3:
                self.state['adverse_reactions'].append('用药不规律')
        
        # 记录交互历史
        self.state['interaction_history'].append({
            'doctor_action': doctor_action.action_type,
            'timestamp': doctor_action.timestamp,
            'state_before': self.state.copy()
        })
        
        return self.state
    
    def generate_response(self, doctor_message: str) -> str:
        """
        根据医生消息生成患者响应
        
        参数：
            doctor_message: 医生的消息内容
        
        返回：
            患者响应文本
        """
        mood = self.state['current_mood']
        symptoms = self.state['symptoms']
        concerns = self.state['concerns']
        
        # 基于状态的响应模板选择
        if mood == 'anxious':
            templates = [
                "医生，我现在感觉{symptoms}，有点担心，{concern}？",
                "我很焦虑，{symptoms}的症状让我很不安，{concern}？",
                "怎么办啊医生，{symptoms}，我好担心{concern}..."
            ]
        elif mood == 'worried':
            templates = [
                "医生，我有点担心，{symptoms}，不知道{concern}？",
                "最近{symptoms}，心里不太踏实，{concern}？",
                "医生你看，{symptoms}，我担心{concern}..."
            ]
        else:
            templates = [
                "医生，{symptoms}，想问问{concern}？",
                "你好医生，我{symptoms}，想了解一下{concern}。",
                "医生你好，最近{symptoms}，{concern}？"
            ]
        
        # 选择随机模板并填充
        template = random.choice(templates)
        
        symptom_str = ', '.join(symptoms[:3]) if symptoms else '身体不舒服'
        concern_str = concerns[0] if concerns else '治疗效果'
        
        return template.format(symptoms=symptom_str, concern=concern_str)
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取当前患者状态
        
        返回：
            当前患者状态
        """
        return self.state
    
    def reset_state(self) -> None:
        """
        重置患者状态
        """
        self.state['current_mood'] = 'neutral'
        self.state['disease_progression'] = 0
        self.state['adverse_reactions'] = []
        self.state['interaction_history'] = []

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
