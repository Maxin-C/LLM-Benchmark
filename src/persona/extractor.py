import re
from typing import List, Dict, Set, Optional
from collections import defaultdict
from ..data_processing.cleaner import clean_content, is_medical_staff
from ..data_processing.parser import (
    SymptomExtractor,
    TreatmentStageExtractor,
    AgeExtractor,
    QuestionIntentClassifier
)


class PatientPersonaExtractor:
    """患者Persona提取器"""
    
    def __init__(self):
        self.symptom_extractor = SymptomExtractor()
        self.stage_extractor = TreatmentStageExtractor()
        self.age_extractor = AgeExtractor()
        self.intent_classifier = QuestionIntentClassifier()
    
    def extract_persona(self, messages: List[Dict]) -> Dict:
        """
        从患者消息中提取Persona
        """
        # 按患者分组消息
        patient_messages = defaultdict(list)
        for msg in messages:
            if not is_medical_staff(msg['sender']):
                patient_messages[msg['sender']].append(msg)
        
        personas = []
        
        for patient_name, patient_msgs in patient_messages.items():
            persona = self._extract_single_persona(patient_name, patient_msgs)
            personas.append(persona)
        
        return personas
    
    def _extract_single_persona(self, patient_name: str, messages: List[Dict]) -> Dict:
        """
        提取单个患者的Persona
        """
        persona = {
            'patient_id': patient_name,
            'name': patient_name,
            'age': None,
            'treatment_stage': None,
            'symptoms': set(),
            'concerns': set(),
            'question_intents': set(),
            'message_count': len(messages),
            'first_message_index': min(m['message_index'] for m in messages),
            'last_message_index': max(m['message_index'] for m in messages),
            'group': messages[0]['group'] if messages else ''
        }
        
        for msg in messages:
            content = clean_content(msg['content'])
            if not content:
                continue
            
            # 提取年龄（优先使用第一个提到的年龄）
            if persona['age'] is None:
                age = self.age_extractor.extract(content)
                if age:
                    persona['age'] = age
            
            # 提取治疗阶段（优先使用第一个提到的阶段）
            if persona['treatment_stage'] is None:
                stage = self.stage_extractor.extract(content)
                if stage:
                    persona['treatment_stage'] = stage
            
            # 提取症状
            symptoms = self.symptom_extractor.extract(content)
            persona['symptoms'].update(symptoms)
            
            # 提取问题意图
            intents = self.intent_classifier.classify(content)
            persona['question_intents'].update(intents)
            
            # 提取关注点（简单提取问题关键词）
            concerns = self._extract_concerns(content)
            persona['concerns'].update(concerns)
        
        # 转换为列表
        persona['symptoms'] = list(persona['symptoms'])
        persona['concerns'] = list(persona['concerns'])
        persona['question_intents'] = list(persona['question_intents'])
        
        return persona
    
    def _extract_concerns(self, content: str) -> Set[str]:
        """
        提取患者关注点
        """
        concern_keywords = [
            '能不能', '可以吗', '怎么办', '如何', '多久', '什么时候',
            '注意', '需要', '应该', '推荐', '建议', '影响', '危险'
        ]
        
        concerns = set()
        for keyword in concern_keywords:
            if keyword in content:
                concerns.add(keyword)
        
        # 提取问题中的核心名词
        question_patterns = [
            r'(什么是|什么叫|何为)(\w+)',
            r'(能不能|可以不可以)(\w+)',
            r'(需要)(\w+)',
        ]
        
        for pattern in question_patterns:
            match = re.search(pattern, content)
            if match:
                concerns.add(match.group(2))
        
        return concerns

    def analyze_conversation_patterns(self, messages: List[Dict]) -> Dict:
        """
        分析对话模式，找出常见问题和互动模式
        """
        pattern_analysis = {
            'common_symptoms': defaultdict(int),
            'common_concerns': defaultdict(int),
            'patient_question_frequency': defaultdict(int),
            'staff_response_rate': 0.0
        }
        
        patient_count = 0
        staff_response_count = 0
        total_messages = len(messages)
        
        for msg in messages:
            content = clean_content(msg['content'])
            if not content:
                continue
            
            if is_medical_staff(msg['sender']):
                staff_response_count += 1
            else:
                patient_count += 1
                # 统计患者问题
                intents = self.intent_classifier.classify(content)
                for intent in intents:
                    pattern_analysis['patient_question_frequency'][intent] += 1
                
                # 统计症状
                symptoms = self.symptom_extractor.extract(content)
                for symptom in symptoms:
                    pattern_analysis['common_symptoms'][symptom] += 1
                
                # 统计关注点
                concerns = self._extract_concerns(content)
                for concern in concerns:
                    pattern_analysis['common_concerns'][concern] += 1
        
        # 计算医护人员响应率
        if patient_count > 0:
            pattern_analysis['staff_response_rate'] = staff_response_count / (patient_count + staff_response_count)
        
        # 转换为排序后的列表
        pattern_analysis['common_symptoms'] = sorted(
            pattern_analysis['common_symptoms'].items(),
            key=lambda x: x[1], reverse=True
        )
        pattern_analysis['common_concerns'] = sorted(
            pattern_analysis['common_concerns'].items(),
            key=lambda x: x[1], reverse=True
        )
        pattern_analysis['patient_question_frequency'] = sorted(
            pattern_analysis['patient_question_frequency'].items(),
            key=lambda x: x[1], reverse=True
        )
        
        return pattern_analysis
