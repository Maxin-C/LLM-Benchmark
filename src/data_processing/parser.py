import re
from typing import List, Dict, Optional, Tuple


class SymptomExtractor:
    """症状提取器"""
    
    SYMPTOM_PATTERNS = [
        # 疼痛相关
        (r'(痛|疼)', '疼痛'),
        (r'(刺痛)', '刺痛'),
        (r'(胀痛)', '胀痛'),
        (r'(头痛)', '头痛'),
        (r'(恶心|想吐)', '恶心'),
        (r'(呕吐)', '呕吐'),
        (r'(脱发)', '脱发'),
        (r'(乏力|疲劳)', '乏力'),
        # 治疗相关
        (r'(化疗)', '化疗'),
        (r'(手术)', '手术'),
        (r'(拔管|管子)', '引流管'),
        (r'(升白针)', '升白针'),
        (r'(病理报告)', '病理报告'),
        # 身体部位
        (r'(胸|乳房)', '胸部'),
        (r'(伤口)', '伤口'),
        (r'(手臂|胳膊)', '手臂'),
    ]
    
    def extract(self, content: str) -> List[str]:
        symptoms = []
        for pattern, label in self.SYMPTOM_PATTERNS:
            if re.search(pattern, content):
                symptoms.append(label)
        return symptoms


class TreatmentStageExtractor:
    """治疗阶段提取器"""
    
    STAGE_PATTERNS = [
        (r'(刚手术|术后|手术完)', '术后康复期'),
        (r'(化疗中|第一次化疗|化疗完)', '化疗阶段'),
        (r'(放疗)', '放疗阶段'),
        (r'(复查|复诊)', '随访阶段'),
    ]
    
    def extract(self, content: str) -> Optional[str]:
        for pattern, stage in self.STAGE_PATTERNS:
            if re.search(pattern, content):
                return stage
        return None


class AgeExtractor:
    """年龄提取器"""
    
    def extract(self, content: str) -> Optional[int]:
        # 匹配年龄模式：XX岁、年龄XX
        match = re.search(r'(?<!\d)(\d{2})(?:岁|年龄)(?!\d)', content)
        if match:
            return int(match.group(1))
        return None


class QuestionIntentClassifier:
    """问题意图分类器"""
    
    INTENT_KEYWORDS = {
        'symptom': ['痛', '疼', '难受', '不舒服', '症状'],
        'diet': ['吃', '喝', '饮食', '能吃', '不能吃'],
        'treatment': ['化疗', '放疗', '手术', '拔管', '升白针'],
        'report': ['报告', '病理', '检查', '结果'],
        'schedule': ['什么时候', '几号', '时间', '安排'],
        'side_effect': ['副作用', '反应', '掉头发', '呕吐'],
    }
    
    def classify(self, content: str) -> List[str]:
        intents = []
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(keyword in content for keyword in keywords):
                intents.append(intent)
        return intents if intents else ['other']
