"""
文本处理工具
"""

import re
from typing import List, Dict, Any

class TextUtils:
    """
    文本处理工具类
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        清洗文本
        
        参数：
            text: 原始文本
        
        返回：
            清洗后的文本
        """
        # 去除多余空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fa5，。！？、；：“”‘’（）]', '', text)
        
        return text
    
    @staticmethod
    def extract_symptoms(text: str) -> List[str]:
        """
        从文本中提取症状
        
        参数：
            text: 文本
        
        返回：
            症状列表
        """
        symptom_keywords = [
            '痛', '胀', '肿', '麻', '痒', '恶心', '呕吐', '头晕', '乏力', '发热',
            '咳嗽', '胸闷', '气短', '腹泻', '便秘', '失眠', '焦虑', '抑郁',
            '疼痛', '酸痛', '胀痛', '刺痛', '隐痛', '剧痛'
        ]
        
        symptoms = []
        for keyword in symptom_keywords:
            if keyword in text:
                symptoms.append(keyword)
        
        return list(set(symptoms))
    
    @staticmethod
    def extract_drugs(text: str) -> List[str]:
        """
        从文本中提取药物
        
        参数：
            text: 文本
        
        返回：
            药物列表
        """
        drug_keywords = [
            '他莫昔芬', '来曲唑', '阿那曲唑', '依西美坦', '紫杉醇', '多西他赛',
            '卡培他滨', '吉西他滨', '顺铂', '卡铂', '赫赛汀', '帕妥珠单抗',
            '化疗', '靶向', '内分泌', '免疫'
        ]
        
        drugs = []
        for keyword in drug_keywords:
            if keyword in text:
                drugs.append(keyword)
        
        return list(set(drugs))
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        参数：
            text1: 文本1
            text2: 文本2
        
        返回：
            相似度分数
        """
        # 简单的Jaccard相似度
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 500) -> str:
        """
        截断文本
        
        参数：
            text: 原始文本
            max_length: 最大长度
        
        返回：
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length] + '...'
    
    @staticmethod
    def parse_json_safe(text: str) -> Dict[str, Any]:
        """
        安全解析JSON
        
        参数：
            text: JSON字符串
        
        返回：
            解析后的字典
        """
        try:
            import json
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """
        移除HTML标签
        
        参数：
            text: 包含HTML的文本
        
        返回：
            清理后的文本
        """
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        规范化空白字符
        
        参数：
            text: 原始文本
        
        返回：
            规范化后的文本
        """
        return ' '.join(text.split())
