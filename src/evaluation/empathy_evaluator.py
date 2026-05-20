"""
人文关怀员
负责共情与沟通易懂度评估
"""

from typing import Dict, List, Any

class EmpathyEvaluator:
    """
    人文关怀员类
    负责评估医生的共情能力和沟通质量
    """
    
    def __init__(self):
        self.empathy_keywords = {
            'acknowledgement': ['理解', '明白', '知道', '了解', '感受到'],
            'support': ['支持', '陪伴', '帮助', '一起', '加油'],
            'validation': ['正常', '可以理解', '没关系', '不是你的错'],
            'encouragement': ['坚持', '相信', '一定', '会好的', '慢慢来']
        }
        
        self.jargon_keywords = [
            '根治术', '化疗', '放疗', '靶向治疗', '内分泌治疗',
            '浸润性', '转移性', '复发', '进展', '生存期'
        ]
    
    def evaluate_empathy(self, response: str) -> Dict[str, Any]:
        """
        评估共情表达
        
        参数：
            response: 医生的响应
        
        返回：
            共情评估结果
        """
        result = {
            'empathy_score': 0,
            'empathy_level': 'none',
            'empathy_elements': [],
            'confidence': 0.0
        }
        
        score = 0
        elements = []
        categories_found = 0
        
        # 检查各类共情关键词
        for category, keywords in self.empathy_keywords.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in response:
                    found_keywords.append(keyword)
            
            if found_keywords:
                score += len(found_keywords)
                categories_found += 1
                elements.append({
                    'category': category,
                    'keywords': found_keywords
                })
        
        # 计算共情分数：基于找到的类别数量和关键词数量
        # 基础分：每个类别得1分，最多4分
        category_score = min(categories_found, 4)
        # 额外分：每找到一个关键词加0.5分，最多1分
        keyword_bonus = min(score * 0.2, 1)
        
        result['empathy_score'] = round(category_score + keyword_bonus)
        
        # 确保分数在1-5范围内
        if result['empathy_score'] == 0:
            result['empathy_score'] = 1
        
        # 确定共情等级
        if result['empathy_score'] >= 4:
            result['empathy_level'] = 'high'
        elif result['empathy_score'] >= 3:
            result['empathy_level'] = 'medium'
        else:
            result['empathy_level'] = 'low'
        
        result['empathy_elements'] = elements
        result['confidence'] = min(1.0, (categories_found + score) / 8)
        
        return result
    
    def evaluate_clarity(self, response: str, patient_education_level: str = 'average') -> Dict[str, Any]:
        """
        评估沟通易懂度
        
        参数：
            response: 医生的响应
            patient_education_level: 患者教育水平 (low/average/high)
        
        返回：
            易懂度评估结果
        """
        result = {
            'clarity_score': 0,
            'jargon_count': 0,
            'jargon_list': [],
            'recommendations': []
        }
        
        # 统计专业术语数量
        jargon_count = 0
        jargon_list = []
        
        for jargon in self.jargon_keywords:
            if jargon in response:
                jargon_count += 1
                jargon_list.append(jargon)
        
        result['jargon_count'] = jargon_count
        result['jargon_list'] = jargon_list
        
        # 根据患者教育水平调整评分
        base_penalty = jargon_count * 0.5
        
        if patient_education_level == 'low':
            base_penalty *= 1.5
        elif patient_education_level == 'high':
            base_penalty *= 0.7
        
        # 计算易懂度分数
        clarity_score = max(1, 5 - base_penalty)
        result['clarity_score'] = round(clarity_score)
        
        # 生成建议
        if jargon_count > 3:
            result['recommendations'].append("建议减少专业术语使用，用更通俗的语言解释")
        
        if '化疗' in jargon_list and patient_education_level != 'high':
            result['recommendations'].append("建议解释'化疗'的含义和过程")
        
        return result
    
    def evaluate_communication_quality(self, response: str, patient_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        综合评估沟通质量
        
        参数：
            response: 医生的响应
            patient_info: 患者信息
        
        返回：
            沟通质量评估结果
        """
        empathy_result = self.evaluate_empathy(response)
        
        education_level = patient_info.get('education_level', 'average') if patient_info else 'average'
        clarity_result = self.evaluate_clarity(response, education_level)
        
        # 综合评分
        overall_score = round((empathy_result['empathy_score'] * 0.6 + clarity_result['clarity_score'] * 0.4))
        
        return {
            'overall_score': overall_score,
            'empathy': empathy_result,
            'clarity': clarity_result,
            'breakdown': {
                'empathy_weight': 0.6,
                'clarity_weight': 0.4
            }
        }
    
    def get_feedback(self, response: str, patient_info: Dict[str, Any] = None) -> List[str]:
        """
        获取改进建议
        
        参数：
            response: 医生的响应
            patient_info: 患者信息
        
        返回：
            改进建议列表
        """
        feedback = []
        
        empathy_result = self.evaluate_empathy(response)
        if empathy_result['empathy_level'] == 'low':
            feedback.append("建议增加共情表达，如使用'我理解你的感受'等语句")
        
        clarity_result = self.evaluate_clarity(response, patient_info.get('education_level', 'average') if patient_info else 'average')
        feedback.extend(clarity_result.get('recommendations', []))
        
        return feedback
