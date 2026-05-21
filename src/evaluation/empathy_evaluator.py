"""
人文关怀员
负责共情与沟通易懂度评估
使用LLM进行深度语义分析
"""

from typing import Dict, List, Any
from src.utils.llm_client import LLMClient

class EmpathyEvaluator:
    """
    人文关怀员类
    使用LLM评估医生的共情能力和沟通质量
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def evaluate_empathy(self, response: str) -> Dict[str, Any]:
        """
        使用LLM评估共情表达
        
        参数：
            response: 医生的响应
        
        返回：
            共情评估结果
        """
        system_prompt = """
你是一位专业的医患沟通评估专家，请评估医生回复中体现的共情水平。

共情评估维度（1-5分）：
1. 情感认同：是否认可和理解患者的感受
2. 支持表达：是否提供情感支持和安慰
3. 鼓励程度：是否给予鼓励和积极反馈
4. 个性化回应：是否针对患者具体情况回应

请输出JSON格式结果，包含以下字段：
- empathy_score: 综合共情分数 (1-5)
- empathy_level: 共情等级 ('high'/'medium'/'low')
- emotional_recognition: 情感认同分数 (1-5)
- support_expression: 支持表达分数 (1-5)
- encouragement: 鼓励程度分数 (1-5)
- explanation: 评估理由
"""
        
        user_prompt = f"""
医生回复：{response}

请评估医生回复的共情水平，输出JSON格式结果。
"""
        
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        
        # 验证结果
        if not result or 'empathy_score' not in result:
            result = {
                'empathy_score': 3,
                'empathy_level': 'medium',
                'emotional_recognition': 3,
                'support_expression': 3,
                'encouragement': 3,
                'explanation': 'LLM评估失败，使用默认值'
            }
        
        return result
    
    def evaluate_clarity(self, response: str, patient_education_level: str = 'average') -> Dict[str, Any]:
        """
        使用LLM评估沟通易懂度
        
        参数：
            response: 医生的响应
            patient_education_level: 患者教育水平 (low/average/high)
        
        返回：
            易懂度评估结果
        """
        system_prompt = """
你是一位专业的医患沟通评估专家，请评估医生回复的易懂程度。

评估标准：
1. 专业术语使用是否适当
2. 解释是否清晰易懂
3. 是否考虑患者的理解能力
4. 是否需要进一步解释

请输出JSON格式结果，包含以下字段：
- clarity_score: 易懂度分数 (1-5)
- jargon_count: 专业术语数量
- jargon_list: 专业术语列表
- recommendations: 改进建议列表
- explanation: 评估理由
"""
        
        user_prompt = f"""
医生回复：{response}
患者教育水平：{patient_education_level}

请评估医生回复的易懂程度，输出JSON格式结果。
"""
        
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        
        # 验证结果
        if not result or 'clarity_score' not in result:
            result = {
                'clarity_score': 3,
                'jargon_count': 0,
                'jargon_list': [],
                'recommendations': [],
                'explanation': 'LLM评估失败，使用默认值'
            }
        
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
            'overall_score': max(1, min(5, overall_score)),
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
        if empathy_result.get('empathy_level') == 'low' or empathy_result.get('empathy_score', 3) < 3:
            feedback.append("建议增加共情表达，认可和理解患者的感受")
        
        clarity_result = self.evaluate_clarity(response, patient_info.get('education_level', 'average') if patient_info else 'average')
        feedback.extend(clarity_result.get('recommendations', []))
        
        return feedback
