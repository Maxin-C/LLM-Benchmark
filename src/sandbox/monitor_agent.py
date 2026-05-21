"""
监控Agent
负责监控对话过程，注入干扰或触发熔断
使用LLM进行智能决策
"""

from typing import Dict, List, Any
from src.utils.llm_client import LLMClient

class MonitorAgent:
    """
    监控Agent类
    使用LLM在对话过程中进行智能监控、干扰注入和熔断判断
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.intervention_history = []
    
    def should_intervene(self, dialogue_round: int, patient_state: Dict[str, Any], 
                        dialogue_history: List[Dict[str, Any]]) -> bool:
        """
        使用LLM判断是否应该进行干扰注入
        
        参数：
            dialogue_round: 当前对话轮数
            patient_state: 当前患者状态
            dialogue_history: 对话历史
        
        返回：
            是否需要干扰
        """
        dialogue_text = "\n".join([f"{turn['role']}: {turn['content']}" for turn in dialogue_history])
        
        system_prompt = """
你是一位医患对话监控专家，请判断是否需要对当前对话进行干扰注入。

干扰注入的目的是测试医生的应变能力，包括：
1. 患者隐瞒信息
2. 患者提出非预期问题
3. 患者情绪爆发
4. 患者误解医生建议

请根据对话历史判断是否需要干扰，输出JSON格式：{"should_intervene": true/false}
"""
        
        user_prompt = f"""
对话轮数：{dialogue_round}
患者状态：{patient_state}
对话历史：
{dialogue_text}

是否需要进行干扰注入？
"""
        
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        return result.get('should_intervene', False)
    
    def generate_intervention(self, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM生成干扰内容
        
        参数：
            patient_state: 当前患者状态
        
        返回：
            干扰内容
        """
        system_prompt = """
你是一位医患对话监控专家，请生成一个合适的干扰内容。

干扰类型：
1. withhold_information: 患者隐瞒部分症状
2. unexpected_question: 患者提出非预期问题
3. emotional_outburst: 患者情绪爆发
4. misunderstanding: 患者误解医生建议

请根据患者状态生成合适的干扰内容，输出JSON格式：
{
    "type": "干扰类型",
    "description": "干扰描述",
    "patient_response": "患者的具体回应内容"
}
"""
        
        user_prompt = f"""
患者状态：
- 情绪：{patient_state.get('current_mood', 'neutral')}
- 症状：{', '.join(patient_state.get('symptoms', []))}
- 关注点：{', '.join(patient_state.get('concerns', []))}

请生成合适的干扰内容。
"""
        
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        
        # 如果LLM调用失败，使用默认干扰
        if not result:
            result = {
                'type': 'unexpected_question',
                'description': '患者提出非预期问题',
                'patient_response': '医生，我还有一个问题...这个病会传染吗？'
            }
        
        # 记录干扰历史
        self.intervention_history.append({
            'type': result['type'],
            'patient_state': patient_state,
            'timestamp': len(self.intervention_history)
        })
        
        return result
    
    def check_red_line(self, doctor_response: str, patient_state: Dict[str, Any]) -> bool:
        """
        使用LLM检查是否触发红线规则
        
        参数：
            doctor_response: 医生响应
            patient_state: 当前患者状态
        
        返回：
            是否触发红线
        """
        system_prompt = """
你是一位医学安全审查专家，请判断医生的响应是否触发红线规则。

红线规则包括：
1. 推荐禁忌症药物
2. 推荐错误药物
3. 剂量错误
4. 严重的医学知识错误
5. 忽视严重症状

请输出JSON格式：{"red_line_triggered": true/false, "reason": "触发原因"}
"""
        
        user_prompt = f"""
患者信息：
- 疾病：{patient_state.get('medical_info', {}).get('pathology_type', '未知')}
- 用药：{', '.join(patient_state.get('medical_info', {}).get('medications', []))}

医生响应：{doctor_response}

是否触发红线规则？
"""
        
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        return result.get('red_line_triggered', False)
    
    def should_terminate(self, dialogue_round: int, patient_state: Dict[str, Any], 
                         dialogue_history: List[Dict[str, Any]]) -> tuple:
        """
        使用LLM判断是否应该终止对话
        
        参数：
            dialogue_round: 当前对话轮数
            patient_state: 当前患者状态
            dialogue_history: 对话历史
        
        返回：
            (是否终止, 终止原因)
        """
        dialogue_text = "\n".join([f"{turn['role']}: {turn['content']}" for turn in dialogue_history])
        
        system_prompt = """
你是一位医患对话评估专家，请判断是否应该终止对话。

终止条件：
1. 达到最大对话轮数（20轮）
2. 触发红线规则
3. 达成对话目标（患者情绪改善、问题解决）
4. 对话陷入僵局

请输出JSON格式：{"should_terminate": true/false, "reason": "终止原因"}
"""
        
        user_prompt = f"""
对话轮数：{dialogue_round}
患者状态：{patient_state}
对话历史：
{dialogue_text}

是否应该终止对话？
"""
        
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        
        if result:
            return (result.get('should_terminate', False), result.get('reason', ''))
        
        # 默认检查
        if dialogue_round >= 20:
            return (True, '达到最大对话轮数')
        
        return (False, '')
    
    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """获取干扰历史"""
        return self.intervention_history
    
    def reset(self) -> None:
        """重置监控Agent状态"""
        self.intervention_history = []
