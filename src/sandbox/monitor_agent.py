"""
监控Agent
负责监控对话过程，注入干扰或触发熔断
"""

import random
from typing import Dict, List, Any, Callable

class MonitorAgent:
    """
    监控Agent类
    负责在对话过程中进行监控、干扰注入和熔断判断
    """
    
    def __init__(self):
        self.intervention_probability = 0.3
        self.max_dialogue_rounds = 20
        self.red_line_rules = []
        self.intervention_history = []
    
    def set_intervention_probability(self, probability: float) -> None:
        """
        设置干扰注入概率
        
        参数：
            probability: 干扰概率 (0-1)
        """
        self.intervention_probability = probability
    
    def set_max_rounds(self, rounds: int) -> None:
        """
        设置最大对话轮数
        
        参数：
            rounds: 最大轮数
        """
        self.max_dialogue_rounds = rounds
    
    def add_red_line_rule(self, rule_name: str, condition: Callable, action: Callable) -> None:
        """
        添加红线规则
        
        参数：
            rule_name: 规则名称
            condition: 条件函数
            action: 触发后的动作函数
        """
        self.red_line_rules.append({
            'name': rule_name,
            'condition': condition,
            'action': action
        })
    
    def should_intervene(self, dialogue_round: int, patient_state: Dict[str, Any]) -> bool:
        """
        判断是否应该进行干扰注入
        
        参数：
            dialogue_round: 当前对话轮数
            patient_state: 当前患者状态
        
        返回：
            是否需要干扰
        """
        # 根据概率决定是否干扰
        if random.random() < self.intervention_probability:
            return True
        
        return False
    
    def generate_intervention(self, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成干扰内容
        
        参数：
            patient_state: 当前患者状态
        
        返回：
            干扰内容
        """
        interventions = [
            {
                'type': 'withhold_information',
                'description': '患者隐瞒部分症状',
                'effect': '患者未提及某些重要症状'
            },
            {
                'type': 'unexpected_question',
                'description': '患者提出非预期问题',
                'effect': '患者询问与当前话题无关的问题'
            },
            {
                'type': 'emotional_outburst',
                'description': '患者情绪爆发',
                'effect': '患者表达强烈的负面情绪'
            },
            {
                'type': 'misunderstanding',
                'description': '患者误解医生建议',
                'effect': '患者错误理解医生的话'
            }
        ]
        
        intervention = random.choice(interventions)
        
        # 记录干扰历史
        self.intervention_history.append({
            'type': intervention['type'],
            'patient_state': patient_state,
            'timestamp': len(self.intervention_history)
        })
        
        return intervention
    
    def check_red_line(self, doctor_response: str, patient_state: Dict[str, Any]) -> bool:
        """
        检查是否触发红线规则
        
        参数：
            doctor_response: 医生响应
            patient_state: 当前患者状态
        
        返回：
            是否触发红线
        """
        for rule in self.red_line_rules:
            if rule['condition'](doctor_response, patient_state):
                rule['action'](doctor_response, patient_state)
                return True
        
        return False
    
    def should_terminate(self, dialogue_round: int, patient_state: Dict[str, Any], 
                         doctor_response: str = None) -> tuple:
        """
        判断是否应该终止对话
        
        参数：
            dialogue_round: 当前对话轮数
            patient_state: 当前患者状态
            doctor_response: 医生响应
        
        返回：
            (是否终止, 终止原因)
        """
        # 检查是否达到最大轮数
        if dialogue_round >= self.max_dialogue_rounds:
            return (True, '达到最大对话轮数')
        
        # 检查是否触发红线
        if doctor_response and self.check_red_line(doctor_response, patient_state):
            return (True, '触发红线规则')
        
        # 检查是否达成目标
        if self._check_goal_achieved(patient_state):
            return (True, '达成对话目标')
        
        return (False, '')
    
    def _check_goal_achieved(self, patient_state: Dict[str, Any]) -> bool:
        """
        检查是否达成对话目标
        
        参数：
            patient_state: 当前患者状态
        
        返回：
            是否达成目标
        """
        # 如果患者情绪从焦虑/担忧恢复到中性，视为达成目标
        if patient_state.get('current_mood') == 'neutral':
            if 'anxious' in [h.get('state_before', {}).get('current_mood') 
                            for h in patient_state.get('interaction_history', [])]:
                return True
        
        # 如果疾病进展得到控制
        if patient_state.get('disease_progression', 0) == 0:
            if any(h.get('state_before', {}).get('disease_progression', 0) > 0 
                   for h in patient_state.get('interaction_history', [])):
                return True
        
        return False
    
    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """
        获取干扰历史
        
        返回：
            干扰历史列表
        """
        return self.intervention_history
    
    def reset(self) -> None:
        """
        重置监控Agent状态
        """
        self.intervention_history = []

class ClinicalMonitorAgent(MonitorAgent):
    """
    临床监控Agent
    包含针对临床场景的特定监控规则
    """
    
    def __init__(self):
        super().__init__()
        self._initialize_clinical_rules()
    
    def _initialize_clinical_rules(self) -> None:
        """
        初始化临床监控规则
        """
        # 药物错误红线
        self.add_red_line_rule(
            'wrong_medication',
            self._check_wrong_medication,
            self._handle_wrong_medication
        )
        
        # 禁忌症红线
        self.add_red_line_rule(
            'contraindication',
            self._check_contraindication,
            self._handle_contraindication
        )
        
        # 剂量错误红线
        self.add_red_line_rule(
            'wrong_dosage',
            self._check_wrong_dosage,
            self._handle_wrong_dosage
        )
    
    def _check_wrong_medication(self, doctor_response: str, patient_state: Dict[str, Any]) -> bool:
        """
        检查是否推荐错误药物
        
        参数：
            doctor_response: 医生响应
            patient_state: 患者状态
        
        返回：
            是否推荐错误药物
        """
        # 简化的检查逻辑
        wrong_drugs = ['阿司匹林', '青霉素']  # 示例：这些药物可能不适合乳腺癌患者
        patient_medications = patient_state.get('medical_info', {}).get('medications', [])
        
        for drug in wrong_drugs:
            if drug in doctor_response and drug not in patient_medications:
                return True
        
        return False
    
    def _handle_wrong_medication(self, doctor_response: str, patient_state: Dict[str, Any]) -> None:
        """
        处理推荐错误药物
        """
        print(f"[红线触发] 检测到错误药物推荐: {doctor_response}")
    
    def _check_contraindication(self, doctor_response: str, patient_state: Dict[str, Any]) -> bool:
        """
        检查是否推荐存在禁忌症的药物
        
        参数：
            doctor_response: 医生响应
            patient_state: 患者状态
        
        返回：
            是否存在禁忌症
        """
        # 简化的检查逻辑
        contraindicated_drugs = {
            '孕妇': ['化疗药物'],
            '肝肾功能不全': ['某些靶向药']
        }
        
        patient_condition = patient_state.get('medical_info', {}).get('condition', '')
        
        for condition, drugs in contraindicated_drugs.items():
            if condition in patient_condition:
                for drug in drugs:
                    if drug in doctor_response:
                        return True
        
        return False
    
    def _handle_contraindication(self, doctor_response: str, patient_state: Dict[str, Any]) -> None:
        """
        处理禁忌症
        """
        print(f"[红线触发] 检测到禁忌症药物推荐: {doctor_response}")
    
    def _check_wrong_dosage(self, doctor_response: str, patient_state: Dict[str, Any]) -> bool:
        """
        检查剂量是否错误
        
        参数：
            doctor_response: 医生响应
            patient_state: 患者状态
        
        返回：
            剂量是否错误
        """
        # 简化的检查逻辑：检查是否有明显不合理的剂量描述
        high_dosage_keywords = ['超大剂量', '过量', '加倍']
        for keyword in high_dosage_keywords:
            if keyword in doctor_response:
                return True
        
        return False
    
    def _handle_wrong_dosage(self, doctor_response: str, patient_state: Dict[str, Any]) -> None:
        """
        处理剂量错误
        """
        print(f"[红线触发] 检测到剂量错误: {doctor_response}")
