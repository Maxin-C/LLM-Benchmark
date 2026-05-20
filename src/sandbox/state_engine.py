"""
状态引擎
管理虚拟患者的状态转移逻辑
"""

from typing import Dict, Any, Callable
from datetime import datetime

class StateEngine:
    """
    状态引擎类
    负责管理虚拟患者的状态转移
    """
    
    def __init__(self):
        self.transition_rules = {}
        self.state_history = []
    
    def add_transition_rule(self, trigger: str, condition: Callable, action: Callable) -> None:
        """
        添加状态转移规则
        
        参数：
            trigger: 触发器名称
            condition: 条件函数，返回布尔值
            action: 动作函数，执行状态转移
        """
        if trigger not in self.transition_rules:
            self.transition_rules[trigger] = []
        
        self.transition_rules[trigger].append({
            'condition': condition,
            'action': action
        })
    
    def apply_rules(self, trigger: str, state: Dict[str, Any], doctor_action: Any = None) -> Dict[str, Any]:
        """
        应用状态转移规则
        
        参数：
            trigger: 触发器名称
            state: 当前状态
            doctor_action: 医生行动（可选）
        
        返回：
            更新后的状态
        """
        if trigger not in self.transition_rules:
            return state
        
        new_state = state.copy()
        
        for rule in self.transition_rules[trigger]:
            if rule['condition'](state, doctor_action):
                new_state = rule['action'](new_state, doctor_action)
        
        # 记录状态历史
        self.state_history.append({
            'timestamp': datetime.now().isoformat(),
            'trigger': trigger,
            'state_before': state,
            'state_after': new_state
        })
        
        return new_state
    
    def get_state_history(self) -> list:
        """
        获取状态历史
        
        返回：
            状态历史列表
        """
        return self.state_history
    
    def clear_history(self) -> None:
        """
        清除状态历史
        """
        self.state_history = []
    
    def get_transition_rules(self) -> Dict[str, list]:
        """
        获取所有转移规则
        
        返回：
            转移规则字典
        """
        return self.transition_rules

class BreastCancerStateEngine(StateEngine):
    """
    乳腺癌专用状态引擎
    包含针对乳腺癌患者的特定状态转移规则
    """
    
    def __init__(self):
        super().__init__()
        self._initialize_breast_cancer_rules()
    
    def _initialize_breast_cancer_rules(self) -> None:
        """
        初始化乳腺癌状态转移规则
        """
        # 药物错误规则
        self.add_transition_rule(
            'wrong_medication',
            lambda state, action: action and action.has_error('wrong_medication'),
            self._handle_wrong_medication
        )
        
        # 随访遗漏规则
        self.add_transition_rule(
            'missed_followup',
            lambda state, action: action and action.has_error('missed_followup'),
            self._handle_missed_followup
        )
        
        # 正确治疗规则
        self.add_transition_rule(
            'correct_treatment',
            lambda state, action: action and action.is_correct('treatment_plan'),
            self._handle_correct_treatment
        )
        
        # 低依从性规则
        self.add_transition_rule(
            'low_compliance',
            lambda state, action: state.get('compliance') == 'low',
            self._handle_low_compliance
        )
        
        # 情绪变化规则
        self.add_transition_rule(
            'symptom_worsening',
            lambda state, action: len(state.get('symptoms', [])) > 3,
            self._handle_symptom_worsening
        )
    
    def _handle_wrong_medication(self, state: Dict[str, Any], action: Any) -> Dict[str, Any]:
        """
        处理药物错误
        """
        new_state = state.copy()
        new_state['adverse_reactions'] = new_state.get('adverse_reactions', []) + ['药物不良反应']
        new_state['current_mood'] = 'anxious'
        if '恶心' not in new_state.get('symptoms', []):
            new_state['symptoms'] = new_state.get('symptoms', []) + ['恶心']
        return new_state
    
    def _handle_missed_followup(self, state: Dict[str, Any], action: Any) -> Dict[str, Any]:
        """
        处理随访遗漏
        """
        new_state = state.copy()
        new_state['disease_progression'] = new_state.get('disease_progression', 0) + 1
        new_state['current_mood'] = 'worried'
        return new_state
    
    def _handle_correct_treatment(self, state: Dict[str, Any], action: Any) -> Dict[str, Any]:
        """
        处理正确治疗
        """
        new_state = state.copy()
        new_state['disease_progression'] = max(0, new_state.get('disease_progression', 0) - 1)
        if new_state.get('current_mood') in ['anxious', 'worried']:
            new_state['current_mood'] = 'neutral'
        return new_state
    
    def _handle_low_compliance(self, state: Dict[str, Any], action: Any) -> Dict[str, Any]:
        """
        处理低依从性
        """
        import random
        new_state = state.copy()
        
        if random.random() < 0.3:
            new_state['adverse_reactions'] = new_state.get('adverse_reactions', []) + ['用药不规律']
        
        return new_state
    
    def _handle_symptom_worsening(self, state: Dict[str, Any], action: Any) -> Dict[str, Any]:
        """
        处理症状恶化
        """
        new_state = state.copy()
        new_state['current_mood'] = 'anxious'
        return new_state
