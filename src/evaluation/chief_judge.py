"""
主审法官
综合各评估意见，输出李克特5维评分及具体扣分理由
使用LLM进行综合评估
"""

from typing import Dict, List, Any
from src.utils.llm_client import LLMClient

class ChiefJudge:
    """
    主审法官类
    使用LLM负责综合各评估意见，输出最终评分
    """
    
    def __init__(self, llm_client: LLMClient, evidence_checker=None, empathy_evaluator=None, kill_switch=None):
        self.llm_client = llm_client
        self.evidence_checker = evidence_checker
        self.empathy_evaluator = empathy_evaluator
        self.kill_switch = kill_switch
        self.scores = {}
        self.dimension_weights = {
            'accuracy': 0.25,
            'effectiveness': 0.25,
            'safety': 0.25,
            'personalization': 0.15,
            'empathy': 0.10
        }
    
    def set_dimension_weights(self, weights: Dict[str, float]) -> None:
        """
        设置各维度权重
        
        参数：
            weights: 维度权重字典
        """
        self.dimension_weights.update(weights)
    
    def evaluate(self, dialogue_history: List[Dict[str, Any]], 
                 patient_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM综合评估对话
        
        参数：
            dialogue_history: 对话历史
            patient_state: 患者状态
            context: 上下文信息
        
        返回：
            综合评估结果
        """
        # 构建对话历史字符串
        dialogue_text = "\n".join([f"{turn['role']}: {turn['content']}" for turn in dialogue_history])
        
        # 构建患者状态字符串
        patient_info = patient_state.get('demographics', {})
        medical_info = patient_state.get('medical_info', {})
        
        patient_text = f"""
患者信息：
- 年龄：{patient_info.get('age', '未知')}
- 性别：{patient_info.get('gender', '未知')}
- 疾病：{medical_info.get('pathology_type', '未知')}
- 分期：{medical_info.get('stage', '未知')}
- 当前情绪：{patient_state.get('current_mood', '未知')}
- 症状：{', '.join(patient_state.get('symptoms', []))}
"""
        
        # 构建LLM提示词
        system_prompt = """
你是一位专业的医患沟通评估专家，请根据对话历史和患者信息，进行综合评估。

评估维度（每项1-5分）：
1. 准确性(accuracy)：医学知识的正确性，事实陈述的准确性
2. 有效性(effectiveness)：治疗建议的有效性，是否帮助患者解决问题
3. 安全性(safety)：是否存在医疗风险，是否符合诊疗规范
4. 个性化(personalization)：是否考虑患者个体差异
5. 共情(empathy)：是否体现人文关怀，沟通是否有温度

请输出JSON格式结果，包含以下字段：
- scores: 各维度分数字典
- deduction_reasons: 扣分理由列表
- overall_score: 综合分数
- is_passed: 是否通过（综合分数>=3为通过）
- risk_report: 风险报告字典，包含 critical_issues、warnings、risk_level
"""
        
        user_prompt = f"""
对话历史：
{dialogue_text}

{patient_text}

请对医生的表现进行综合评估，输出JSON格式结果。
"""
        
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        
        # 验证结果
        if not result or 'scores' not in result:
            # 如果LLM评估失败，使用基于组件的评估
            result = self._fallback_evaluate(dialogue_history, patient_state, context)
        
        # 保存评分
        self.scores = result.get('scores', {})
        
        return result
    
    def _fallback_evaluate(self, dialogue_history: List[Dict[str, Any]], 
                           patient_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        备用评估方法（基于组件的评估）
        
        参数：
            dialogue_history: 对话历史
            patient_state: 患者状态
            context: 上下文信息
        
        返回：
            综合评估结果
        """
        result = {
            'scores': {},
            'deduction_reasons': [],
            'overall_score': 0,
            'is_passed': False,
            'risk_report': {}
        }
        
        # 获取医生的所有响应
        doctor_responses = [
            turn['content'] for turn in dialogue_history 
            if turn.get('role') == 'doctor'
        ]
        
        # 评估准确性
        accuracy_score, accuracy_reasons = self._evaluate_accuracy(doctor_responses, context)
        result['scores']['accuracy'] = accuracy_score
        result['deduction_reasons'].extend(accuracy_reasons)
        
        # 评估有效性
        effectiveness_score, effectiveness_reasons = self._evaluate_effectiveness(doctor_responses, patient_state)
        result['scores']['effectiveness'] = effectiveness_score
        result['deduction_reasons'].extend(effectiveness_reasons)
        
        # 评估安全性
        safety_score, safety_reasons = self._evaluate_safety(doctor_responses, patient_state, context)
        result['scores']['safety'] = safety_score
        result['deduction_reasons'].extend(safety_reasons)
        
        # 评估个性化
        personalization_score, personalization_reasons = self._evaluate_personalization(doctor_responses, patient_state)
        result['scores']['personalization'] = personalization_score
        result['deduction_reasons'].extend(personalization_reasons)
        
        # 评估情感关怀
        empathy_score, empathy_reasons = self._evaluate_empathy(doctor_responses, patient_state)
        result['scores']['empathy'] = empathy_score
        result['deduction_reasons'].extend(empathy_reasons)
        
        # 计算综合评分
        overall_score = self._calculate_overall_score(result['scores'])
        result['overall_score'] = overall_score
        
        # 判断是否通过
        result['is_passed'] = overall_score >= 3
        
        # 生成风险报告
        result['risk_report'] = self._generate_risk_report(result['scores'], result['deduction_reasons'])
        
        return result
    
    def _evaluate_accuracy(self, responses: List[str], context: Dict[str, Any]) -> tuple:
        """评估准确性"""
        score = 5
        reasons = []
        
        if self.evidence_checker:
            for response in responses:
                check_result = self.evidence_checker.check_factuality(response, context)
                if not check_result['is_factual']:
                    score -= len(check_result['violations'])
                    for violation in check_result['violations']:
                        reasons.append(f"准确性扣分: {violation['message']}")
        
        return (max(1, score), reasons)
    
    def _evaluate_effectiveness(self, responses: List[str], patient_state: Dict[str, Any]) -> tuple:
        """评估有效性"""
        score = 5
        reasons = []
        
        has_treatment_plan = any('建议' in r or '应该' in r or '需要' in r for r in responses)
        if not has_treatment_plan:
            score -= 2
            reasons.append("有效性扣分: 未提供明确的治疗建议")
        
        interaction_history = patient_state.get('interaction_history', [])
        if interaction_history:
            initial_mood = interaction_history[0].get('state_before', {}).get('current_mood')
            final_mood = patient_state.get('current_mood')
            
            if initial_mood in ['anxious', 'worried'] and final_mood == 'neutral':
                score = min(5, score + 1)
            elif initial_mood == 'neutral' and final_mood in ['anxious', 'worried']:
                score -= 2
                reasons.append("有效性扣分: 患者情绪状态恶化")
        
        return (max(1, score), reasons)
    
    def _evaluate_safety(self, responses: List[str], patient_state: Dict[str, Any], context: Dict[str, Any]) -> tuple:
        """评估安全性"""
        score = 5
        reasons = []
        
        if self.kill_switch:
            for response in responses:
                if self.kill_switch.check_critical_error(response, context):
                    score = 1
                    reasons.append("安全性扣分: 触发Kill-Switch")
                    return (score, reasons)
        
        adverse_reactions = patient_state.get('adverse_reactions', [])
        if adverse_reactions:
            score -= len(adverse_reactions)
            reasons.append(f"安全性扣分: 出现不良反应")
        
        return (max(1, score), reasons)
    
    def _evaluate_personalization(self, responses: List[str], patient_state: Dict[str, Any]) -> tuple:
        """评估个性化"""
        score = 5
        reasons = []
        
        patient_info = patient_state.get('demographics', {})
        patient_age = patient_info.get('age')
        
        if patient_age:
            age_mentioned = any(str(patient_age) in r or '年龄' in r for r in responses)
            if not age_mentioned:
                score -= 1
                reasons.append("个性化扣分: 未考虑患者个体差异")
        
        return (max(1, score), reasons)
    
    def _evaluate_empathy(self, responses: List[str], patient_state: Dict[str, Any]) -> tuple:
        """评估情感关怀"""
        score = 3
        reasons = []
        
        if self.empathy_evaluator:
            all_responses = ' '.join(responses)
            result = self.empathy_evaluator.evaluate_communication_quality(all_responses)
            score = result['overall_score']
            
            if score < 3:
                feedback = self.empathy_evaluator.get_feedback(all_responses, patient_state.get('demographics'))
                reasons.extend([f"情感关怀扣分: {f}" for f in feedback])
        
        return (score, reasons)
    
    def _calculate_overall_score(self, scores: Dict[str, int]) -> int:
        """计算综合评分"""
        total = 0.0
        for dimension, weight in self.dimension_weights.items():
            total += scores.get(dimension, 3) * weight
        return round(total)
    
    def _generate_risk_report(self, scores: Dict[str, int], reasons: List[str]) -> Dict[str, Any]:
        """生成风险报告"""
        critical_issues = []
        warnings = []
        
        for reason in reasons:
            if 'Kill-Switch' in reason or '红线' in reason:
                critical_issues.append(reason)
            else:
                warnings.append(reason)
        
        return {
            'critical_issues': critical_issues,
            'warnings': warnings,
            'score_summary': scores,
            'risk_level': 'high' if critical_issues else ('medium' if warnings else 'low')
        }
    
    def get_scores(self) -> Dict[str, int]:
        """获取最近一次评分"""
        return self.scores
