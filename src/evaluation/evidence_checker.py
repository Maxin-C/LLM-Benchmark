"""
循证审查官
负责事实核查与红线拦截
"""

from typing import Dict, List, Any

class EvidenceChecker:
    """
    循证审查官类
    负责对照指南知识图谱进行事实核查与红线拦截
    """
    
    def __init__(self, knowledge_graph=None, guidelines=None):
        self.knowledge_graph = knowledge_graph
        self.guidelines = guidelines
        self.violations = []
    
    def check_factuality(self, statement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查陈述的事实正确性
        
        参数：
            statement: 医生的陈述
            context: 上下文信息（患者信息、场景等）
        
        返回：
            检查结果
        """
        result = {
            'is_factual': True,
            'violations': [],
            'supporting_evidence': [],
            'confidence': 1.0
        }
        
        # 检查药物信息
        drug_check = self._check_drug_information(statement, context)
        result['violations'].extend(drug_check.get('violations', []))
        result['supporting_evidence'].extend(drug_check.get('supporting_evidence', []))
        
        # 检查治疗方案
        treatment_check = self._check_treatment_plan(statement, context)
        result['violations'].extend(treatment_check.get('violations', []))
        result['supporting_evidence'].extend(treatment_check.get('supporting_evidence', []))
        
        # 检查疾病信息
        disease_check = self._check_disease_information(statement, context)
        result['violations'].extend(disease_check.get('violations', []))
        result['supporting_evidence'].extend(disease_check.get('supporting_evidence', []))
        
        result['is_factual'] = len(result['violations']) == 0
        
        if result['violations']:
            result['confidence'] = max(0.0, 1.0 - len(result['violations']) * 0.2)
        
        # 记录违规
        self.violations.extend(result['violations'])
        
        return result
    
    def _check_drug_information(self, statement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查药物信息
        
        参数：
            statement: 医生的陈述
            context: 上下文信息
        
        返回：
            检查结果
        """
        result = {'violations': [], 'supporting_evidence': []}
        
        # 检查是否推荐了禁忌症药物
        if self.guidelines:
            patient_info = context.get('patient_info', {})
            contraindicated = self.guidelines.check_contraindication(statement, patient_info)
            if contraindicated:
                result['violations'].append({
                    'type': 'contraindication',
                    'message': f"检测到禁忌症药物推荐: {statement}"
                })
        
        return result
    
    def _check_treatment_plan(self, statement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查治疗方案
        
        参数：
            statement: 医生的陈述
            context: 上下文信息
        
        返回：
            检查结果
        """
        result = {'violations': [], 'supporting_evidence': []}
        
        disease_name = context.get('disease_name', '')
        
        if self.guidelines and disease_name:
            treatment_plan = {
                'drugs': self._extract_drugs(statement),
                'patient_info': context.get('patient_info', {})
            }
            
            validation = self.guidelines.validate_treatment_plan(disease_name, treatment_plan)
            
            if not validation['is_valid']:
                for error in validation.get('errors', []):
                    result['violations'].append({
                        'type': 'treatment_violation',
                        'message': error
                    })
                
                for warning in validation.get('warnings', []):
                    result['supporting_evidence'].append({
                        'type': 'warning',
                        'message': warning
                    })
        
        return result
    
    def _check_disease_information(self, statement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查疾病信息
        
        参数：
            statement: 医生的陈述
            context: 上下文信息
        
        返回：
            检查结果
        """
        result = {'violations': [], 'supporting_evidence': []}
        
        # 检查分期错误
        correct_stage = context.get('patient_info', {}).get('stage', '')
        if correct_stage:
            if '早期' in statement and '晚期' in correct_stage:
                result['violations'].append({
                    'type': 'stage_error',
                    'message': f"分期描述错误: 患者实际为{correct_stage}，但陈述中提到早期"
                })
            elif '晚期' in statement and '早期' in correct_stage:
                result['violations'].append({
                    'type': 'stage_error',
                    'message': f"分期描述错误: 患者实际为{correct_stage}，但陈述中提到晚期"
                })
        
        return result
    
    def _extract_drugs(self, statement: str) -> List[Dict[str, Any]]:
        """
        从陈述中提取药物信息
        
        参数：
            statement: 医生的陈述
        
        返回：
            药物列表
        """
        # 简化的药物提取逻辑
        drug_keywords = ['他莫昔芬', '来曲唑', '紫杉醇', '化疗', '靶向']
        drugs = []
        
        for keyword in drug_keywords:
            if keyword in statement:
                drugs.append({'name': keyword, 'dosage': None})
        
        return drugs
    
    def check_red_line(self, statement: str, context: Dict[str, Any]) -> bool:
        """
        检查是否触发红线
        
        参数：
            statement: 医生的陈述
            context: 上下文信息
        
        返回：
            是否触发红线
        """
        result = self.check_factuality(statement, context)
        
        # 如果存在严重违规，触发红线
        for violation in result['violations']:
            if violation['type'] in ['contraindication', 'stage_error']:
                return True
        
        return False
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """
        获取所有违规记录
        
        返回：
            违规记录列表
        """
        return self.violations
    
    def reset(self) -> None:
        """
        重置审查官状态
        """
        self.violations = []
