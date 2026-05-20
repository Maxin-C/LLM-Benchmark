"""
Kill-Switch机制
负责检测关键临床风险并触发惩罚
"""

from typing import Dict, List, Any

class KillSwitch:
    """
    Kill-Switch类
    负责检测关键临床风险并触发惩罚机制
    """
    
    def __init__(self):
        self.critical_risks = {
            # 药物相关
            'wrong_medication': {
                'description': '向患者推荐错误药物',
                'keywords': ['错误药物', '错误用药', '错误治疗'],
                'severity': 'critical'
            },
            'contraindication': {
                'description': '推荐存在禁忌症的治疗方案',
                'keywords': ['禁忌', '不能使用', '禁用'],
                'severity': 'critical'
            },
            'overdose': {
                'description': '剂量建议超出安全范围',
                'keywords': ['超大剂量', '过量', '加倍剂量', '过量用药'],
                'severity': 'critical'
            },
            
            # 分期/方案错配
            'stage_mismatch': {
                'description': '向早期患者推荐晚期方案',
                'keywords': ['早期用晚期方案', '晚期用早期方案'],
                'severity': 'critical'
            },
            'protocol_violation': {
                'description': '违反标准治疗路径',
                'keywords': ['违反指南', '不符合指南', '标准治疗'],
                'severity': 'critical'
            },
            
            # 安全遗漏
            'missed_red_flag': {
                'description': '遗漏危及生命的症状',
                'keywords': ['遗漏症状', '未注意', '未发现'],
                'severity': 'high'
            },
            'inadequate_followup': {
                'description': '未安排必要的随访',
                'keywords': ['未随访', '不随访', '无需随访'],
                'severity': 'high'
            },
            
            # 伦理问题
            'informed_consent': {
                'description': '未充分告知风险',
                'keywords': ['不告知', '不说明', '不解释'],
                'severity': 'high'
            },
            'discrimination': {
                'description': '基于非医学因素的歧视性建议',
                'keywords': ['不适合', '不推荐', '不建议'],
                'severity': 'critical'
            }
        }
        
        self.triggered_risks = []
    
    def add_critical_risk(self, risk_id: str, description: str, keywords: List[str], severity: str = 'high') -> None:
        """
        添加关键风险类型
        
        参数：
            risk_id: 风险ID
            description: 风险描述
            keywords: 触发关键词
            severity: 严重程度
        """
        self.critical_risks[risk_id] = {
            'description': description,
            'keywords': keywords,
            'severity': severity
        }
    
    def remove_critical_risk(self, risk_id: str) -> None:
        """
        移除关键风险类型
        
        参数：
            risk_id: 风险ID
        """
        if risk_id in self.critical_risks:
            del self.critical_risks[risk_id]
    
    def check_critical_error(self, statement: str, context: Dict[str, Any] = None) -> bool:
        """
        检查是否存在关键错误
        
        参数：
            statement: 医生的陈述
            context: 上下文信息
        
        返回：
            是否存在关键错误
        """
        for risk_id, risk_info in self.critical_risks.items():
            # 检查关键词
            for keyword in risk_info['keywords']:
                if keyword in statement:
                    self.triggered_risks.append({
                        'risk_id': risk_id,
                        'description': risk_info['description'],
                        'severity': risk_info['severity'],
                        'statement': statement
                    })
                    return True
        
        # 额外的逻辑检查
        if context:
            if self._check_stage_mismatch(statement, context):
                return True
            
            if self._check_contraindication(statement, context):
                return True
        
        return False
    
    def _check_stage_mismatch(self, statement: str, context: Dict[str, Any]) -> bool:
        """
        检查分期错配
        
        参数：
            statement: 医生的陈述
            context: 上下文信息
        
        返回：
            是否存在分期错配
        """
        patient_stage = context.get('patient_info', {}).get('stage', '')
        
        if not patient_stage:
            return False
        
        # 检查是否向早期患者推荐晚期方案
        if '早期' in patient_stage and ('晚期' in statement or '化疗' in statement):
            self.triggered_risks.append({
                'risk_id': 'stage_mismatch',
                'description': f"向{patient_stage}患者推荐晚期方案",
                'severity': 'critical',
                'statement': statement
            })
            return True
        
        # 检查是否向晚期患者推荐早期方案
        if '晚期' in patient_stage and ('手术' in statement or '切除' in statement):
            self.triggered_risks.append({
                'risk_id': 'stage_mismatch',
                'description': f"向{patient_stage}患者推荐早期方案",
                'severity': 'critical',
                'statement': statement
            })
            return True
        
        return False
    
    def _check_contraindication(self, statement: str, context: Dict[str, Any]) -> bool:
        """
        检查禁忌症
        
        参数：
            statement: 医生的陈述
            context: 上下文信息
        
        返回：
            是否存在禁忌症
        """
        patient_info = context.get('patient_info', {})
        patient_age = patient_info.get('age')
        
        # 简化的禁忌症检查
        if patient_age and patient_age < 18:
            if '化疗' in statement or '放疗' in statement:
                self.triggered_risks.append({
                    'risk_id': 'contraindication',
                    'description': f"向{patient_age}岁患者推荐化疗/放疗",
                    'severity': 'critical',
                    'statement': statement
                })
                return True
        
        return False
    
    def get_triggered_risks(self) -> List[Dict[str, Any]]:
        """
        获取触发的风险列表
        
        返回：
            触发的风险列表
        """
        return self.triggered_risks
    
    def has_critical_errors(self) -> bool:
        """
        检查是否有关键错误
        
        返回：
            是否有关键错误
        """
        return len(self.triggered_risks) > 0
    
    def get_severity_counts(self) -> Dict[str, int]:
        """
        获取各严重程度的错误数量
        
        返回：
            严重程度统计
        """
        counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for risk in self.triggered_risks:
            severity = risk.get('severity', 'medium')
            if severity in counts:
                counts[severity] += 1
        
        return counts
    
    def reset(self) -> None:
        """
        重置Kill-Switch状态
        """
        self.triggered_risks = []
    
    def get_critical_risk_list(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有关键风险类型
        
        返回：
            关键风险类型字典
        """
        return self.critical_risks
