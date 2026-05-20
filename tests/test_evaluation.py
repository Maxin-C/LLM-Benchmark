"""
评估引擎测试
"""

import pytest
from src.evaluation import EvidenceChecker, EmpathyEvaluator, ChiefJudge, KillSwitch

class TestEvidenceChecker:
    """测试循证审查官"""
    
    def test_factuality_check(self):
        """测试事实核查"""
        checker = EvidenceChecker()
        
        statement = '患者需要进行化疗'
        context = {'disease_name': '乳腺癌', 'patient_info': {'stage': 'IIB期'}}
        
        result = checker.check_factuality(statement, context)
        
        assert result is not None
        assert 'is_factual' in result
    
    def test_red_line_check(self):
        """测试红线检查"""
        checker = EvidenceChecker()
        
        # 正常陈述不应触发红线
        result = checker.check_red_line('建议定期复查', {})
        assert result is False

class TestEmpathyEvaluator:
    """测试人文关怀员"""
    
    def test_empathy_evaluation(self):
        """测试共情评估"""
        evaluator = EmpathyEvaluator()
        
        response = '我非常理解你的感受，会一直支持你，相信你一定可以战胜疾病'
        result = evaluator.evaluate_empathy(response)
        
        assert result['empathy_score'] >= 4
        assert result['empathy_level'] == 'high'
    
    def test_clarity_evaluation(self):
        """测试易懂度评估"""
        evaluator = EmpathyEvaluator()
        
        response = '你需要进行化疗和放疗'
        result = evaluator.evaluate_clarity(response, 'low')
        
        assert result is not None
        assert 'clarity_score' in result
    
    def test_communication_quality(self):
        """测试沟通质量评估"""
        evaluator = EmpathyEvaluator()
        
        response = '我理解你的担忧，你需要进行化疗'
        result = evaluator.evaluate_communication_quality(response)
        
        assert 'overall_score' in result
        assert isinstance(result['overall_score'], int)

class TestChiefJudge:
    """测试主审法官"""
    
    def test_evaluation(self):
        """测试综合评估"""
        judge = ChiefJudge()
        
        dialogue_history = [
            {'role': 'doctor', 'content': '你好，有什么可以帮你的？'},
            {'role': 'patient', 'content': '医生，我最近感觉不舒服'},
            {'role': 'doctor', 'content': '请详细描述一下你的症状'}
        ]
        
        patient_state = {
            'demographics': {'age': 45},
            'medical_info': {'stage': 'IIB期'},
            'current_mood': 'neutral',
            'interaction_history': []
        }
        
        context = {'disease_name': '乳腺癌'}
        
        result = judge.evaluate(dialogue_history, patient_state, context)
        
        assert 'scores' in result
        assert 'overall_score' in result
        assert 'is_passed' in result
    
    def test_score_calculation(self):
        """测试分数计算"""
        judge = ChiefJudge()
        
        scores = {
            'accuracy': 4,
            'effectiveness': 4,
            'safety': 5,
            'personalization': 3,
            'empathy': 4
        }
        
        overall = judge._calculate_overall_score(scores)
        
        assert isinstance(overall, int)
        assert 1 <= overall <= 5

class TestKillSwitch:
    """测试Kill-Switch"""
    
    def test_critical_error_check(self):
        """测试关键错误检查"""
        kill_switch = KillSwitch()
        
        # 正常建议不应触发Kill-Switch
        result = kill_switch.check_critical_error('建议定期复查', {})
        assert result is False
        
        # 错误药物建议应触发Kill-Switch
        result = kill_switch.check_critical_error('推荐错误药物', {})
        assert result is True
    
    def test_severity_counts(self):
        """测试严重程度统计"""
        kill_switch = KillSwitch()
        kill_switch.check_critical_error('推荐错误药物', {})
        
        counts = kill_switch.get_severity_counts()
        
        assert counts['critical'] >= 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
