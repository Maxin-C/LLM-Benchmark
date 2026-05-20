"""
元评估模块测试
"""

import pytest
import numpy as np
from src.meta_evaluation import ICCCalculator, ValidityAnalyzer, SensitivityTester

class TestICCCalculator:
    """测试ICC计算器"""
    
    def test_add_rater(self):
        """测试添加评分者"""
        icc_calculator = ICCCalculator()
        
        # 添加医生评分
        doctor_ratings = [[4, 3, 5, 4, 3], [4, 4, 5, 3, 4]]
        icc_calculator.add_rater('doctor_1', doctor_ratings)
        
        assert len(icc_calculator.raters) == 1
    
    def test_calculate_icc(self):
        """测试ICC计算"""
        icc_calculator = ICCCalculator()
        
        # 添加更多模拟评分数据（至少10个病例以获得稳定的ICC值）
        doctor_1_ratings = [
            [4, 3, 5, 4, 3], [5, 4, 5, 4, 4], [3, 3, 4, 3, 3],
            [4, 4, 5, 5, 4], [5, 5, 4, 4, 5], [3, 4, 3, 4, 3],
            [4, 5, 4, 5, 4], [5, 3, 5, 3, 5], [4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5]
        ]
        doctor_2_ratings = [
            [4, 4, 5, 3, 4], [5, 4, 5, 4, 5], [3, 3, 4, 3, 2],
            [4, 5, 5, 5, 4], [5, 5, 4, 4, 5], [3, 4, 3, 4, 3],
            [4, 5, 4, 5, 4], [5, 3, 5, 3, 5], [4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5]
        ]
        judge_ratings = [
            [4, 3, 5, 4, 3], [5, 4, 5, 4, 4], [3, 3, 4, 3, 3],
            [4, 4, 5, 5, 4], [5, 5, 4, 4, 5], [3, 4, 3, 4, 3],
            [4, 5, 4, 5, 4], [5, 3, 5, 3, 5], [4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5]
        ]
        
        icc_calculator.add_rater('doctor_1', doctor_1_ratings)
        icc_calculator.add_rater('doctor_2', doctor_2_ratings)
        icc_calculator.add_rater('ease_judge', judge_ratings)
        
        result = icc_calculator.calculate_icc()
        
        assert 'overall_icc' in result
        assert 'dimension_iccs' in result
        # ICC值通常在0-1之间，允许少量负值（数据不足时可能出现）
        assert -1 <= result['overall_icc']['icc'] <= 1
    
    def test_consistency_level(self):
        """测试一致性等级"""
        icc_calculator = ICCCalculator()
        
        assert icc_calculator.get_consistency_level(0.85) == '优秀'
        assert icc_calculator.get_consistency_level(0.70) == '良好'
        assert icc_calculator.get_consistency_level(0.50) == '中等'
        assert icc_calculator.get_consistency_level(0.30) == '差'

class TestValidityAnalyzer:
    """测试效度分析器"""
    
    def test_content_validity(self):
        """测试内容效度分析"""
        analyzer = ValidityAnalyzer()
        
        test_cases = [
            {'case_id': 'case_001', 'disease': '乳腺癌'},
            {'case_id': 'case_002', 'disease': '肺癌'}
        ]
        
        result = analyzer.analyze_content_validity(test_cases)
        
        assert 'total_cases' in result
        assert 'coverage_ratio' in result
    
    def test_discriminant_validity(self):
        """测试判别效度分析"""
        analyzer = ValidityAnalyzer()
        
        model_results = {
            'ModelA': [{'overall_score': 4}, {'overall_score': 5}, {'overall_score': 4}],
            'ModelB': [{'overall_score': 3}, {'overall_score': 3}, {'overall_score': 4}],
            'ModelC': [{'overall_score': 2}, {'overall_score': 2}, {'overall_score': 3}]
        }
        
        result = analyzer.analyze_discriminant_validity(model_results)
        
        assert 'models' in result
        assert 'discrimination_metric' in result
        assert result['discrimination_metric'] > 0

class TestSensitivityTester:
    """测试敏感度测试器"""
    
    def test_run_sensitivity_test(self):
        """测试运行敏感度测试"""
        tester = SensitivityTester()
        
        tester.add_model_results('ModelA', [
            {'overall_score': 4, 'details': {}},
            {'overall_score': 5, 'details': {}},
            {'overall_score': 4, 'details': {}}
        ])
        
        tester.add_model_results('ModelB', [
            {'overall_score': 3, 'details': {}},
            {'overall_score': 3, 'details': {}},
            {'overall_score': 4, 'details': {}}
        ])
        
        result = tester.run_sensitivity_test()
        
        assert 'model_comparison' in result
        assert 'discriminant_power' in result
        assert 'gradient_analysis' in result
    
    def test_discriminant_power(self):
        """测试判别力计算"""
        tester = SensitivityTester()
        
        tester.add_model_results('HighTier', [{'overall_score': 5}]*5)
        tester.add_model_results('LowTier', [{'overall_score': 2}]*5)
        
        result = tester.run_sensitivity_test()
        
        assert result['discriminant_power'] > 0.5

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
