"""
效度分析器
负责分析评估框架的内容效度和判别效度
"""

import json
import os
from typing import Dict, List, Any

class ValidityAnalyzer:
    """
    效度分析器类
    负责分析评估框架的内容效度和判别效度
    """
    
    def __init__(self, guidelines_dir: str = "dataset/guidlines/formated_data_llm_optimized"):
        self.guidelines = {}
        self.guidelines_dir = guidelines_dir
        
        # 自动加载所有指南
        self._load_all_guidelines()
    
    def _load_all_guidelines(self):
        """
        自动加载指南目录中的所有JSON文件
        """
        if not os.path.exists(self.guidelines_dir):
            print(f"警告: 指南目录不存在: {self.guidelines_dir}")
            return
        
        json_files = [f for f in os.listdir(self.guidelines_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            guideline_name = json_file.replace('.json', '')
            file_path = os.path.join(self.guidelines_dir, json_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.guidelines[guideline_name] = data
                print(f"已加载指南: {guideline_name}")
            except Exception as e:
                print(f"加载指南失败 {guideline_name}: {e}")
    
    def set_guidelines(self, guidelines: Dict[str, Any]) -> None:
        """
        设置指南数据
        
        参数：
            guidelines: 指南数据
        """
        self.guidelines = guidelines
    
    def search_guideline_content(self, keyword: str) -> List[Dict[str, Any]]:
        """
        在所有指南中搜索包含关键字的章节
        
        参数：
            keyword: 关键字
        
        返回：
            匹配的章节列表
        """
        results = []
        
        for guideline_name, guideline in self.guidelines.items():
            sections = guideline.get('sections', [])
            
            for section in sections:
                section_name = section.get('section_name', '')
                content = section.get('content', '')
                
                if keyword in section_name or keyword in content:
                    results.append({
                        'guideline_name': guideline_name,
                        'section_name': section_name,
                        'content': content
                    })
        
        return results
    
    def analyze_content_validity(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析内容效度（指南覆盖度分析）
        
        参数：
            test_cases: 测试用例列表
        
        返回：
            内容效度分析结果
        """
        coverage = {
            'total_cases': len(test_cases),
            'covered_cases': 0,
            'coverage_ratio': 0.0,
            'guidelines_used': [],
            'case_coverage': []
        }
        
        for case in test_cases:
            case_coverage = self._analyze_case_coverage(case)
            coverage['case_coverage'].append(case_coverage)
            
            if case_coverage['is_covered']:
                coverage['covered_cases'] += 1
                
                for guideline in case_coverage['matched_guidelines']:
                    if guideline not in coverage['guidelines_used']:
                        coverage['guidelines_used'].append(guideline)
        
        if coverage['total_cases'] > 0:
            coverage['coverage_ratio'] = coverage['covered_cases'] / coverage['total_cases']
        
        return coverage
    
    def _analyze_case_coverage(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析单个测试用例的指南覆盖度
        
        参数：
            case: 测试用例
        
        返回：
            覆盖度分析结果
        """
        result = {
            'case_id': case.get('case_id', ''),
            'disease': case.get('disease', ''),
            'is_covered': False,
            'matched_guidelines': [],
            'matched_items': [],
            'coverage_ratio': 0.0
        }
        
        disease = case.get('disease')
        
        if disease and self.guidelines:
            for guideline_name, guideline in self.guidelines.items():
                title = guideline.get('title', '')
                sections = guideline.get('sections', [])
                
                # 检查指南标题或章节是否包含疾病名称
                if disease in title or any(disease in section.get('section_name', '') for section in sections):
                    result['is_covered'] = True
                    result['matched_guidelines'].append(guideline_name)
                    
                    # 提取匹配的章节
                    for section in sections:
                        if disease in section.get('section_name', '') or disease in section.get('content', ''):
                            result['matched_items'].append(section.get('section_name', ''))
        
        if result['matched_items']:
            result['coverage_ratio'] = 1.0
        
        return result
    
    def analyze_discriminant_validity(self, model_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        分析判别效度（模型梯度敏感度测试）
        
        参数：
            model_results: 各模型的测试结果，格式为 {model_name: [results]}
        
        返回：
            判别效度分析结果
        """
        result = {
            'models': [],
            'overall_scores': {},
            'discrimination_metric': 0.0,
            'model_ranking': [],
            'sensitivity_analysis': []
        }
        
        if not model_results:
            return result
        
        # 计算各模型的平均分数
        model_scores = {}
        for model_name, results in model_results.items():
            if results:
                avg_score = sum(r.get('overall_score', 0) for r in results) / len(results)
                model_scores[model_name] = avg_score
        
        # 按分数排序
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        result['model_ranking'] = [(model, score) for model, score in sorted_models]
        result['overall_scores'] = model_scores
        
        # 计算判别度指标（最高分与最低分的差异）
        scores = list(model_scores.values())
        if len(scores) >= 2:
            max_score = max(scores)
            min_score = min(scores)
            
            # 判别度 = (最高分 - 最低分) / 最高分
            if max_score > 0:
                result['discrimination_metric'] = (max_score - min_score) / max_score
            
            # 分析敏感度
            for i, (model_name, score) in enumerate(sorted_models):
                sensitivity_info = {
                    'model_name': model_name,
                    'score': score,
                    'rank': i + 1,
                    'relative_performance': score / max_score if max_score > 0 else 0
                }
                result['sensitivity_analysis'].append(sensitivity_info)
        
        result['models'] = list(model_scores.keys())
        
        return result
    
    def generate_validity_report(self, test_cases: List[Dict[str, Any]], 
                                model_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        生成完整的效度报告
        
        参数：
            test_cases: 测试用例列表
            model_results: 各模型的测试结果
        
        返回：
            效度报告
        """
        content_validity = self.analyze_content_validity(test_cases)
        discriminant_validity = self.analyze_discriminant_validity(model_results)
        
        return {
            'content_validity': content_validity,
            'discriminant_validity': discriminant_validity,
            'summary': self._generate_summary(content_validity, discriminant_validity)
        }
    
    def _generate_summary(self, content_validity: Dict[str, Any], 
                          discriminant_validity: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成摘要
        
        参数：
            content_validity: 内容效度分析结果
            discriminant_validity: 判别效度分析结果
        
        返回：
            摘要
        """
        summary = {
            'content_validity_level': self._get_validity_level(content_validity['coverage_ratio']),
            'discriminant_validity_level': self._get_validity_level(discriminant_validity['discrimination_metric']),
            'is_valid': False
        }
        
        # 判断整体效度是否合格
        content_ok = content_validity['coverage_ratio'] >= 0.7
        discriminant_ok = discriminant_validity['discrimination_metric'] >= 0.3
        
        summary['is_valid'] = content_ok and discriminant_ok
        
        return summary
    
    def _get_validity_level(self, metric: float) -> str:
        """
        获取效度等级
        
        参数：
            metric: 效度指标值
        
        返回：
            效度等级
        """
        if metric >= 0.8:
            return '优秀'
        elif metric >= 0.6:
            return '良好'
        elif metric >= 0.4:
            return '中等'
        else:
            return '差'