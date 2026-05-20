"""
敏感度测试器
负责测试评估框架对不同模型的区分能力
"""

from typing import Dict, List, Any

class SensitivityTester:
    """
    敏感度测试器类
    负责测试评估框架对不同模型的区分能力
    """
    
    def __init__(self):
        self.model_results = {}
    
    def add_model_results(self, model_name: str, results: List[Dict[str, Any]]) -> None:
        """
        添加模型测试结果
        
        参数：
            model_name: 模型名称
            results: 测试结果列表
        """
        self.model_results[model_name] = results
    
    def run_sensitivity_test(self) -> Dict[str, Any]:
        """
        运行敏感度测试
        
        返回：
            敏感度测试结果
        """
        result = {
            'model_comparison': [],
            'score_distribution': {},
            'gradient_analysis': {},
            'discriminant_power': 0.0,
            'recommendations': []
        }
        
        if not self.model_results:
            return result
        
        # 计算各模型的统计信息
        model_stats = {}
        for model_name, results in self.model_results.items():
            scores = [r.get('overall_score', 0) for r in results]
            
            if scores:
                model_stats[model_name] = {
                    'mean': sum(scores) / len(scores),
                    'std': self._calculate_std(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        # 按平均分排序
        sorted_stats = sorted(model_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        # 构建模型比较表
        for model_name, stats in sorted_stats:
            result['model_comparison'].append({
                'model_name': model_name,
                'mean_score': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'rank': len(result['model_comparison']) + 1
            })
        
        # 计算分数分布
        result['score_distribution'] = self._compute_score_distribution()
        
        # 进行梯度分析
        result['gradient_analysis'] = self._analyze_gradient(sorted_stats)
        
        # 计算判别力
        result['discriminant_power'] = self._compute_discriminant_power(model_stats)
        
        # 生成建议
        result['recommendations'] = self._generate_recommendations(sorted_stats)
        
        return result
    
    def _calculate_std(self, scores: List[float]) -> float:
        """
        计算标准差
        
        参数：
            scores: 分数列表
        
        返回：
            标准差
        """
        if len(scores) == 0:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        
        return variance ** 0.5
    
    def _compute_score_distribution(self) -> Dict[str, Any]:
        """
        计算分数分布
        
        返回：
            分数分布
        """
        distribution = {
            'score_bins': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'total_samples': 0
        }
        
        for results in self.model_results.values():
            for r in results:
                score = r.get('overall_score', 3)
                if 1 <= score <= 5:
                    distribution['score_bins'][score] += 1
                    distribution['total_samples'] += 1
        
        # 计算百分比
        if distribution['total_samples'] > 0:
            distribution['score_percentages'] = {
                k: v / distribution['total_samples'] * 100 
                for k, v in distribution['score_bins'].items()
            }
        
        return distribution
    
    def _analyze_gradient(self, sorted_stats: List[tuple]) -> Dict[str, Any]:
        """
        分析模型梯度
        
        参数：
            sorted_stats: 排序后的模型统计信息
        
        返回：
            梯度分析结果
        """
        gradient = {
            'levels': [],
            'gaps': []
        }
        
        if len(sorted_stats) < 2:
            return gradient
        
        # 定义等级
        levels = ['S', 'A', 'B', 'C']
        num_models = len(sorted_stats)
        level_size = max(1, num_models // 4)
        
        for i, level in enumerate(levels):
            start_idx = i * level_size
            end_idx = min((i + 1) * level_size, num_models)
            
            if start_idx < num_models:
                level_models = sorted_stats[start_idx:end_idx]
                gradient['levels'].append({
                    'level': level,
                    'models': [m[0] for m in level_models],
                    'avg_score': sum(m[1]['mean'] for m in level_models) / len(level_models)
                })
        
        # 计算等级间的差距
        for i in range(len(gradient['levels']) - 1):
            current_level = gradient['levels'][i]
            next_level = gradient['levels'][i + 1]
            
            gap = current_level['avg_score'] - next_level['avg_score']
            gradient['gaps'].append({
                'between': f"{current_level['level']}-{next_level['level']}",
                'gap': gap
            })
        
        return gradient
    
    def _compute_discriminant_power(self, model_stats: Dict[str, Dict[str, float]]) -> float:
        """
        计算判别力
        
        参数：
            model_stats: 模型统计信息
        
        返回：
            判别力值
        """
        if len(model_stats) < 2:
            return 0.0
        
        means = [stats['mean'] for stats in model_stats.values()]
        max_mean = max(means)
        min_mean = min(means)
        avg_std = sum(stats['std'] for stats in model_stats.values()) / len(model_stats)
        
        # 判别力 = (最高分 - 最低分) / (平均标准差 + 1)
        if avg_std >= 0:
            discriminant_power = (max_mean - min_mean) / (avg_std + 1)
        else:
            discriminant_power = max_mean - min_mean
        
        # 归一化到0-1范围
        return min(1.0, max(0.0, discriminant_power / 5))
    
    def _generate_recommendations(self, sorted_stats: List[tuple]) -> List[str]:
        """
        生成建议
        
        参数：
            sorted_stats: 排序后的模型统计信息
        
        返回：
            建议列表
        """
        recommendations = []
        
        if len(sorted_stats) >= 4:
            # 检查是否有明显的梯度
            top_mean = sorted_stats[0][1]['mean']
            bottom_mean = sorted_stats[-1][1]['mean']
            
            if top_mean - bottom_mean >= 2:
                recommendations.append("评估框架具有良好的模型区分能力")
            elif top_mean - bottom_mean >= 1:
                recommendations.append("评估框架具有中等的模型区分能力")
            else:
                recommendations.append("建议增加测试用例的难度以提高区分能力")
        
        # 检查分数分布是否合理
        scores = []
        for results in self.model_results.values():
            scores.extend([r.get('overall_score', 0) for r in results])
        
        if scores:
            avg_score = sum(scores) / len(scores)
            
            if avg_score < 2:
                recommendations.append("测试用例可能过于困难")
            elif avg_score > 4:
                recommendations.append("测试用例可能过于简单")
        
        return recommendations
    
    def clear_results(self) -> None:
        """
        清除所有模型结果
        """
        self.model_results = {}
