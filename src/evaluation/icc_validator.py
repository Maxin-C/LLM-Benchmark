"""
ICC一致性验证模块

用于计算评估模型与真实医生评估之间的组内相关系数(ICC)，确保评估结果的可靠性。
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


class ICCValidator:
    """
    ICC一致性验证器
    
    计算LLM评估器与人工评估之间的一致性，支持多种ICC模型：
    - ICC(1,1): 单评分者、随机效应模型
    - ICC(2,1): 双评分者、随机效应模型
    - ICC(3,1): 固定效应模型
    
    参考：Shrout & Fleiss (1979)
    """
    
    def __init__(self):
        pass
    
    def calculate_icc(self, model_ratings: List[float], human_ratings: List[float]) -> Dict[str, float]:
        """
        计算多种ICC指标
        
        参数：
            model_ratings: LLM评估器的评分列表
            human_ratings: 人工评估的评分列表
        
        返回：
            包含各ICC值的字典
        """
        if len(model_ratings) != len(human_ratings):
            raise ValueError("评分列表长度不一致")
        
        n = len(model_ratings)
        k = 2  # 两位评估者：模型和人工
        
        # 转换为numpy数组
        model_arr = np.array(model_ratings)
        human_arr = np.array(human_ratings)
        
        # 计算均值
        grand_mean = np.mean([model_arr, human_arr])
        model_mean = np.mean(model_arr)
        human_mean = np.mean(human_arr)
        
        # 计算平方和
        # SS_total = sum over all observations (x_ij - grand_mean)^2
        ss_total = np.sum((model_arr - grand_mean) ** 2) + np.sum((human_arr - grand_mean) ** 2)
        
        # SS_between = k * sum over subjects (mean_i - grand_mean)^2
        subject_means = (model_arr + human_arr) / 2
        ss_between = k * np.sum((subject_means - grand_mean) ** 2)
        
        # SS_error = SS_total - SS_between - SS_rater
        # SS_rater = n * sum over raters (rater_mean - grand_mean)^2
        ss_rater = n * ((model_mean - grand_mean) ** 2 + (human_mean - grand_mean) ** 2)
        
        # SS_within = SS_total - SS_between
        ss_within = ss_total - ss_between
        
        # SS_error = SS_within - SS_rater (for ICC(2,1) and ICC(3,1))
        ss_error = ss_within - ss_rater
        
        # 自由度
        df_between = n - 1
        df_within = n * (k - 1)
        df_rater = k - 1
        df_error = (n - 1) * (k - 1)
        
        # 均方
        ms_between = ss_between / df_between
        ms_error = ss_error / df_error
        ms_within = ss_within / df_within
        
        # ICC(1,1): 单评分者、随机效应
        # ICC(1,1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
        icc11 = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        
        # ICC(2,1): 双评分者、随机效应
        # ICC(2,1) = (MS_between - MS_error) / (MS_between + (k-1)*MS_error + k*(MS_rater - MS_error)/n)
        ms_rater = ss_rater / df_rater
        icc21 = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error + k * (ms_rater - ms_error) / n)
        
        # ICC(3,1): 固定效应
        # ICC(3,1) = (MS_between - MS_error) / (MS_between + (k-1)*MS_error)
        icc31 = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error)
        
        return {
            'ICC(1,1)': round(icc11, 4),
            'ICC(2,1)': round(icc21, 4),
            'ICC(3,1)': round(icc31, 4),
            'n': n,
            'model_mean': round(model_mean, 4),
            'human_mean': round(human_mean, 4),
            'model_std': round(np.std(model_arr), 4),
            'human_std': round(np.std(human_arr), 4)
        }
    
    def calculate_pearson_correlation(self, model_ratings: List[float], human_ratings: List[float]) -> float:
        """
        计算皮尔逊相关系数
        
        参数：
            model_ratings: LLM评估器的评分列表
            human_ratings: 人工评估的评分列表
        
        返回：
            皮尔逊相关系数
        """
        if len(model_ratings) < 2 or len(human_ratings) < 2:
            return 0.0
        
        corr, _ = stats.pearsonr(model_ratings, human_ratings)
        return round(corr, 4)
    
    def calculate_spearman_correlation(self, model_ratings: List[float], human_ratings: List[float]) -> float:
        """
        计算斯皮尔曼秩相关系数
        
        参数：
            model_ratings: LLM评估器的评分列表
            human_ratings: 人工评估的评分列表
        
        返回：
            斯皮尔曼相关系数
        """
        if len(model_ratings) < 2 or len(human_ratings) < 2:
            return 0.0
        
        corr, _ = stats.spearmanr(model_ratings, human_ratings)
        return round(corr, 4)
    
    def calculate_mae(self, model_ratings: List[float], human_ratings: List[float]) -> float:
        """
        计算平均绝对误差
        
        参数：
            model_ratings: LLM评估器的评分列表
            human_ratings: 人工评估的评分列表
        
        返回：
            MAE值
        """
        model_arr = np.array(model_ratings)
        human_arr = np.array(human_ratings)
        return round(np.mean(np.abs(model_arr - human_arr)), 4)
    
    def calculate_rmse(self, model_ratings: List[float], human_ratings: List[float]) -> float:
        """
        计算均方根误差
        
        参数：
            model_ratings: LLM评估器的评分列表
            human_ratings: 人工评估的评分列表
        
        返回：
            RMSE值
        """
        model_arr = np.array(model_ratings)
        human_arr = np.array(human_ratings)
        return round(np.sqrt(np.mean((model_arr - human_arr) ** 2)), 4)
    
    def calculate_bland_altman_stats(self, model_ratings: List[float], human_ratings: List[float]) -> Dict[str, float]:
        """
        计算Bland-Altman分析统计量
        
        参数：
            model_ratings: LLM评估器的评分列表
            human_ratings: 人工评估的评分列表
        
        返回：
            包含均值差、标准差、95%一致性界限的字典
        """
        model_arr = np.array(model_ratings)
        human_arr = np.array(human_ratings)
        
        # 计算差值
        differences = model_arr - human_arr
        
        # 统计量
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        loa_lower = mean_diff - 1.96 * std_diff
        loa_upper = mean_diff + 1.96 * std_diff
        
        return {
            'mean_difference': round(mean_diff, 4),
            'std_difference': round(std_diff, 4),
            'loa_lower': round(loa_lower, 4),
            'loa_upper': round(loa_upper, 4)
        }
    
    def validate_consistency(self, model_ratings: List[float], human_ratings: List[float]) -> Dict[str, float]:
        """
        执行完整的一致性验证
        
        参数：
            model_ratings: LLM评估器的评分列表
            human_ratings: 人工评估的评分列表
        
        返回：
            包含所有一致性指标的字典
        """
        results = {}
        
        # ICC指标
        icc_results = self.calculate_icc(model_ratings, human_ratings)
        results.update(icc_results)
        
        # 相关系数
        results['pearson_correlation'] = self.calculate_pearson_correlation(model_ratings, human_ratings)
        results['spearman_correlation'] = self.calculate_spearman_correlation(model_ratings, human_ratings)
        
        # 误差指标
        results['mae'] = self.calculate_mae(model_ratings, human_ratings)
        results['rmse'] = self.calculate_rmse(model_ratings, human_ratings)
        
        # Bland-Altman统计
        ba_results = self.calculate_bland_altman_stats(model_ratings, human_ratings)
        results.update(ba_results)
        
        # 一致性判定
        icc_threshold = 0.75  # ICC≥0.75表示良好一致性
        corr_threshold = 0.7   # 相关系数≥0.7表示良好相关性
        mae_threshold = 0.5    # MAE≤0.5表示误差较小
        
        results['is_consistent'] = (
            icc_results['ICC(2,1)'] >= icc_threshold and
            results['pearson_correlation'] >= corr_threshold and
            results['mae'] <= mae_threshold
        )
        
        return results
    
    def interpret_icc(self, icc_value: float) -> str:
        """
        解释ICC值的含义
        
        参数：
            icc_value: ICC值
        
        返回：
            解释文本
        """
        if icc_value >= 0.90:
            return "极好的一致性 (Excellent)"
        elif icc_value >= 0.75:
            return "良好的一致性 (Good)"
        elif icc_value >= 0.60:
            return "中等一致性 (Moderate)"
        elif icc_value >= 0.40:
            return "一般一致性 (Fair)"
        else:
            return "较差的一致性 (Poor)"


def load_human_evaluations(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    加载人工评估数据
    
    参数：
        file_path: 人工评估数据文件路径（JSON格式）
    
    返回：
        人工评估数据字典
    """
    import json
    
    if not file_path:
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载人工评估数据失败: {e}")
        return {}


def run_icc_validation(model_results: List[Dict], human_eval_file: str = None) -> Dict[str, float]:
    """
    运行完整的ICC一致性验证流程
    
    参数：
        model_results: 模型评估结果列表
        human_eval_file: 人工评估数据文件路径
    
    返回：
        一致性验证结果
    """
    # 提取模型评分
    model_ratings = [result.get('evaluation', {}).get('overall_score', 0) for result in model_results]
    
    # 加载人工评分
    human_evaluations = load_human_evaluations(human_eval_file)
    
    # 匹配案例
    human_ratings = []
    for result in model_results:
        case_id = result.get('case_id')
        patient_id = result.get('patient_id')
        
        # 尝试匹配
        human_score = None
        if case_id and str(case_id) in human_evaluations:
            human_score = human_evaluations[str(case_id)].get('overall_score')
        elif patient_id and patient_id in human_evaluations:
            human_score = human_evaluations[patient_id].get('overall_score')
        
        if human_score is not None:
            human_ratings.append(human_score)
    
    # 如果没有人工评分，返回空结果
    if not human_ratings or len(model_ratings) != len(human_ratings):
        print("警告：无法匹配足够的人工评估数据")
        return {'error': '无法匹配人工评估数据'}
    
    # 执行验证
    validator = ICCValidator()
    return validator.validate_consistency(model_ratings, human_ratings)