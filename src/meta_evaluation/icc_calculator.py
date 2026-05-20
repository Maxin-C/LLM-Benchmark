"""
ICC计算器
负责计算评分者间一致性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

class ICCCalculator:
    """
    ICC一致性计算接口
    
    输入：
        - 100份对话数据
        - 4位医生对这100份对话的评估结果（每份对话包含5维评分）
        - [可选] EASE-Judge对这100份对话的自动化评估结果
    
    输出：
        - ICC值及置信区间
        - 各维度ICC值
        - 评分者间一致性分析报告
    """
    
    def __init__(self):
        self.raters = []
    
    def add_rater(self, rater_id: str, ratings: List[List[float]]) -> None:
        """
        添加一位评分者的评分数据
        
        参数：
            rater_id: 评分者标识（如 "doctor_1", "doctor_2", "ease_judge"）
            ratings: 评分列表，格式为 [(acc1, eff1, safe1, pers1, emp1), ...]
                     长度需与对话数量一致
        """
        self.raters.append({
            'rater_id': rater_id,
            'ratings': np.array(ratings)
        })
    
    def calculate_icc(self, target_rater: str = "ease_judge") -> Dict[str, Any]:
        """
        计算指定评分者与所有医生之间的ICC
        
        参数：
            target_rater: 目标评分者标识，默认计算EASE-Judge与医生的一致性
        
        返回：
            包含ICC值、置信区间、各维度分析的字典
        """
        # 分离目标评分者和医生评分者
        target_ratings = None
        doctor_ratings = []
        
        for rater in self.raters:
            if rater['rater_id'] == target_rater:
                target_ratings = rater['ratings']
            elif rater['rater_id'].startswith('doctor'):
                doctor_ratings.append(rater['ratings'])
        
        if target_ratings is None:
            raise ValueError(f"未找到目标评分者: {target_rater}")
        
        if not doctor_ratings:
            raise ValueError("未找到医生评分数据")
        
        # 计算综合ICC（5维评分合并）
        all_ratings = np.array([target_ratings] + doctor_ratings)
        overall_icc = self._compute_icc(all_ratings)
        
        # 计算各维度ICC
        dimension_iccs = {}
        dimensions = ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']
        for i, dim in enumerate(dimensions):
            dim_ratings = np.array([r[:, i] for r in [target_ratings] + doctor_ratings])
            dimension_iccs[dim] = self._compute_icc(dim_ratings)
        
        return {
            'overall_icc': overall_icc,
            'dimension_iccs': dimension_iccs,
            'num_raters': len(doctor_ratings) + 1,
            'num_items': len(target_ratings),
            'target_rater': target_rater,
            'doctor_count': len(doctor_ratings)
        }
    
    def _compute_icc(self, ratings: np.ndarray) -> Dict[str, float]:
        """
        使用ICC(2,1)双向随机效应模型计算组内相关系数
        
        参数：
            ratings: 评分矩阵，形状为 (评分者数量, 项目数量) 或 (评分者数量, 项目数量, 维度数量)
        
        返回：
            包含icc值和置信区间的字典
        """
        # 如果是多维评分，先计算平均值
        if ratings.ndim == 3:
            ratings = np.mean(ratings, axis=2)
        
        n_raters, n_items = ratings.shape
        
        # 计算总均值
        grand_mean = np.mean(ratings)
        
        # 计算评分者均值和项目均值
        rater_means = np.mean(ratings, axis=1)
        item_means = np.mean(ratings, axis=0)
        
        # 计算平方和
        ss_total = np.sum((ratings - grand_mean) ** 2)
        ss_rater = n_items * np.sum((rater_means - grand_mean) ** 2)
        ss_item = n_raters * np.sum((item_means - grand_mean) ** 2)
        ss_error = ss_total - ss_rater - ss_item
        
        # 计算自由度
        df_rater = n_raters - 1
        df_item = n_items - 1
        df_error = (n_raters - 1) * (n_items - 1)
        
        # 计算均方
        ms_rater = ss_rater / df_rater
        ms_item = ss_item / df_item
        ms_error = ss_error / df_error
        
        # 计算ICC(2,1)
        icc_value = (ms_rater - ms_error) / (ms_rater + (n_raters - 1) * ms_error)
        
        # 计算标准误和置信区间（简化版本）
        se = np.sqrt((2 * (1 - icc_value) ** 2 * (1 + (n_raters - 1) * icc_value) ** 2) / 
                     (n_items * n_raters * (n_raters - 1)))
        
        # 95%置信区间
        ci_low = max(0, icc_value - 1.96 * se)
        ci_high = min(1, icc_value + 1.96 * se)
        
        return {
            'icc': icc_value,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'standard_error': se
        }
    
    def calculate_doctor_agreement(self) -> Dict[str, Any]:
        """
        计算医生之间的一致性
        
        返回：
            医生间ICC结果
        """
        doctor_ratings = []
        
        for rater in self.raters:
            if rater['rater_id'].startswith('doctor'):
                doctor_ratings.append(rater['ratings'])
        
        if len(doctor_ratings) < 2:
            raise ValueError("至少需要2位医生的评分数据")
        
        all_ratings = np.array(doctor_ratings)
        
        # 计算综合ICC
        overall_icc = self._compute_icc(all_ratings)
        
        # 计算各维度ICC
        dimension_iccs = {}
        dimensions = ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']
        for i, dim in enumerate(dimensions):
            dim_ratings = np.array([r[:, i] for r in doctor_ratings])
            dimension_iccs[dim] = self._compute_icc(dim_ratings)
        
        return {
            'overall_icc': overall_icc,
            'dimension_iccs': dimension_iccs,
            'num_doctors': len(doctor_ratings),
            'num_items': len(doctor_ratings[0])
        }
    
    def get_consistency_level(self, icc_value: float) -> str:
        """
        获取一致性等级
        
        参数：
            icc_value: ICC值
        
        返回：
            一致性等级
        """
        if icc_value >= 0.80:
            return '优秀'
        elif icc_value >= 0.60:
            return '良好'
        elif icc_value >= 0.40:
            return '中等'
        else:
            return '差'
