"""
专家数据处理器
负责处理和清洗专家评分数据
"""

import json
import pandas as pd
from typing import Dict, List, Any

class ExpertDataProcessor:
    def __init__(self):
        self.ratings = []
        self.dialogues = []
        self.dimensions = ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']
    
    def load_from_json(self, file_path: str) -> None:
        """
        从JSON文件加载专家评分数据
        
        参数：
            file_path: 专家评分JSON文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'doctor_ratings' in data:
            self.ratings = data['doctor_ratings']
        
        if 'dialogues' in data:
            self.dialogues = data['dialogues']
    
    def load_from_excel(self, file_path: str) -> None:
        """
        从Excel文件加载专家评分数据
        
        参数：
            file_path: 专家评分Excel文件路径
        """
        df = pd.read_excel(file_path)
        self._parse_excel_data(df)
    
    def _parse_excel_data(self, df: pd.DataFrame) -> None:
        """
        解析Excel数据
        
        参数：
            df: 专家评分DataFrame
        """
        for _, row in df.iterrows():
            dialogue_id = str(row.get('dialogue_id', ''))
            if not dialogue_id:
                continue
            
            rating = {
                'dialogue_id': dialogue_id,
                'doctor_ratings': {}
            }
            
            for dim in self.dimensions:
                rating['doctor_ratings'][dim] = row.get(dim)
            
            rating['doctor_rationale'] = row.get('doctor_rationale', '')
            rating['critical_errors'] = self._parse_critical_errors(row.get('critical_errors', ''))
            
            self.ratings.append(rating)
    
    def _parse_critical_errors(self, errors: str) -> List[str]:
        """
        解析严重错误列表
        
        参数：
            errors: 错误字符串
        
        返回：
            错误列表
        """
        if not errors:
            return []
        
        if isinstance(errors, str):
            return [e.strip() for e in errors.split(',') if e.strip()]
        elif isinstance(errors, list):
            return errors
        else:
            return []
    
    def clean_data(self) -> None:
        """
        清洗数据
        """
        cleaned_ratings = []
        
        for rating in self.ratings:
            # 验证必要字段
            if not rating.get('dialogue_id'):
                continue
            
            # 验证评分维度
            valid = True
            for dim in self.dimensions:
                value = rating['doctor_ratings'].get(dim)
                if value is None or not (1 <= value <= 5):
                    valid = False
                    break
            
            if valid:
                cleaned_ratings.append(rating)
        
        self.ratings = cleaned_ratings
    
    def split_train_test(self, test_ratio: float = 0.2) -> tuple:
        """
        划分训练集和测试集
        
        参数：
            test_ratio: 测试集比例
        
        返回：
            (训练集, 测试集)
        """
        import random
        
        shuffled = self.ratings.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - test_ratio))
        train_set = shuffled[:split_idx]
        test_set = shuffled[split_idx:]
        
        return train_set, test_set
    
    def get_few_shot_samples(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """
        获取Few-shot样本
        
        参数：
            num_samples: 样本数量
        
        返回：
            Few-shot样本列表
        """
        import random
        
        if len(self.ratings) <= num_samples:
            return self.ratings.copy()
        
        return random.sample(self.ratings, num_samples)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        返回：
            统计信息字典
        """
        if not self.ratings:
            return {}
        
        # 计算各维度评分统计
        stats = {
            'total_samples': len(self.ratings),
            'dimensions': {}
        }
        
        for dim in self.dimensions:
            values = [r['doctor_ratings'][dim] for r in self.ratings]
            stats['dimensions'][dim] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': (sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5
            }
        
        return stats
    
    def save_to_json(self, file_path: str) -> None:
        """
        保存处理后的数据到JSON文件
        
        参数：
            file_path: 输出文件路径
        """
        output = {
            'doctor_ratings': self.ratings,
            'dialogues': self.dialogues,
            'dimensions': self.dimensions
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
