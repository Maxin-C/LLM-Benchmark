"""
数据处理模块测试
"""

import pytest
import tempfile
import os
import pandas as pd
from src.data_processing import EHRParser, KnowledgeGraphLoader, ExpertDataProcessor

class TestEHRParser:
    """测试EHR解析器"""
    
    def test_parse_excel(self):
        """测试解析Excel文件"""
        parser = EHRParser()
        
        # 创建临时Excel文件
        temp_path = tempfile.mktemp(suffix='.xlsx')
        
        # 创建测试数据
        data = {
            'patient_id': ['P001', 'P002'],
            'age': [45, 52],
            'gender': ['女', '女'],
            'diagnosis': ['乳腺癌', '乳腺癌'],
            'stage': ['IIB期', 'III期']
        }
        df = pd.DataFrame(data)
        df.to_excel(temp_path, index=False)
        
        try:
            result = parser.parse_excel(temp_path)
            
            assert result is not None
            assert len(result) == 2
            assert result[0]['patient_id'] == 'P001'
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_extract_patient(self):
        """测试提取患者信息"""
        parser = EHRParser()
        
        row_data = {
            'patient_id': 'P001',
            'age': 45,
            'gender': '女',
            'pathology_type': '乳腺癌',
            'stage': 'IIB期',
            'surgery_type': '乳房切除术'
        }
        
        patient = parser._parse_row(row_data)
        
        assert patient['patient_id'] == 'P001'
        assert patient['age'] == 45
        assert patient['pathology_type'] == '乳腺癌'

class TestKnowledgeGraphLoader:
    """测试知识图谱加载器"""
    
    def test_load_json(self):
        """测试加载JSON格式图谱"""
        loader = KnowledgeGraphLoader()
        
        # 创建临时JSON文件
        import json
        temp_path = tempfile.mktemp(suffix='.json')
        kg_data = {
            'nodes': [{'id': 'n1', 'label': '乳腺癌', 'type': 'disease'}],
            'edges': [{'source': 'n1', 'target': 'n2', 'relation': 'treats'}]
        }
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f)
        
        try:
            loader.load_json(temp_path)
            
            assert loader.graph is not None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_node_info(self):
        """测试获取节点信息"""
        loader = KnowledgeGraphLoader()
        
        # 创建临时JSON文件
        import json
        temp_path = tempfile.mktemp(suffix='.json')
        kg_data = {
            'nodes': [{'id': 'breast_cancer', 'label': '乳腺癌', 'type': 'disease'}],
            'edges': []
        }
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f)
        
        try:
            loader.load_json(temp_path)
            info = loader.get_node_info('breast_cancer')
            
            assert info is not None
            assert info['label'] == '乳腺癌'
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestExpertDataProcessor:
    """测试专家数据处理器"""
    
    def test_load_from_json(self):
        """测试从JSON加载数据"""
        processor = ExpertDataProcessor()
        
        # 创建临时JSON文件
        import json
        temp_path = tempfile.mktemp(suffix='.json')
        expert_data = {
            'doctor_ratings': [
                {'dialogue_id': 'case_001', 'doctor_ratings': {'accuracy': 4, 'empathy': 3}},
                {'dialogue_id': 'case_002', 'doctor_ratings': {'accuracy': 5, 'empathy': 4}}
            ]
        }
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(expert_data, f)
        
        try:
            processor.load_from_json(temp_path)
            
            assert len(processor.ratings) == 2
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_split_train_test(self):
        """测试划分训练测试集"""
        processor = ExpertDataProcessor()
        
        # 添加测试数据
        processor.ratings = [
            {'dialogue_id': f'case_{i:03d}', 'doctor_ratings': {'accuracy': 4, 'effectiveness': 4, 'safety': 4, 'personalization': 4, 'empathy': 4}} 
            for i in range(10)
        ]
        
        train, test = processor.split_train_test(test_ratio=0.3)
        
        assert len(train) == 7
        assert len(test) == 3
    
    def test_get_few_shot_samples(self):
        """测试获取Few-shot样本"""
        processor = ExpertDataProcessor()
        
        processor.ratings = [
            {'dialogue_id': f'case_{i:03d}', 'doctor_ratings': {'accuracy': 4, 'effectiveness': 4, 'safety': 4, 'personalization': 4, 'empathy': 4}} 
            for i in range(5)
        ]
        
        samples = processor.get_few_shot_samples(num_samples=3)
        
        assert len(samples) == 3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
