"""
EHR数据解析器
负责从EHR数据中提取患者的人口统计学信息和医学信息
"""

import pandas as pd
import json
from typing import Dict, Any, Optional

class EHRParser:
    def __init__(self):
        pass
    
    def parse_excel(self, file_path: str) -> list:
        """
        从Excel文件中解析EHR数据
        
        参数：
            file_path: EHR数据Excel文件路径
        
        返回：
            患者EHR数据列表
        """
        df = pd.read_excel(file_path)
        patients = []
        
        for _, row in df.iterrows():
            patient = self._parse_row(row)
            if patient:
                patients.append(patient)
        
        return patients
    
    def parse_json(self, file_path: str) -> list:
        """
        从JSON文件中解析EHR数据
        
        参数：
            file_path: EHR数据JSON文件路径
        
        返回：
            患者EHR数据列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'patients' in data:
            return data['patients']
        else:
            return [data]
    
    def _parse_row(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        解析单行EHR数据
        
        参数：
            row: 单行EHR数据
        
        返回：
            结构化的患者EHR数据
        """
        patient = {
            'patient_id': str(row.get('patient_id', '')),
            'age': row.get('age'),
            'gender': row.get('gender'),
            'occupation': row.get('occupation'),
            'pathology_type': row.get('pathology_type'),
            'stage': row.get('stage'),
            'surgery_type': row.get('surgery_type'),
            'medications': self._parse_medications(row.get('medications', '')),
            'treatment_stage': row.get('treatment_stage'),
            'diagnosis_date': str(row.get('diagnosis_date', '')),
            'surgery_date': str(row.get('surgery_date', ''))
        }
        
        # 验证必要字段
        if not patient['patient_id']:
            return None
        
        return patient
    
    def _parse_medications(self, medications: str) -> list:
        """
        解析用药列表
        
        参数：
            medications: 用药字符串，以逗号分隔
        
        返回：
            用药列表
        """
        if not medications:
            return []
        
        if isinstance(medications, str):
            return [m.strip() for m in medications.split(',') if m.strip()]
        elif isinstance(medications, list):
            return medications
        else:
            return []
    
    def extract_patient(self, patient_id: str, ehr_data: list) -> Optional[Dict[str, Any]]:
        """
        根据患者ID提取特定患者的EHR数据
        
        参数：
            patient_id: 患者ID
            ehr_data: EHR数据列表
        
        返回：
            特定患者的EHR数据
        """
        for patient in ehr_data:
            if patient.get('patient_id') == patient_id:
                return patient
        
        return None
