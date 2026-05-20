"""
指南处理器
负责处理和管理医学指南数据
"""

import json
from typing import Dict, List, Any

class GuidelineProcessor:
    def __init__(self):
        self.guidelines = {}
        self.disease_map = {}
    
    def load_guideline(self, guideline_name: str, file_path: str) -> None:
        """
        加载指南文件
        
        参数：
            guideline_name: 指南名称
            file_path: 指南文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.guidelines[guideline_name] = data
        
        # 更新疾病映射
        if 'diseases' in data:
            for disease in data['diseases']:
                if disease not in self.disease_map:
                    self.disease_map[disease] = []
                self.disease_map[disease].append(guideline_name)
    
    def get_guideline(self, guideline_name: str) -> Dict[str, Any]:
        """
        获取指南内容
        
        参数：
            guideline_name: 指南名称
        
        返回：
            指南内容
        """
        return self.guidelines.get(guideline_name, {})
    
    def get_treatment_recommendations(self, disease_name: str) -> List[Dict[str, Any]]:
        """
        获取疾病的治疗推荐
        
        参数：
            disease_name: 疾病名称
        
        返回：
            治疗推荐列表
        """
        recommendations = []
        
        # 查找相关指南
        if disease_name in self.disease_map:
            for guideline_name in self.disease_map[disease_name]:
                guideline = self.guidelines[guideline_name]
                if 'treatments' in guideline:
                    for treatment in guideline['treatments']:
                        if treatment.get('disease') == disease_name:
                            recommendations.append(treatment)
        
        return recommendations
    
    def check_contraindication(self, drug_name: str, patient_info: Dict[str, Any]) -> bool:
        """
        检查药物禁忌症
        
        参数：
            drug_name: 药物名称
            patient_info: 患者信息
        
        返回：
            是否存在禁忌症
        """
        for guideline in self.guidelines.values():
            if 'contraindications' in guideline:
                for contraindication in guideline['contraindications']:
                    if contraindication.get('drug') == drug_name:
                        conditions = contraindication.get('conditions', [])
                        for condition in conditions:
                            if self._check_condition(condition, patient_info):
                                return True
        
        return False
    
    def _check_condition(self, condition: Dict[str, Any], patient_info: Dict[str, Any]) -> bool:
        """
        检查条件是否满足
        
        参数：
            condition: 条件
            patient_info: 患者信息
        
        返回：
            条件是否满足
        """
        field = condition.get('field')
        operator = condition.get('operator')
        value = condition.get('value')
        
        patient_value = patient_info.get(field)
        
        if patient_value is None:
            return False
        
        if operator == 'equals':
            return patient_value == value
        elif operator == 'greater_than':
            return patient_value > value
        elif operator == 'less_than':
            return patient_value < value
        elif operator == 'contains':
            return value in str(patient_value)
        elif operator == 'in':
            return patient_value in value
        
        return False
    
    def get_dosage_recommendation(self, drug_name: str, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取药物剂量推荐
        
        参数：
            drug_name: 药物名称
            patient_info: 患者信息
        
        返回：
            剂量推荐
        """
        for guideline in self.guidelines.values():
            if 'dosages' in guideline:
                for dosage in guideline['dosages']:
                    if dosage.get('drug') == drug_name:
                        # 检查适用条件
                        if 'conditions' in dosage:
                            conditions_met = True
                            for condition in dosage['conditions']:
                                if not self._check_condition(condition, patient_info):
                                    conditions_met = False
                                    break
                            
                            if conditions_met:
                                return dosage
                        else:
                            return dosage
        
        return {}
    
    def validate_treatment_plan(self, disease_name: str, treatment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证治疗方案是否符合指南
        
        参数：
            disease_name: 疾病名称
            treatment_plan: 治疗方案
        
        返回：
            验证结果
        """
        recommendations = self.get_treatment_recommendations(disease_name)
        
        result = {
            'is_valid': False,
            'matches_guideline': False,
            'errors': [],
            'warnings': []
        }
        
        if not recommendations:
            result['warnings'].append(f"未找到{disease_name}的治疗指南")
            return result
        
        # 检查治疗方案是否在推荐范围内
        plan_drugs = treatment_plan.get('drugs', [])
        
        for drug in plan_drugs:
            drug_name = drug.get('name')
            found = False
            
            for rec in recommendations:
                rec_drugs = rec.get('drugs', [])
                if drug_name in rec_drugs:
                    found = True
                    break
            
            if not found:
                result['errors'].append(f"药物{drug_name}不在{disease_name}的推荐治疗方案中")
        
        # 检查剂量
        for drug in plan_drugs:
            drug_name = drug.get('name')
            dosage = drug.get('dosage')
            
            if dosage:
                rec_dosage = self.get_dosage_recommendation(drug_name, treatment_plan.get('patient_info', {}))
                if rec_dosage:
                    rec_min = rec_dosage.get('min_dosage')
                    rec_max = rec_dosage.get('max_dosage')
                    
                    if dosage < rec_min:
                        result['warnings'].append(f"{drug_name}剂量低于推荐最小值")
                    elif dosage > rec_max:
                        result['errors'].append(f"{drug_name}剂量超过推荐最大值")
        
        result['is_valid'] = len(result['errors']) == 0
        result['matches_guideline'] = len(result['errors']) == 0 and len(result['warnings']) == 0
        
        return result
    
    def get_guideline_names(self) -> List[str]:
        """
        获取所有指南名称
        
        返回：
            指南名称列表
        """
        return list(self.guidelines.keys())
    
    def get_diseases(self) -> List[str]:
        """
        获取所有涵盖的疾病
        
        返回：
            疾病列表
        """
        return list(self.disease_map.keys())
