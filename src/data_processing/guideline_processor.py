"""
指南处理器
负责处理和管理医学指南数据
"""

import json
import os
from typing import Dict, List, Any

class GuidelineProcessor:
    def __init__(self, guidelines_dir: str = "dataset/guidlines/formated_data_llm_optimized"):
        self.guidelines = {}
        self.disease_map = {}
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
            self.load_guideline(guideline_name, file_path)
    
    def load_guideline(self, guideline_name: str, file_path: str) -> None:
        """
        加载指南文件
        
        参数：
            guideline_name: 指南名称
            file_path: 指南文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.guidelines[guideline_name] = data
            
            # 尝试从标题中提取疾病信息
            title = data.get('title', '')
            if '乳腺癌' in title:
                if '乳腺癌' not in self.disease_map:
                    self.disease_map['乳腺癌'] = []
                self.disease_map['乳腺癌'].append(guideline_name)
            
            print(f"已加载指南: {guideline_name}")
        except Exception as e:
            print(f"加载指南失败 {guideline_name}: {e}")
    
    def get_guideline(self, guideline_name: str) -> Dict[str, Any]:
        """
        获取指南内容
        
        参数：
            guideline_name: 指南名称
        
        返回：
            指南内容
        """
        return self.guidelines.get(guideline_name, {})
    
    def search_sections(self, keyword: str) -> List[Dict[str, Any]]:
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
    
    def get_treatment_recommendations(self, disease_name: str) -> List[Dict[str, Any]]:
        """
        获取疾病的治疗推荐
        
        参数：
            disease_name: 疾病名称
        
        返回：
            治疗推荐列表
        """
        recommendations = []
        
        # 搜索包含"治疗"关键字的章节
        sections = self.search_sections('治疗')
        
        for section in sections:
            if disease_name in section['guideline_name'] or disease_name in section['section_name']:
                recommendations.append({
                    'source': section['guideline_name'],
                    'section': section['section_name'],
                    'content': section['content']
                })
        
        return recommendations
    
    def check_contraindication(self, drug_name: str, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查药物禁忌症
        
        参数：
            drug_name: 药物名称
            patient_info: 患者信息
        
        返回：
            禁忌症检查结果
        """
        result = {
            'has_contraindication': False,
            'details': []
        }
        
        # 搜索包含"禁忌"关键字的章节
        sections = self.search_sections('禁忌')
        
        for section in sections:
            content = section['content']
            if drug_name in content:
                result['has_contraindication'] = True
                result['details'].append({
                    'source': section['guideline_name'],
                    'section': section['section_name'],
                    'content': content
                })
        
        return result
    
    def get_dosage_recommendation(self, drug_name: str) -> Dict[str, Any]:
        """
        获取药物剂量推荐
        
        参数：
            drug_name: 药物名称
        
        返回：
            剂量推荐
        """
        result = {
            'found': False,
            'recommendations': []
        }
        
        # 搜索包含"剂量"关键字的章节
        sections = self.search_sections('剂量')
        
        for section in sections:
            content = section['content']
            if drug_name in content:
                result['found'] = True
                result['recommendations'].append({
                    'source': section['guideline_name'],
                    'section': section['section_name'],
                    'content': content
                })
        
        return result
    
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
            'warnings': [],
            'recommendations': recommendations
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
                content = rec.get('content', '')
                if drug_name in content:
                    found = True
                    break
            
            if not found:
                result['warnings'].append(f"药物{drug_name}未在指南推荐中明确提及")
        
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
    
    def get_all_sections(self, guideline_name: str) -> List[Dict[str, Any]]:
        """
        获取指定指南的所有章节
        
        参数：
            guideline_name: 指南名称
        
        返回：
            章节列表
        """
        guideline = self.get_guideline(guideline_name)
        return guideline.get('sections', [])