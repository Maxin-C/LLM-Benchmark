"""
医疗专业工具
"""

from typing import List, Dict, Any

class MedicalUtils:
    """
    医疗专业工具类
    """
    
    # 乳腺癌分期标准
    STAGE_CRITERIA = {
        '0期': {'tumor_size': '<=2cm', 'lymph_node': '无转移', 'metastasis': '无'},
        'I期': {'tumor_size': '<=2cm', 'lymph_node': '无转移', 'metastasis': '无'},
        'IIA期': {'tumor_size': '2-5cm', 'lymph_node': '无转移', 'metastasis': '无'},
        'IIB期': {'tumor_size': '>5cm', 'lymph_node': '无转移', 'metastasis': '无'},
        'IIIA期': {'tumor_size': '任意', 'lymph_node': '1-3个转移', 'metastasis': '无'},
        'IIIB期': {'tumor_size': '任意', 'lymph_node': '4-9个转移', 'metastasis': '无'},
        'IIIC期': {'tumor_size': '任意', 'lymph_node': '>=10个转移', 'metastasis': '无'},
        'IV期': {'tumor_size': '任意', 'lymph_node': '任意', 'metastasis': '有'}
    }
    
    # 常见治疗方案
    TREATMENT_PROTOCOLS = {
        '手术治疗': ['乳房切除术', '保乳手术', '前哨淋巴结活检', '腋窝淋巴结清扫'],
        '化疗': ['AC方案', 'TC方案', 'AC-T方案', 'CMF方案'],
        '放疗': ['全乳放疗', '部分乳腺放疗', '淋巴结放疗'],
        '内分泌治疗': ['他莫昔芬', '来曲唑', '阿那曲唑', '依西美坦'],
        '靶向治疗': ['曲妥珠单抗', '帕妥珠单抗', '拉帕替尼', '吡咯替尼']
    }
    
    # 常见症状
    SYMPTOMS = {
        '局部症状': ['乳房肿块', '乳房疼痛', '乳头溢液', '乳头内陷', '皮肤改变'],
        '全身症状': ['乏力', '发热', '体重下降', '食欲不振'],
        '转移症状': ['骨痛', '咳嗽', '呼吸困难', '头痛', '黄疸']
    }
    
    @staticmethod
    def validate_stage(stage: str, patient_info: Dict[str, Any]) -> bool:
        """
        验证分期是否合理
        
        参数：
            stage: 分期
            patient_info: 患者信息
        
        返回：
            是否合理
        """
        if stage not in MedicalUtils.STAGE_CRITERIA:
            return False
        
        criteria = MedicalUtils.STAGE_CRITERIA[stage]
        
        # 简单验证逻辑
        if 'IV期' in stage:
            return patient_info.get('metastasis', '无') == '有'
        
        return True
    
    @staticmethod
    def get_treatment_options(stage: str) -> List[str]:
        """
        根据分期获取推荐治疗方案
        
        参数：
            stage: 分期
        
        返回：
            治疗方案列表
        """
        if stage in ['0期', 'I期']:
            return ['手术治疗', '放疗']
        elif stage in ['IIA期', 'IIB期']:
            return ['手术治疗', '化疗', '放疗', '内分泌治疗']
        elif stage in ['IIIA期', 'IIIB期', 'IIIC期']:
            return ['手术治疗', '化疗', '放疗', '内分泌治疗', '靶向治疗']
        elif 'IV期' in stage:
            return ['化疗', '内分泌治疗', '靶向治疗', '支持治疗']
        else:
            return []
    
    @staticmethod
    def check_treatment_eligibility(treatment: str, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查治疗方案的适用性
        
        参数：
            treatment: 治疗方案
            patient_info: 患者信息
        
        返回：
            适用性检查结果
        """
        result = {
            'is_eligible': True,
            'warnings': [],
            'contraindications': []
        }
        
        age = patient_info.get('age', 0)
        stage = patient_info.get('stage', '')
        
        # 年龄相关限制
        if treatment == '化疗' and age > 70:
            result['warnings'].append("年龄较大，需评估化疗耐受性")
        
        if treatment == '放疗' and age > 80:
            result['warnings'].append("年龄较大，需评估放疗耐受性")
        
        # 分期相关限制
        if treatment == '手术治疗' and 'IV期' in stage:
            result['is_eligible'] = False
            result['contraindications'].append("IV期患者通常不建议手术治疗")
        
        return result
    
    @staticmethod
    def calculate_risk_score(patient_info: Dict[str, Any]) -> float:
        """
        计算风险评分
        
        参数：
            patient_info: 患者信息
        
        返回：
            风险评分 (0-1)
        """
        score = 0.0
        
        # 分期评分
        stage = patient_info.get('stage', '')
        if 'IV期' in stage:
            score += 0.4
        elif 'III期' in stage:
            score += 0.25
        elif 'II期' in stage:
            score += 0.1
        
        # 年龄评分
        age = patient_info.get('age', 0)
        if age > 70:
            score += 0.15
        
        # 并发症评分
        complications = patient_info.get('complications', [])
        if complications:
            score += min(0.15, len(complications) * 0.05)
        
        return min(1.0, score)
    
    @staticmethod
    def generate_prognosis(patient_info: Dict[str, Any]) -> str:
        """
        生成预后评估
        
        参数：
            patient_info: 患者信息
        
        返回：
            预后评估文本
        """
        risk_score = MedicalUtils.calculate_risk_score(patient_info)
        
        if risk_score < 0.2:
            return "预后良好，5年生存率较高"
        elif risk_score < 0.4:
            return "预后中等，需要定期随访"
        elif risk_score < 0.6:
            return "预后较差，需要加强治疗和监测"
        else:
            return "预后不良，建议积极治疗和支持护理"
    
    @staticmethod
    def format_patient_summary(patient_info: Dict[str, Any]) -> str:
        """
        格式化患者摘要
        
        参数：
            patient_info: 患者信息
        
        返回：
            格式化的患者摘要
        """
        summary = f"患者信息：\n"
        summary += f"- 年龄：{patient_info.get('age', '未知')}岁\n"
        summary += f"- 性别：{patient_info.get('gender', '未知')}\n"
        summary += f"- 病理类型：{patient_info.get('pathology_type', '未知')}\n"
        summary += f"- 分期：{patient_info.get('stage', '未知')}\n"
        summary += f"- 手术方式：{patient_info.get('surgery_type', '未知')}\n"
        summary += f"- 治疗阶段：{patient_info.get('treatment_stage', '未知')}\n"
        
        medications = patient_info.get('medications', [])
        if medications:
            summary += f"- 当前用药：{', '.join(medications)}\n"
        
        return summary
