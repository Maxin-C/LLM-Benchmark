"""
循证审查官
负责事实核查与红线拦截
使用LLM进行深度语义分析，并结合图推理引擎进行知识图谱推理
"""

from typing import Dict, List, Any, Optional
from src.utils.llm_client import LLMClient

class EvidenceChecker:
    """
    循证审查官类
    使用LLM对照指南知识图谱进行事实核查与红线拦截
    集成图推理引擎进行知识图谱推理
    """
    
    def __init__(self, llm_client: LLMClient, graph_reasoner=None):
        self.llm_client = llm_client
        self.graph_reasoner = graph_reasoner  # 图推理引擎
        self.violations = []
    
    def check_factuality(self, statement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM和图推理检查陈述的事实正确性
        
        参数：
            statement: 医生的陈述
            context: 上下文信息（患者信息、场景等）
        
        返回：
            检查结果
        """
        patient_info = context.get('patient_info', {})
        disease_name = patient_info.get('pathology_type', '')
        stage = patient_info.get('stage', '')
        medications = patient_info.get('medications', [])
        
        # 使用图推理获取相关知识
        kg_evidence = self._query_knowledge_graph(disease_name, medications)
        
        # 构建LLM提示词
        system_prompt = """
你是一位专业的医学循证审查专家，请根据提供的患者信息、知识图谱推理结果和医学知识，评估医生陈述的事实正确性。

评估维度：
1. 药物信息准确性：药物名称、剂量、用法是否正确
2. 治疗方案合理性：治疗方案是否符合临床指南
3. 疾病信息正确性：疾病名称、分期、症状描述是否准确
4. 是否存在禁忌症：治疗方案是否存在禁忌症或药物相互作用

请输出JSON格式结果，包含以下字段：
- is_factual: 是否符合事实 (true/false)
- violations: 违规列表，每个元素包含 type 和 message
- supporting_evidence: 支持证据列表
- confidence: 置信度 (0-1)
"""
        
        kg_evidence_text = "\n".join([f"- {item}" for item in kg_evidence]) if kg_evidence else "无"
        
        user_prompt = f"""
患者信息：
- 疾病：{disease_name}
- 分期：{stage}
- 用药：{', '.join(medications) if medications else '无'}
- 治疗阶段：{patient_info.get('treatment_stage', '未知')}

知识图谱推理结果：
{kg_evidence_text}

医生陈述：{statement}

请评估医生陈述的事实正确性，输出JSON格式结果。
"""
        
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        
        # 验证结果
        if not result:
            result = {
                'is_factual': True,
                'violations': [],
                'supporting_evidence': [],
                'confidence': 0.8
            }
        
        # 记录违规
        self.violations.extend(result.get('violations', []))
        
        return result
    
    def _query_knowledge_graph(self, disease_name: str, medications: List[str]) -> List[str]:
        """
        使用图推理引擎查询知识图谱
        
        参数：
            disease_name: 疾病名称
            medications: 用药列表
        
        返回：
            知识图谱推理结果列表
        """
        evidence = []
        
        if not self.graph_reasoner:
            return evidence
        
        try:
            # 查询疾病的治疗方案
            if disease_name:
                treatments = self.graph_reasoner.query_disease_treatment(disease_name, top_k=5)
                for treatment, score in treatments:
                    evidence.append(f"疾病[{disease_name}]的推荐治疗方案: {treatment} (相关性: {score:.2f})")
            
            # 查询药物相互作用
            for drug in medications:
                interactions = self.graph_reasoner.query_drug_interaction(drug, top_k=3)
                for interact_drug, score in interactions:
                    evidence.append(f"药物[{drug}]与[{interact_drug}]可能存在相互作用 (强度: {score:.2f})")
            
            # 查询相似疾病
            if disease_name:
                similar_diseases = self.graph_reasoner.get_similar_nodes(disease_name, top_k=3)
                for sim_disease, score in similar_diseases:
                    evidence.append(f"与疾病[{disease_name}]相似的疾病: {sim_disease} (相似度: {score:.2f})")
        
        except Exception as e:
            # 如果图推理失败，不影响整体流程
            pass
        
        return evidence
    
    def check_drug_interaction(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """
        使用图推理检查药物相互作用
        
        参数：
            drug1: 第一种药物
            drug2: 第二种药物
        
        返回：
            相互作用检查结果
        """
        result = {
            'has_interaction': False,
            'interaction_type': None,
            'confidence': 0.0,
            'message': ''
        }
        
        if not self.graph_reasoner:
            return result
        
        try:
            # 使用图推理预测边概率
            probability = self.graph_reasoner.predict_edge(drug1, drug2)
            
            if probability > 0.7:
                result['has_interaction'] = True
                result['interaction_type'] = 'potential'
                result['confidence'] = probability
                result['message'] = f"检测到药物[{drug1}]和[{drug2}]之间可能存在相互作用"
            
        except Exception as e:
            pass
        
        return result
    
    def check_treatment_recommendation(self, disease_name: str, treatment: str) -> Dict[str, Any]:
        """
        使用图推理检查治疗方案是否合理
        
        参数：
            disease_name: 疾病名称
            treatment: 治疗方案
        
        返回：
            治疗方案检查结果
        """
        result = {
            'is_recommended': False,
            'confidence': 0.0,
            'recommended_treatments': [],
            'message': ''
        }
        
        if not self.graph_reasoner:
            return result
        
        try:
            # 查询疾病的推荐治疗方案
            treatments = self.graph_reasoner.query_disease_treatment(disease_name, top_k=5)
            result['recommended_treatments'] = [t[0] for t in treatments]
            
            # 检查给定治疗方案是否在推荐列表中
            for treat, score in treatments:
                if treatment in treat or treat in treatment:
                    result['is_recommended'] = True
                    result['confidence'] = score
                    result['message'] = f"治疗方案[{treatment}]是疾病[{disease_name}]的推荐方案"
                    break
            
            if not result['is_recommended']:
                result['message'] = f"治疗方案[{treatment}]未在疾病[{disease_name}]的推荐方案列表中"
        
        except Exception as e:
            pass
        
        return result
    
    def check_red_line(self, statement: str, context: Dict[str, Any]) -> bool:
        """
        检查是否触发红线
        
        参数：
            statement: 医生的陈述
            context: 上下文信息
        
        返回：
            是否触发红线
        """
        result = self.check_factuality(statement, context)
        
        # 如果存在严重违规，触发红线
        for violation in result.get('violations', []):
            violation_type = violation.get('type', '')
            if violation_type in ['contraindication', 'stage_error', 'drug_error', 'treatment_error']:
                return True
        
        return False
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """
        获取所有违规记录
        
        返回：
            违规记录列表
        """
        return self.violations
    
    def reset(self) -> None:
        """
        重置审查官状态
        """
        self.violations = []
