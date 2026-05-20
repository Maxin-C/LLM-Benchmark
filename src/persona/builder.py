import json
import os
from typing import List, Dict
from .extractor import PatientPersonaExtractor


class PersonaBuilder:
    """Persona构建器"""
    
    def __init__(self, output_dir: str = None):
        self.extractor = PatientPersonaExtractor()
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'dataset', 'persona_data'
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def build_personas(self, messages: List[Dict]) -> Dict:
        """
        从消息构建患者Persona
        """
        # 提取患者Persona
        personas = self.extractor.extract_persona(messages)
        
        # 分析对话模式
        patterns = self.extractor.analyze_conversation_patterns(messages)
        
        return {
            'personas': personas,
            'analysis': patterns,
            'total_patients': len(personas),
            'total_messages': len(messages)
        }
    
    def save_personas(self, result: Dict, filename: str = 'personas.json'):
        """
        保存Persona数据到文件
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return filepath
    
    def build_from_files(self, filepaths: List[str]) -> Dict:
        """
        从多个文件构建Persona
        """
        all_messages = []
        
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                all_messages.extend(messages)
        
        return self.build_personas(all_messages)
    
    def generate_knowledge_graph(self, personas: List[Dict]) -> Dict:
        """
        生成患者知识图谱
        """
        graph = {
            'nodes': [],
            'edges': []
        }
        
        node_id_counter = 0
        
        # 添加患者节点
        for persona in personas:
            patient_node = {
                'id': node_id_counter,
                'type': 'patient',
                'name': persona['name'],
                'attributes': {
                    'age': persona['age'],
                    'treatment_stage': persona['treatment_stage'],
                    'message_count': persona['message_count']
                }
            }
            graph['nodes'].append(patient_node)
            patient_node_id = node_id_counter
            node_id_counter += 1
            
            # 添加症状节点和边
            for symptom in persona.get('symptoms', []):
                symptom_node = {
                    'id': node_id_counter,
                    'type': 'symptom',
                    'name': symptom,
                    'attributes': {}
                }
                graph['nodes'].append(symptom_node)
                graph['edges'].append({
                    'source': patient_node_id,
                    'target': node_id_counter,
                    'relation': 'has_symptom'
                })
                node_id_counter += 1
            
            # 添加意图节点和边
            for intent in persona.get('question_intents', []):
                intent_node = {
                    'id': node_id_counter,
                    'type': 'intent',
                    'name': intent,
                    'attributes': {}
                }
                graph['nodes'].append(intent_node)
                graph['edges'].append({
                    'source': patient_node_id,
                    'target': node_id_counter,
                    'relation': 'asks_about'
                })
                node_id_counter += 1
        
        return graph
    
    def save_knowledge_graph(self, graph: Dict, filename: str = 'knowledge_graph.json'):
        """
        保存知识图谱
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        return filepath
