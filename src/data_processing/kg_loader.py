"""
知识图谱加载器
负责加载和管理CMeKG医疗知识图谱
"""

import pickle
import networkx as nx
from typing import Dict, Any, List, Tuple

class KnowledgeGraphLoader:
    def __init__(self):
        self.graph = None
        self.node_types = set()
        self.edge_types = set()
    
    def load_pkl(self, file_path: str) -> None:
        """
        从pickle文件加载知识图谱
        
        参数：
            file_path: 知识图谱pickle文件路径
        """
        with open(file_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # 提取节点和边类型
        if isinstance(self.graph, nx.Graph):
            for node, data in self.graph.nodes(data=True):
                self.node_types.add(data.get('type', 'unknown'))
            
            for _, _, data in self.graph.edges(data=True):
                self.edge_types.add(data.get('relation', 'unknown'))
    
    def load_json(self, file_path: str) -> None:
        """
        从JSON文件加载知识图谱
        
        参数：
            file_path: 知识图谱JSON文件路径
        """
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.graph = nx.Graph()
        
        # 添加节点
        for node in data.get('nodes', []):
            node_id = node.get('id')
            node_type = node.get('type', 'unknown')
            attributes = {k: v for k, v in node.items() if k not in ['id', 'type']}
            self.graph.add_node(node_id, type=node_type, **attributes)
            self.node_types.add(node_type)
        
        # 添加边
        for edge in data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            relation = edge.get('relation', 'related_to')
            self.graph.add_edge(source, target, relation=relation)
            self.edge_types.add(relation)
    
    def get_node_info(self, node_id: str) -> Dict[str, Any]:
        """
        获取节点信息
        
        参数：
            node_id: 节点ID
        
        返回：
            节点属性字典
        """
        if self.graph is None:
            raise ValueError("知识图谱未加载")
        
        if node_id in self.graph.nodes:
            return dict(self.graph.nodes[node_id])
        else:
            return {}
    
    def get_related_nodes(self, node_id: str, relation_type: str = None) -> List[str]:
        """
        获取相关节点
        
        参数：
            node_id: 节点ID
            relation_type: 关系类型（可选）
        
        返回：
            相关节点ID列表
        """
        if self.graph is None:
            raise ValueError("知识图谱未加载")
        
        related_nodes = []
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            if relation_type is None or edge_data.get('relation') == relation_type:
                related_nodes.append(neighbor)
        
        return related_nodes
    
    def query_drug_interactions(self, drug_name: str) -> List[Tuple[str, str]]:
        """
        查询药物相互作用
        
        参数：
            drug_name: 药物名称
        
        返回：
            相互作用药物列表（药物名, 相互作用类型）
        """
        interactions = []
        related_nodes = self.get_related_nodes(drug_name)
        
        for node in related_nodes:
            edge_data = self.graph.get_edge_data(drug_name, node)
            relation = edge_data.get('relation', '')
            
            if 'interaction' in relation.lower():
                interactions.append((node, relation))
        
        return interactions
    
    def get_disease_treatments(self, disease_name: str) -> List[str]:
        """
        获取疾病的治疗方案
        
        参数：
            disease_name: 疾病名称
        
        返回：
            治疗方案列表
        """
        treatments = []
        related_nodes = self.get_related_nodes(disease_name)
        
        for node in related_nodes:
            node_data = self.get_node_info(node)
            if node_data.get('type') == 'treatment':
                treatments.append(node)
        
        return treatments
    
    def get_node_types(self) -> set:
        """
        获取所有节点类型
        
        返回：
            节点类型集合
        """
        return self.node_types
    
    def get_edge_types(self) -> set:
        """
        获取所有边类型
        
        返回：
            边类型集合
        """
        return self.edge_types
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取图谱统计信息
        
        返回：
            统计信息字典
        """
        if self.graph is None:
            return {}
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_node_types': len(self.node_types),
            'num_edge_types': len(self.edge_types)
        }
