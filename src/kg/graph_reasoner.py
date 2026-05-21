"""
图推理引擎
基于GATv2模型实现知识图谱推理功能
使用CMeKG-RAG项目中的训练好的模型进行推理
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Dict, Any, List, Tuple, Optional

class JK_GATv2(nn.Module):
    """
    JK-GATv2模型实现
    采用跳跃连接(Jump Knowledge)策略融合多层特征
    """
    def __init__(self, in_dim, hidden, out_dim, heads, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.dropout = dropout
        
        self.conv1 = GATv2Conv(hidden, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden*heads, hidden, heads=heads, dropout=dropout)
        
        self.jk_proj = nn.Linear(hidden*heads*2, hidden)
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        x0 = F.elu(self.input_proj(x))
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        
        x1 = F.elu(self.conv1(x0, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        x2 = F.elu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        jk = torch.cat([x1, x2], dim=-1)
        jk = self.jk_proj(jk)
        
        emb = self.out_proj(jk)
        return F.normalize(emb, p=2, dim=-1)

class GraphReasoner:
    """
    图推理引擎类
    基于预训练的GATv2模型进行知识图谱推理
    """
    
    def __init__(self, model_path: str, kg_path: str, embedding_cache_path: str):
        """
        初始化图推理引擎
        
        参数：
            model_path: 预训练模型路径
            kg_path: 知识图谱pickle文件路径
            embedding_cache_path: 节点嵌入缓存路径
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.graph_data = None
        self.node_id_to_idx = None
        self.edge_index = None
        self.node_embeddings = None
        self.x_raw = None
        
        # 加载数据
        self._load_kg(kg_path)
        self._load_embeddings(embedding_cache_path)
        self._load_model(model_path)
    
    def _load_kg(self, kg_path: str):
        """加载知识图谱数据"""
        with open(kg_path, 'rb') as f:
            self.graph_data = pickle.load(f)
        
        # 构建节点ID到索引的映射
        self.node_id_to_idx = {n['id']: i for i, n in enumerate(self.graph_data['nodes'])}
        
        # 构建边索引
        self.edge_index = torch.tensor(
            [[self.node_id_to_idx[e['src']], self.node_id_to_idx[e['dst']]] 
             for e in self.graph_data['edges']],
            dtype=torch.long
        ).t().contiguous().to(self.device)
    
    def _load_embeddings(self, embedding_cache_path: str):
        """加载节点嵌入向量"""
        self.x_raw = torch.from_numpy(np.load(embedding_cache_path)).float().to(self.device)
    
    def _load_model(self, model_path: str):
        """加载预训练模型"""
        # 模型配置
        config = {
            "hidden_dim": 256,
            "out_dim": 256,
            "heads": 4,
            "dropout": 0.3,
            "in_dim": 1024  # BGE embedding dimension
        }
        
        # 创建模型
        self.model = JK_GATv2(
            in_dim=config['in_dim'],
            hidden=config['hidden_dim'],
            out_dim=config['out_dim'],
            heads=config['heads'],
            dropout=config['dropout']
        ).to(self.device)
        
        # 加载预训练权重
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 预计算所有节点的嵌入
        self._compute_all_embeddings()
    
    def _compute_all_embeddings(self):
        """预计算所有节点的图嵌入"""
        with torch.no_grad():
            self.node_embeddings = self.model(self.x_raw, self.edge_index)
    
    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        获取节点的图嵌入向量
        
        参数：
            node_id: 节点ID
        
        返回：
            节点嵌入向量（numpy数组），如果节点不存在返回None
        """
        if node_id not in self.node_id_to_idx:
            return None
        
        idx = self.node_id_to_idx[node_id]
        return self.node_embeddings[idx].cpu().numpy()
    
    def get_similar_nodes(self, node_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        查找相似节点（基于图嵌入的余弦相似度）
        
        参数：
            node_id: 目标节点ID
            top_k: 返回前k个相似节点
        
        返回：
            相似节点列表，每个元素为(节点ID, 相似度分数)
        """
        if node_id not in self.node_id_to_idx:
            return []
        
        target_idx = self.node_id_to_idx[node_id]
        target_emb = self.node_embeddings[target_idx]
        
        # 计算余弦相似度
        similarities = F.cosine_similarity(
            target_emb.unsqueeze(0), 
            self.node_embeddings, 
            dim=1
        )
        
        # 获取top_k相似节点（排除自身）
        values, indices = torch.topk(similarities, top_k + 1)
        
        results = []
        for val, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
            if idx != target_idx:  # 排除自身
                node_id = [k for k, v in self.node_id_to_idx.items() if v == idx][0]
                results.append((node_id, float(val)))
        
        return results
    
    def predict_edge(self, src_id: str, dst_id: str) -> float:
        """
        预测两个节点之间存在边的概率（链接预测）
        
        参数：
            src_id: 源节点ID
            dst_id: 目标节点ID
        
        返回：
            边存在的概率（0-1之间）
        """
        if src_id not in self.node_id_to_idx or dst_id not in self.node_id_to_idx:
            return 0.0
        
        src_idx = self.node_id_to_idx[src_id]
        dst_idx = self.node_id_to_idx[dst_id]
        
        src_emb = self.node_embeddings[src_idx]
        dst_emb = self.node_embeddings[dst_idx]
        
        # 余弦相似度作为边存在概率的近似
        similarity = F.cosine_similarity(src_emb.unsqueeze(0), dst_emb.unsqueeze(0))
        return float(similarity)
    
    def reasoning_path(self, start_id: str, target_id: str, max_hops: int = 3) -> List[List[str]]:
        """
        查找从起始节点到目标节点的推理路径
        
        参数：
            start_id: 起始节点ID
            target_id: 目标节点ID
            max_hops: 最大跳数
        
        返回：
            推理路径列表，每条路径是节点ID的序列
        """
        if start_id not in self.node_id_to_idx or target_id not in self.node_id_to_idx:
            return []
        
        paths = []
        visited = set()
        
        def dfs(current, path, hops):
            if hops > max_hops:
                return
            
            visited.add(current)
            
            if current == target_id:
                paths.append(path.copy())
                visited.remove(current)
                return
            
            # 获取当前节点的邻居
            neighbors = self._get_neighbors(current)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, path, hops + 1)
                    path.pop()
            
            visited.remove(current)
        
        dfs(start_id, [start_id], 0)
        return paths
    
    def _get_neighbors(self, node_id: str) -> List[str]:
        """获取节点的邻居节点"""
        neighbors = []
        for edge in self.graph_data.get('edges', []):
            if edge['src'] == node_id:
                neighbors.append(edge['dst'])
            elif edge['dst'] == node_id:
                neighbors.append(edge['src'])
        return list(set(neighbors))  # 去重
    
    def query_disease_treatment(self, disease_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查询疾病的治疗方案（基于图推理）
        
        参数：
            disease_name: 疾病名称
            top_k: 返回前k个治疗方案
        
        返回：
            治疗方案列表，每个元素为(治疗方案名称, 相关性分数)
        """
        # 先找到疾病节点的相似节点
        similar_nodes = self.get_similar_nodes(disease_name, top_k * 2)
        
        # 过滤出治疗类型的节点
        treatments = []
        for node_id, score in similar_nodes:
            node_info = self._get_node_info(node_id)
            if node_info.get('type') == 'treatment' or '治疗' in node_id:
                treatments.append((node_id, score))
        
        return treatments[:top_k]
    
    def query_drug_interaction(self, drug_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查询药物相互作用（基于图推理）
        
        参数：
            drug_name: 药物名称
            top_k: 返回前k个相互作用药物
        
        返回：
            相互作用药物列表，每个元素为(药物名称, 相互作用强度)
        """
        # 查找相似节点
        similar_nodes = self.get_similar_nodes(drug_name, top_k * 2)
        
        # 过滤出药物类型的节点
        interactions = []
        for node_id, score in similar_nodes:
            node_info = self._get_node_info(node_id)
            if node_info.get('type') == 'drug' or '药物' in node_id or '药' in node_id:
                if node_id != drug_name:
                    interactions.append((node_id, score))
        
        return interactions[:top_k]
    
    def _get_node_info(self, node_id: str) -> Dict[str, Any]:
        """获取节点详细信息"""
        for node in self.graph_data.get('nodes', []):
            if node['id'] == node_id:
                return node
        return {}
    
    def get_node_by_name(self, name: str) -> Optional[str]:
        """
        根据名称查找节点ID
        
        参数：
            name: 节点名称（部分匹配）
        
        返回：
            匹配的节点ID，如果未找到返回None
        """
        for node in self.graph_data.get('nodes', []):
            node_id = node.get('id', '')
            if name in node_id or node_id in name:
                return node_id
        return None
    
    def reasoning_chain(self, query: str) -> Dict[str, Any]:
        """
        执行复杂推理链
        
        参数：
            query: 查询语句
        
        返回：
            推理结果字典，包含推理路径和结论
        """
        # 简单实现：解析查询中的实体，执行推理
        result = {
            'query': query,
            'entities': [],
            'relations': [],
            'paths': [],
            'conclusion': ''
        }
        
        # 提取实体（简化实现）
        for node in self.graph_data.get('nodes', []):
            node_id = node.get('id', '')
            if node_id in query:
                result['entities'].append(node_id)
        
        # 如果找到两个实体，尝试找路径
        if len(result['entities']) >= 2:
            paths = self.reasoning_path(result['entities'][0], result['entities'][1])
            result['paths'] = paths
        
        return result
