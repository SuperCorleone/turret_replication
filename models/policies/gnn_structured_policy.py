import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# 使用相对导入
from ...models.networks import InputNetwork, OutputNetwork
from ...models.networks.attention_propagation import MultiHeadAttentionLayer
from ...models.networks.set_transformer import MultiHeadSetTransformer
from ...models.morphology import MorphologyGraph, Node, Edge, StandardRobotMorphology


class GNNStructuredPolicyNetwork(nn.Module):
    """完整的GNN结构化策略网络 - Phase 6+ 版本"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.device = config.get("device", "cpu")
        
        # 网络维度 - 带默认值
        self.node_observation_dim = config.get("node_observation_dim", 17)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.output_dim = config.get("output_dim", 6)
        self.n_gnn_layers = config.get("n_gnn_layers", 3)
        self.n_attention_heads = config.get("n_attention_heads", 8)
        
        # 稳定性参数
        self.gradient_clip_norm = config.get("gradient_clip_norm", 1.0)
        self.action_clip = config.get("action_clip", 10.0)
        
        # 初始化组件
        self._init_input_network()
        self._init_gnn_layers()
        self._init_output_network()
        
        # 状态嵌入
        self.set_transformer = MultiHeadSetTransformer({
            "input_dim": self.hidden_dim,
            "hidden_dim": 128,
            "output_dim": 128,
            "n_heads": 4,
            "n_blocks": 2
        })
    
    def _init_input_network(self):
        """初始化输入网络"""
        self.input_network = InputNetwork({
            "node_observation_dim": self.node_observation_dim,
            "hidden_dim": self.hidden_dim,
            "shared_across_nodes": True,
            "device": self.device
        })
    
    def _init_gnn_layers(self):
        """初始化GNN层"""
        self.gnn_layers = nn.ModuleList()
        
        for layer_idx in range(self.n_gnn_layers):
            layer_config = {
                "hidden_dim": self.hidden_dim,
                "n_heads": self.n_attention_heads,
                "edge_feature_dim": 4,  # 简化的边特征维度
                "dropout": 0.1
            }
            self.gnn_layers.append(MultiHeadAttentionLayer(layer_config))
    
    def _init_output_network(self):
        """初始化输出网络"""
        self.output_network = OutputNetwork({
            "state_embedding_dim": 128,
            "total_action_dim": self.output_dim,
            "learn_std": True,
            "device": self.device
        })
    
    def forward(self, 
                node_observations: Dict[str, torch.Tensor],
                morphology_graph: MorphologyGraph) -> tuple:
        """
        完整的GNN前向传播
        
        Args:
            node_observations: 节点观察字典
            morphology_graph: 形态学图
            
        Returns:
            mean: 动作均值
            std: 动作标准差
        """
        # 1. 输入处理
        initial_representations = self.input_network(node_observations)
        
        # 2. 准备GNN输入
        node_states, adjacency_list = self._prepare_gnn_input(initial_representations, morphology_graph)
        
        # 3. GNN传播
        node_states = self._apply_gnn_propagation(node_states, adjacency_list, morphology_graph)
        
        # 4. 状态嵌入
        state_embedding = self._compute_state_embedding(node_states, morphology_graph)
        
        # 5. 输出动作分布
        mean, std = self.output_network(state_embedding)
        
        return mean, std
    
    def _prepare_gnn_input(self, 
                          node_representations: Dict[str, torch.Tensor],
                          morphology_graph: MorphologyGraph) -> Tuple[torch.Tensor, List[List[int]]]:
        """准备GNN输入"""
        # 节点排序
        node_ids = sorted(node_representations.keys())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # 节点状态矩阵
        node_states = torch.stack([node_representations[node_id] for node_id in node_ids])
        
        # 构建邻接表
        adjacency_list = [[] for _ in range(len(node_ids))]
        
        for edge in morphology_graph.edges:
            source_idx = node_to_idx.get(edge.source.node_id)
            target_idx = node_to_idx.get(edge.target.node_id)
            if source_idx is not None and target_idx is not None:
                adjacency_list[target_idx].append(source_idx)  # 目标节点关注源节点
        
        return node_states, adjacency_list
    
    def _apply_gnn_propagation(self,
                             node_states: torch.Tensor,
                             adjacency_list: List[List[int]],
                             morphology_graph: MorphologyGraph) -> torch.Tensor:
        """应用GNN传播"""
        current_states = node_states
        
        for gnn_layer in self.gnn_layers:
            current_states = gnn_layer(current_states, adjacency_list)
        
        return current_states
    
    def _compute_state_embedding(self,
                               node_states: torch.Tensor,
                               morphology_graph: MorphologyGraph) -> torch.Tensor:
        """计算状态嵌入"""
        # 将节点状态转换为矩阵格式 [batch_size, num_nodes, hidden_dim]
        node_matrix = node_states.unsqueeze(0)  # 添加batch维度
        
        # 通过Set Transformer
        state_embedding = self.set_transformer(node_matrix)
        
        return state_embedding.squeeze(0)  # 移除batch维度
    
    def get_node_representations(self,
                               node_observations: Dict[str, torch.Tensor],
                               morphology_graph: MorphologyGraph) -> Dict[str, torch.Tensor]:
        """获取节点表示（用于迁移学习）"""
        with torch.no_grad():
            initial_reps = self.input_network(node_observations)
            node_states, adjacency_list = self._prepare_gnn_input(initial_reps, morphology_graph)
            final_reps = self._apply_gnn_propagation(node_states, adjacency_list, morphology_graph)
            
            # 转换回字典格式
            node_ids = sorted(node_observations.keys())
            node_representations = {}
            for idx, node_id in enumerate(node_ids):
                node_representations[node_id] = final_reps[idx]
            
            return node_representations
    
    def compute_state_embedding(self,
                              node_observations: Dict[str, torch.Tensor],
                              morphology_graph: MorphologyGraph) -> torch.Tensor:
        """计算状态嵌入（公开接口）"""
        mean, std = self.forward(node_observations, morphology_graph)
        
        # 重新计算状态嵌入（简化实现）
        initial_reps = self.input_network(node_observations)
        node_states, adjacency_list = self._prepare_gnn_input(initial_reps, morphology_graph)
        state_embedding = self._compute_state_embedding(node_states, morphology_graph)
        
        return state_embedding