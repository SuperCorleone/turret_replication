import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import math

class MultiHeadAttentionLayer(nn.Module):
    """完整的多头注意力传播层"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.n_heads = config["n_heads"]
        self.head_dim = self.hidden_dim // self.n_heads
        self.edge_feature_dim = config.get("edge_feature_dim", 0)
        
        # 线性变换
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # 边特征处理
        if self.edge_feature_dim > 0:
            self.W_e = nn.Linear(self.edge_feature_dim, self.n_heads)
        
        # 输出变换
        self.W_o = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # 层归一化和Dropout
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        
        # Feed-forward网络
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, 
                node_states: torch.Tensor,
                adjacency_list: List[List[int]],
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_states: [num_nodes, hidden_dim]
            adjacency_list: 邻接表
            edge_features: [num_edges, edge_feature_dim]
        """
        batch_size = 1  # 单图处理
        num_nodes = node_states.size(0)
        
        # 残差连接1
        residual = node_states
        
        # 线性变换
        Q = self.W_q(node_states)  # [num_nodes, hidden_dim]
        K = self.W_k(node_states)  # [num_nodes, hidden_dim]
        V = self.W_v(node_states)  # [num_nodes, hidden_dim]
        
        # 多头分割
        Q = Q.view(num_nodes, self.n_heads, self.head_dim).transpose(0, 1)  # [n_heads, num_nodes, head_dim]
        K = K.view(num_nodes, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(num_nodes, self.n_heads, self.head_dim).transpose(0, 1)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [n_heads, num_nodes, num_nodes]
        
        # 应用邻接矩阵掩码
        attention_mask = self._create_attention_mask(adjacency_list, num_nodes)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # 应用边特征（如果存在）
        if edge_features is not None and self.edge_feature_dim > 0:
            edge_attention = self.W_e(edge_features)  # [num_edges, n_heads]
            attention_scores = self._apply_edge_features(attention_scores, edge_attention, adjacency_list)
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [n_heads, num_nodes, num_nodes]
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        node_outputs = torch.matmul(attention_weights, V)  # [n_heads, num_nodes, head_dim]
        node_outputs = node_outputs.transpose(0, 1).contiguous().view(num_nodes, -1)  # [num_nodes, hidden_dim]
        node_outputs = self.W_o(node_outputs)
        
        # 残差连接和层归一化
        node_outputs = self.layer_norm1(node_outputs + residual)
        
        # Feed-forward网络
        residual_ffn = node_outputs
        node_outputs = self.ffn(node_outputs)
        node_outputs = self.layer_norm2(node_outputs + residual_ffn)
        
        return node_outputs
    
    def _create_attention_mask(self, adjacency_list: List[List[int]], num_nodes: int) -> torch.Tensor:
        """创建注意力掩码"""
        mask = torch.zeros(self.n_heads, num_nodes, num_nodes, dtype=torch.bool)
        
        for target_idx in range(num_nodes):
            neighbors = adjacency_list[target_idx]
            for source_idx in neighbors:
                mask[:, target_idx, source_idx] = 1
        
        return mask
    
    def _apply_edge_features(self, 
                           attention_scores: torch.Tensor,
                           edge_attention: torch.Tensor,
                           adjacency_list: List[List[int]]) -> torch.Tensor:
        """应用边特征到注意力分数"""
        # 简化实现：为每个注意力头添加边特征偏置
        edge_bias = torch.zeros_like(attention_scores)
        edge_idx = 0
        
        for target_idx in range(attention_scores.size(1)):
            neighbors = adjacency_list[target_idx]
            for source_idx in neighbors:
                if edge_idx < edge_attention.size(0):
                    edge_bias[:, target_idx, source_idx] = edge_attention[edge_idx]
                    edge_idx += 1
        
        return attention_scores + edge_bias