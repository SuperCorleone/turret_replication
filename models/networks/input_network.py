import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import numpy as np

class InputNetwork(nn.Module):
    """
    Input network that processes node observations (F_in in the paper)
    Transforms raw node observations to initial node representations
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.max_observation_dim = config["node_observation_dim"]  # 改为最大观察维度
        self.hidden_dim = config["hidden_dim"]
        self.shared_across_nodes = config.get("shared_across_nodes", True)
        self.device = config.get("device", "cpu")
        
        # 创建MLP用于处理节点观察
        mlp_config = {
            "input_dim": self.max_observation_dim,  # 使用最大维度
            "output_dim": self.hidden_dim,
            "hidden_dims": config.get("hidden_dims", [128]),
            "activation": config.get("activation", "relu"),
            "device": self.device
        }
        
        self.node_mlp = self._create_mlp(mlp_config)
        
        # 可选：不同类型节点的不同网络
        self.node_type_networks = None
        if not self.shared_across_nodes and "node_types" in config:
            self.node_type_networks = nn.ModuleDict()
            for node_type in config["node_types"]:
                self.node_type_networks[node_type] = self._create_mlp(mlp_config)
    
    def _create_mlp(self, mlp_config: Dict[str, Any]) -> nn.Module:
        """创建MLP网络"""
        from .base_networks import MLP
        return MLP(mlp_config)
    
    def forward(self, node_observations: Dict[str, torch.Tensor], 
                node_types: Dict[str, str] = None) -> Dict[str, torch.Tensor]:
        """
        Process node observations to initial representations
        
        Args:
            node_observations: Dict mapping node_id -> observation tensor
            node_types: Dict mapping node_id -> node_type (if using type-specific networks)
        
        Returns:
            Dict mapping node_id -> initial representation
        """
        initial_representations = {}
        
        for node_id, observation in node_observations.items():
            # 选择适当的网络
            if (self.node_type_networks is not None and 
                node_types is not None and 
                node_types.get(node_id) in self.node_type_networks):
                mlp = self.node_type_networks[node_types[node_id]]
            else:
                mlp = self.node_mlp
            
            # 确保观察在正确的设备上并有正确的形状
            observation = observation.to(self.device)
            
            # 如果观察维度小于最大维度，进行填充
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)  # 添加batch维度
            
            obs_dim = observation.shape[-1]
            if obs_dim < self.max_observation_dim:
                # 填充到最大维度
                padding = torch.zeros(observation.shape[0], 
                                    self.max_observation_dim - obs_dim, 
                                    device=self.device)
                observation = torch.cat([observation, padding], dim=-1)
            elif obs_dim > self.max_observation_dim:
                # 截断到最大维度
                observation = observation[:, :self.max_observation_dim]
            
            representation = mlp(observation)
            initial_representations[node_id] = representation.squeeze(0)  # 移除batch维度
        
        return initial_representations
    
    def process_batch(self, batch_observations: torch.Tensor) -> torch.Tensor:
        """
        Process batch of node observations (for efficiency)
        
        Args:
            batch_observations: Tensor of shape [batch_size, num_nodes, observation_dim]
        
        Returns:
            Tensor of shape [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, obs_dim = batch_observations.shape
        
        # 确保维度匹配
        if obs_dim < self.max_observation_dim:
            padding = torch.zeros(batch_size, num_nodes, 
                                self.max_observation_dim - obs_dim,
                                device=self.device)
            batch_observations = torch.cat([batch_observations, padding], dim=-1)
        elif obs_dim > self.max_observation_dim:
            batch_observations = batch_observations[:, :, :self.max_observation_dim]
        
        # 重塑以一次性处理所有节点
        flat_observations = batch_observations.reshape(-1, self.max_observation_dim)
        flat_representations = self.node_mlp(flat_observations)
        
        # 重塑回原始维度
        representations = flat_representations.reshape(batch_size, num_nodes, -1)
        return representations