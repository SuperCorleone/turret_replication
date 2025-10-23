import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import numpy as np

from ..networks import InputNetwork, OutputNetwork
from ..networks.set_transformer import MultiHeadSetTransformer


class StructuredPolicyNetwork(nn.Module):
    """
    Structured policy network using GNN for robot control
    Implements the policy network architecture from TURRET paper
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.device = config.get("device", "cpu")
        
        # Network dimensions
        self.node_observation_dim = config["node_observation_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.output_dim = config["output_dim"]
        self.shared_across_nodes = config.get("shared_across_nodes", True)
        
        # 添加数值稳定性保护
        self.gradient_clip_norm = config.get("gradient_clip_norm", 1.0)
        self.action_clip = config.get("action_clip", 10.0)
        
        # Initialize components
        self._init_input_network()
        self._init_propagation_network()
        self._init_output_network()
        
        # For state embedding (used in transfer)
        self.set_transformer = MultiHeadSetTransformer({
            "input_dim": self.hidden_dim,
            "hidden_dim": 128,
            "output_dim": 128,
            "n_heads": 4,
            "n_blocks": 2,
            "n_induction_points": 4
        })
    
    def _init_input_network(self) -> None:
        """Initialize input network for processing node observations"""
        self.input_network = InputNetwork({
            "node_observation_dim": self.node_observation_dim,
            "hidden_dim": self.hidden_dim,
            "shared_across_nodes": self.shared_across_nodes,
            "device": self.device
        })
    
    def _init_propagation_network(self) -> None:
        """Initialize GNN propagation network"""
        # For Phase 5, we use a simplified MLP-based propagation
        # In Phase 6, this will be replaced with full GNN attention mechanism
        self.propagation_network = nn.Sequential(
            nn.Linear(self.node_observation_dim, self.hidden_dim),  # 修复：使用 node_observation_dim
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
    
    def _init_output_network(self) -> None:
        """Initialize output network for action prediction"""
        self.output_network = OutputNetwork({
            "state_embedding_dim": 128,  # From set transformer
            "total_action_dim": self.output_dim,
            "learn_std": True,
            "device": self.device
        })
        
        # For simplified version without set transformer
        self.simple_output = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim * 2)  # Mean and log_std
        )
    
    def forward(self, observations: torch.Tensor) -> tuple:
        """
        Forward pass of structured policy network
        
        Args:
            observations: Input observations [batch_size, obs_dim]
            
        Returns:
            mean: Action means [batch_size, action_dim]
            std: Action standard deviations [batch_size, action_dim]
        """
        batch_size = observations.shape[0]
        
        # 输入验证和清理
        if torch.isnan(observations).any():
            observations = torch.nan_to_num(observations, nan=0.0)
        
        # 使用更稳定的网络结构
        features = self.propagation_network(observations)
        
        # 梯度裁剪保护
        if features.requires_grad:
            features.register_hook(lambda grad: torch.clamp(grad, -self.gradient_clip_norm, self.gradient_clip_norm))
        
        # 输出层带稳定性保护
        output = self.simple_output(features)
        
        # 分离均值和标准差计算
        mean = torch.tanh(output[:, :self.output_dim])
        log_std = output[:, self.output_dim:]
        
        # 限制标准差范围防止数值问题
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        # 最终输出验证
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("Warning: NaN detected in policy output, using safe fallback")
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std) * 0.1
        
        return mean, std
    
    def get_node_representations(self, 
                               node_observations: Dict[str, torch.Tensor],
                               morphology_graph: Any) -> Dict[str, torch.Tensor]:
        """
        Get node representations from GNN (simplified for Phase 5)
        
        Args:
            node_observations: Dictionary of node observations
            morphology_graph: Robot morphology graph
            
        Returns:
            Dictionary of node representations
        """
        # For Phase 5, return simplified representations
        # In Phase 6, this will implement full GNN propagation
        
        representations = {}
        for node_id, observation in node_observations.items():
            # Simple processing - in Phase 6 this will use actual GNN
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            representation = self.propagation_network(observation)
            representations[node_id] = representation.squeeze(0)
        
        return representations
    
    def compute_state_embedding(self,
                              node_representations: Dict[str, torch.Tensor],
                              morphology_graph: Any) -> torch.Tensor:
        """
        Compute global state embedding using Set Transformer
        
        Args:
            node_representations: Dictionary of node representations
            morphology_graph: Robot morphology graph
            
        Returns:
            Global state embedding
        """
        # Convert node representations to matrix
        node_ids = sorted(node_representations.keys())
        node_matrix = torch.stack([node_representations[node_id] for node_id in node_ids])
        
        # Add batch dimension if needed
        if node_matrix.dim() == 2:
            node_matrix = node_matrix.unsqueeze(0)
        
        # Get state embedding through Set Transformer
        state_embedding = self.set_transformer(node_matrix)
        
        return state_embedding
    
    def get_layer_activations(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate layer activations for transfer learning
        
        Args:
            observations: Input observations
            
        Returns:
            Dictionary of layer activations
        """
        activations = {}
        
        # Input processing
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        
        # Get activations from each component
        activations['input'] = observations
        
        # Propagation network activations
        features = self.propagation_network(observations)
        activations['propagation'] = features
        
        # Output activations
        output = self.simple_output(features)
        activations['output'] = output
        
        return activations
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)