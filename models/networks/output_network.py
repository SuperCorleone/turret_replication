import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base_networks import BaseNetwork, MLP
from ..components import GaussianDistribution


class OutputNetwork(BaseNetwork):
    """
    Output network that predicts action distributions from state embeddings
    Corresponds to F_out in the paper
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_embedding_dim = config["state_embedding_dim"]
        self.total_action_dim = config["total_action_dim"]  # Sum of all node action dims
        self.learn_std = config.get("learn_std", True)
        
        # MLP for predicting action means
        self.mean_network = MLP({
            "input_dim": self.state_embedding_dim,
            "output_dim": self.total_action_dim,
            "hidden_dims": config.get("hidden_dims", [256, 256]),
            "activation": config.get("activation", "tanh"),
            "device": self.device
        })
        
        # Standard deviation (can be learned or fixed)
        if self.learn_std:
            self.log_std = nn.Parameter(torch.zeros(self.total_action_dim))
        else:
            self.log_std = torch.zeros(self.total_action_dim)
        
        # Action distribution
        self.distribution = GaussianDistribution(self.total_action_dim)
    
    def forward(self, state_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict action distribution from state embedding
        
        Args:
            state_embedding: Tensor of shape [batch_size, state_embedding_dim]
            
        Returns:
            mean: Tensor of shape [batch_size, total_action_dim]
            std: Tensor of shape [batch_size, total_action_dim] or [total_action_dim]
        """
        # Predict action means
        mean = self.mean_network(state_embedding)
        
        # Get standard deviation
        if self.learn_std:
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean).to(mean.device)
        
        return mean, std
    
    def get_action_distribution(self, state_embedding: torch.Tensor) -> GaussianDistribution:
        """Get action distribution for given state embedding"""
        mean, std = self.forward(state_embedding)
        return self.distribution.make_distribution(mean, std)
    
    def split_actions_by_nodes(self, actions: torch.Tensor, 
                             node_action_dims: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Split flattened actions into per-node actions
        
        Args:
            actions: Tensor of shape [batch_size, total_action_dim]
            node_action_dims: Dict mapping node_id -> action_dim
            
        Returns:
            Dict mapping node_id -> action tensor of shape [batch_size, action_dim]
        """
        batch_size = actions.shape[0]
        node_actions = {}
        start_idx = 0
        
        for node_id, action_dim in node_action_dims.items():
            end_idx = start_idx + action_dim
            node_actions[node_id] = actions[:, start_idx:end_idx]
            start_idx = end_idx
        
        return node_actions