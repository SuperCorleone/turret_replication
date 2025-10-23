import torch
import torch.nn as nn
import torch.distributions as distributions
from typing import Tuple, Optional


class GaussianDistribution:
    """Wrapper for Gaussian distribution used in continuous control"""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    
    def make_distribution(self, mean: torch.Tensor, std: torch.Tensor) -> distributions.Normal:
        """Create Gaussian distribution from mean and std"""
        return distributions.Normal(mean, std)
    
    def sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Sample from Gaussian distribution"""
        distribution = self.make_distribution(mean, std)
        return distribution.rsample()  # Use reparameterization trick
    
    def log_prob(self, actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions"""
        distribution = self.make_distribution(mean, std)
        return distribution.log_prob(actions).sum(dim=-1)
    
    def entropy(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Compute entropy of distribution"""
        distribution = self.make_distribution(mean, std)
        return distribution.entropy().sum(dim=-1)