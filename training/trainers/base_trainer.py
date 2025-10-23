import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

# 添加导入
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from configs.base_config import TURRETConfig

class BaseTrainer(ABC):
    """
    Abstract base class for all trainers
    """
    
    def __init__(self, config: TURRETConfig):  # 明确类型注解
        self.config = config
        self.device = config.device if hasattr(config, 'device') else config.get("device", "cpu")
        self.total_steps = 0
        self.episode_count = 0
        self.best_reward = -float('inf')
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy_losses': [],
            'total_losses': [],
        }
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step"""
        pass
    
    @abstractmethod
    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute losses for a batch of data"""
        pass
    
    @abstractmethod
    def update_parameters(self, losses: Dict[str, torch.Tensor]) -> None:
        """Update model parameters based on computed losses"""
        pass
    
    def log_episode(self, episode_reward: float, episode_length: int) -> None:
        """Log episode statistics"""
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        self.episode_count += 1
        
        # Update best reward
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        stats = {}
        
        for key, values in self.training_stats.items():
            if values:
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_std'] = float(np.std(values))
                stats[f'{key}_current'] = float(values[-1])
        
        stats['total_steps'] = self.total_steps
        stats['episode_count'] = self.episode_count
        stats['best_reward'] = self.best_reward
        
        return stats
    
    def clear_episode_stats(self) -> None:
        """Clear episode statistics (for new logging period)"""
        for key in self.training_stats.keys():
            self.training_stats[key] = []
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save trainer checkpoint"""
        raise NotImplementedError("Subclasses must implement save_checkpoint")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load trainer checkpoint"""
        raise NotImplementedError("Subclasses must implement load_checkpoint")