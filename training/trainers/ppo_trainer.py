import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional
import numpy as np

from .base_trainer import BaseTrainer
from ..optimizers import LearningRateScheduler, GradientManager, PPOScheduler
from ..buffers import ExperienceBuffer, TrajectoryBuffer

class PPOTrainer(BaseTrainer):
    """
    Proximal Policy Optimization (PPO) trainer
    Implements the PPO algorithm from the paper
    """
    
    def __init__(self, 
                 policy_network: nn.Module,
                 value_network: nn.Module,
                 config):  # 接受任何配置类型，内部处理
        # 统一配置处理
        if hasattr(config, '__dict__'):  # TURRETConfig dataclass
            config_dict = config.__dict__
        else:  # 普通字典
            config_dict = config
            
        super().__init__(config_dict)
        
        self.policy_network = policy_network
        self.value_network = value_network
        
        # PPO hyperparameters - 统一访问方式
        self.ppo_epochs = config_dict.get("ppo_epochs", 10)
        self.mini_batch_size = config_dict.get("mini_batch_size", 64)
        self.clip_epsilon = config_dict.get("clip_epsilon", 0.2)
        self.value_coef = config_dict.get("value_coef", 0.5)
        self.entropy_coef = config_dict.get("entropy_coef", 0.01)
        self.max_grad_norm = config_dict.get("max_grad_norm", 0.5)
        self.target_kl = config_dict.get("target_kl", 0.01)
        
        # Optimizers
        learning_rate = config_dict.get("learning_rate", 3e-4)
        self.policy_optimizer = optim.Adam(
            policy_network.parameters(),
            lr=learning_rate,
            eps=1e-5
        )
        self.value_optimizer = optim.Adam(
            value_network.parameters(),
            lr=learning_rate,
            eps=1e-5
        )
        
        # Learning rate schedulers
        self.policy_scheduler = LearningRateScheduler(
            self.policy_optimizer, config_dict
        )
        self.value_scheduler = LearningRateScheduler(
            self.value_optimizer, config_dict
        )
        
        # PPO-specific scheduler
        self.ppo_scheduler = PPOScheduler(config_dict)
        
        # Gradient monitor
        self.gradient_manager = GradientManager()
        
        # Buffers
        self.experience_buffer = ExperienceBuffer(
            capacity=config_dict.get("buffer_size", 10000),
            observation_shape=config_dict.get("observation_shape", (10,)),
            action_shape=config_dict.get("action_shape", (4,))
        )
        
        self.trajectory_buffer = TrajectoryBuffer(
            gamma=config_dict.get("gamma", 0.99),
            gae_lambda=config_dict.get("gae_lambda", 0.95)
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a PPO training step"""
        
        # Move batch to device
        device_batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Perform multiple PPO epochs
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        clip_fractions = []
        
        for epoch in range(self.ppo_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(device_batch['observations'].size(0))
            
            for start in range(0, indices.size(0), self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indices = indices[start:end]
                
                mini_batch = {}
                for key, value in device_batch.items():
                    mini_batch[key] = value[mini_batch_indices]
                
                # Compute losses
                losses = self.compute_losses(mini_batch)
                
                # Update parameters
                self.update_parameters(losses)
                
                # Store losses for logging
                policy_losses.append(losses['policy_loss'].item())
                value_losses.append(losses['value_loss'].item())
                entropy_losses.append(losses['entropy_loss'].item())
                total_losses.append(losses['total_loss'].item())
                
                if 'approx_kl' in losses:
                    approx_kls.append(losses['approx_kl'].item())
                if 'clip_fraction' in losses:
                    clip_fractions.append(losses['clip_fraction'].item())
        
        # Update learning rates
        self.policy_scheduler.step()
        self.value_scheduler.step()
        self.ppo_scheduler.step()
        
        # Update step count
        self.total_steps += 1
        
        # Return statistics
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(total_losses),
            'policy_lr': self.policy_scheduler.get_current_lr(),
            'value_lr': self.value_scheduler.get_current_lr(),
            'clip_epsilon': self.ppo_scheduler.get_clip_epsilon(),
        }
        
        if approx_kls:
            stats['approx_kl'] = np.mean(approx_kls)
        if clip_fractions:
            stats['clip_fraction'] = np.mean(clip_fractions)
        
        # Store for training statistics
        self.training_stats['policy_losses'].append(stats['policy_loss'])
        self.training_stats['value_losses'].append(stats['value_loss'])
        self.training_stats['entropy_losses'].append(stats['entropy_loss'])
        self.training_stats['total_losses'].append(stats['total_loss'])
        
        return stats
    
    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute PPO losses for a batch of data"""
        
        observations = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch.get('advantages', torch.ones_like(batch['rewards']))
        returns = batch.get('returns', batch['rewards'])
        
        # Get current policy distribution
        current_mean, current_std = self.policy_network(observations)
        current_dist = torch.distributions.Normal(current_mean, current_std)
        
        # Compute new log probabilities and entropy
        new_log_probs = current_dist.log_prob(actions).sum(dim=-1)
        entropy = current_dist.entropy().sum(dim=-1)
        
        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        
        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        current_values = self.value_network(observations).squeeze(-1)
        value_loss_clipped = (current_values - returns).pow(2)
        value_loss = value_loss_clipped.mean()
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_coef * value_loss + 
                     self.entropy_coef * entropy_loss)
        
        # Additional statistics
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().item()
            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean().item()
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
            'approx_kl': torch.tensor(approx_kl),
            'clip_fraction': torch.tensor(clip_fraction),
        }
    
    def update_parameters(self, losses: Dict[str, torch.Tensor]) -> None:
        """Update model parameters using computed losses"""
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        losses['policy_loss'].backward(retain_graph=True)
        if self.max_grad_norm > 0:
            self.gradient_manager.clip_grad_norm(
                self.policy_network.parameters(), self.max_grad_norm
            )
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        losses['value_loss'].backward()
        if self.max_grad_norm > 0:
            self.gradient_manager.clip_grad_norm(
                self.value_network.parameters(), self.max_grad_norm
            )
        self.value_optimizer.step()
    
    def compute_advantages(self, last_value: float = 0.0) -> None:
        """Compute advantages for all trajectories in buffer"""
        self.trajectory_buffer.end_trajectory(last_value)
    
    def get_trajectory_data(self) -> List[Dict[str, Any]]:
        """Get all trajectory data for training"""
        return self.trajectory_buffer.get_all_trajectories()
    
    def clear_buffers(self) -> None:
        """Clear experience and trajectory buffers"""
        self.experience_buffer.clear()
        self.trajectory_buffer.clear()