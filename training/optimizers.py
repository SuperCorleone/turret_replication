# training/optimizers.py - 完整版本

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import math
import numpy as np

class GradientManager:
    """
    Utility class for gradient management (clipping, logging, etc.)
    """
    @staticmethod
    def clip_grad_norm(parameters: List[torch.Tensor], max_norm: float) -> float:
        """Clip gradient norm and return the actual norm"""
        if max_norm <= 0:
            return 0.0
        
        parameters = [p for p in parameters if p.grad is not None]
        if not parameters:
            return 0.0
        
        # 计算总梯度范数
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]),
            2
        )
        
        clip_coef = max_norm / (total_norm + 1e-6)
        # 如果梯度范数超过最大值，进行裁剪
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        
        return total_norm.item()  # 返回裁剪前的原始范数

    @staticmethod
    def clip_grad_norm_pytorch(parameters: List[torch.Tensor], max_norm: float) -> float:
        """Use PyTorch's built-in gradient clipping"""
        parameters = [p for p in parameters if p.grad is not None]
        if not parameters:
            return 0.0
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()

    @staticmethod
    def compute_grad_norm(parameters: List[torch.Tensor]) -> float:
        """Compute total gradient norm"""
        parameters = [p for p in parameters if p.grad is not None]
        if not parameters:
            return 0.0
        return torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]),
            2
        ).item()

    @staticmethod
    def compute_param_norm(parameters: List[torch.Tensor]) -> float:
        """Compute total parameter norm"""
        parameters = [p for p in parameters if p is not None]
        if not parameters:
            return 0.0
        return torch.norm(
            torch.stack([torch.norm(p.detach(), 2) for p in parameters]),
            2
        ).item()

    @staticmethod
    def zero_grad(parameters: List[torch.Tensor]) -> None:
        """Zero gradients for parameters"""
        for param in parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    @staticmethod
    def enable_gradients(model: nn.Module, enable: bool = True) -> None:
        """Enable or disable gradients for model parameters"""
        for param in model.parameters():
            param.requires_grad = enable


class LearningRateScheduler:
    """
    Learning rate scheduler with various scheduling strategies
    """
    def __init__(self, optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
        self.optimizer = optimizer
        self.config = config
        self.scheduler_type = config.get("scheduler_type", "constant")
        self.initial_lr = config.get("learning_rate", 3e-4)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.decay_steps = config.get("decay_steps", 1000000)
        self.min_lr = config.get("min_lr", 1e-6)
        self.step_count = 0
        
        # Initialize scheduler
        self._init_scheduler()

    def _init_scheduler(self) -> None:
        """Initialize the appropriate scheduler"""
        if self.scheduler_type == "constant":
            self.scheduler = None
        elif self.scheduler_type == "linear":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(
                    self.min_lr / self.initial_lr,
                    1.0 - step / self.decay_steps
                )
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.decay_steps,
                eta_min=self.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def step(self) -> None:
        """Update learning rate"""
        self.step_count += 1
        
        # Warmup phase
        if self.step_count < self.warmup_steps and self.warmup_steps > 0:
            lr_scale = min(1.0, float(self.step_count) / float(self.warmup_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * lr_scale
        elif self.scheduler is not None:
            self.scheduler.step()

    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing"""
        state = {
            'step_count': self.step_count,
            'config': self.config
        }
        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint"""
        self.step_count = state_dict['step_count']
        if self.scheduler is not None and 'scheduler_state' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler_state'])


class PPOScheduler:
    """
    Specialized scheduler for PPO with clipping schedule
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_clip = config.get("clip_epsilon", 0.2)
        self.final_clip = config.get("final_clip_epsilon", 0.1)
        self.decay_steps = config.get("clip_decay_steps", 1000000)
        self.step_count = 0

    def get_clip_epsilon(self) -> float:
        """Get current clip epsilon value"""
        if self.step_count >= self.decay_steps:
            return self.final_clip
        
        progress = self.step_count / self.decay_steps
        return self.initial_clip - (self.initial_clip - self.final_clip) * progress

    def step(self) -> None:
        """Update step count"""
        self.step_count += 1


class GradientMonitor:
    """
    Monitor gradient statistics during training
    """
    def __init__(self):
        self.grad_norms = []
        self.param_norms = []
        self.grad_means = []
        self.grad_stds = []

    def update(self, model: nn.Module) -> Dict[str, float]:
        """Update gradient statistics"""
        grad_norms = []
        param_norms = []
        grad_values = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                grad_norms.append(grad_norm)
                param_norms.append(param_norm)
                grad_values.extend(param.grad.cpu().flatten().tolist())
        
        stats = {}
        if grad_norms:
            stats['grad_norm_mean'] = float(np.mean(grad_norms))
            stats['grad_norm_std'] = float(np.std(grad_norms))
            stats['grad_norm_max'] = float(np.max(grad_norms))
            stats['param_norm_mean'] = float(np.mean(param_norms))
        
        if grad_values:
            stats['grad_mean'] = float(np.mean(grad_values))
            stats['grad_std'] = float(np.std(grad_values))
        
        # Store for history
        self.grad_norms.append(stats.get('grad_norm_mean', 0.0))
        self.param_norms.append(stats.get('param_norm_mean', 0.0))
        self.grad_means.append(stats.get('grad_mean', 0.0))
        self.grad_stds.append(stats.get('grad_std', 0.0))
        
        return stats