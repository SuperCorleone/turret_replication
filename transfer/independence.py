import torch
import torch.nn as nn
from typing import Dict, Any, List
import math
import numpy as np  # 添加缺失的导入


class GradualIndependenceScheduler:
    """
    Implements the gradual independence mechanism from the paper
    Controls the p factor that determines target network independence
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Scheduler parameters
        self.initial_p = config.get("initial_p", 0.0)
        self.final_p = config.get("final_p", 1.0)
        self.total_steps = config.get("independence_steps", 1000000)
        self.schedule_type = config.get("schedule_type", "linear")
        
        # Current state
        self.current_step = 0
        self.current_p = self.initial_p
        
        # Performance-based adaptation
        self.performance_threshold = config.get("performance_threshold", 0.0)
        self.performance_adaptive = config.get("performance_adaptive", False)
        self.performance_history = []
    
    def get_current_p(self) -> float:
        """Get current independence factor p"""
        return self.current_p
    
    def step(self, performance: float = None) -> None:
        """Update scheduler step"""
        self.current_step += 1
        
        if self.schedule_type == "linear":
            self._linear_schedule()
        elif self.schedule_type == "exponential":
            self._exponential_schedule()
        elif self.schedule_type == "cosine":
            self._cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        # Performance-based adaptation
        if self.performance_adaptive and performance is not None:
            self._performance_adaptation(performance)
        
        # Ensure p is within bounds
        self.current_p = max(self.initial_p, min(self.final_p, self.current_p))
    
    def _linear_schedule(self) -> None:
        """Linear schedule from initial_p to final_p"""
        progress = min(1.0, self.current_step / self.total_steps)
        self.current_p = self.initial_p + (self.final_p - self.initial_p) * progress
    
    def _exponential_schedule(self) -> None:
        """Exponential schedule"""
        progress = min(1.0, self.current_step / self.total_steps)
        # Exponential decay of dependence (1-p grows exponentially)
        dependence = 1 - self.current_p
        dependence = (1 - self.initial_p) * math.exp(-5 * progress)
        self.current_p = 1 - dependence
    
    def _cosine_schedule(self) -> None:
        """Cosine annealing schedule"""
        progress = min(1.0, self.current_step / self.total_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        self.current_p = self.final_p - (self.final_p - self.initial_p) * cosine_decay
    
    def _performance_adaptation(self, performance: float) -> None:
        """Adapt schedule based on performance"""
        self.performance_history.append(performance)
        
        # Use moving average of recent performance
        window_size = min(100, len(self.performance_history))
        recent_performance = np.mean(self.performance_history[-window_size:])
        
        if recent_performance > self.performance_threshold:
            # Good performance: accelerate independence
            acceleration = min(2.0, 1.0 + (recent_performance - self.performance_threshold))
            effective_step = min(self.total_steps, int(self.current_step * acceleration))
            progress = min(1.0, effective_step / self.total_steps)
            self.current_p = self.initial_p + (self.final_p - self.initial_p) * progress
        else:
            # Poor performance: maintain more dependence
            deceleration = max(0.5, 1.0 - (self.performance_threshold - recent_performance))
            effective_step = max(0, int(self.current_step * deceleration))
            progress = min(1.0, effective_step / self.total_steps)
            self.current_p = self.initial_p + (self.final_p - self.initial_p) * progress
    
    def should_transfer(self) -> bool:
        """Check if transfer should occur based on current p"""
        return self.current_p < 1.0  # Transfer when p < 1
    
    def get_transfer_strength(self) -> float:
        """Get current transfer strength (1 - p)"""
        return 1.0 - self.current_p
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing"""
        return {
            'current_step': self.current_step,
            'current_p': self.current_p,
            'performance_history': self.performance_history,
            'config': self.config
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint"""
        self.current_step = state_dict['current_step']
        self.current_p = state_dict['current_p']
        self.performance_history = state_dict['performance_history']