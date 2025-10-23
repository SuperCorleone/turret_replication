# configs/base_config.py

from dataclasses import dataclass, field, fields
from typing import Dict, Any, List, Optional, Union
import os

@dataclass
class TURRETConfig:
    """TURRET 实验配置类"""
    
    # 基础实验参数
    device: str = "cpu"
    total_episodes: int = 500
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_episode_steps: int = 1000
    hidden_dim: int = 256
    
    # PPO 参数
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    buffer_size: int = 10000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # 迁移学习参数
    embedding_dim: int = 128
    temperature: float = 1.0
    min_weight: float = 0.01
    initial_p: float = 0.0
    final_p: float = 1.0
    independence_steps: int = 1000000
    
    # 实验参数
    num_seeds: int = 5
    log_level: int = 20

    # 环境类型 - 新增字段
    environment_type: str = "centipede"
    
    # 实验特定参数 - 新增字段
    experiment_id: str = "default_experiment"
    results_dir: str = field(default_factory=lambda: os.path.join("data", "final_results"))
    
    # 规模迁移实验参数 - 新增字段
    source_robots: List[str] = field(default_factory=lambda: ["HalfCheetah", "Ant"])
    target_robot: str = "Humanoid"
    
    # 形态迁移实验参数 - 新增字段
    transfer_type: str = "quad_to_biped"
    
    # 消融实验参数 - 新增字段
    ablation_type: str = "no_attention"
    
    # 基线对比实验参数 - 新增字段
    baseline_method: str = "PPO"
    
    # 观察和动作空间（用于缓冲区）
    observation_shape: tuple = field(default_factory=lambda: (10,))
    action_shape: tuple = field(default_factory=lambda: (4,))
    
    # 动态字段存储（用于存储未在dataclass中定义的参数）
    _extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保列表字段是可变的
        if not hasattr(self, '_extra_params'):
            self._extra_params = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and not callable(v):
                result[k] = v
        # 包含额外参数
        result.update(self._extra_params)
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TURRETConfig':
        """从字典创建配置实例"""
        # 分离已知字段和额外字段
        known_fields = {f.name for f in fields(cls) if f.name != '_extra_params'}
        known_params = {}
        extra_params = {}
        
        for k, v in config_dict.items():
            if k in known_fields:
                known_params[k] = v
            else:
                extra_params[k] = v
        
        instance = cls(**known_params)
        instance._extra_params = extra_params
        return instance
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持额外参数"""
        if hasattr(self, key):
            return getattr(self, key)
        return self._extra_params.get(key, default)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._extra_params[key] = value
    
    def copy(self) -> 'TURRETConfig':
        """创建配置的副本"""
        return self.from_dict(self.to_dict())