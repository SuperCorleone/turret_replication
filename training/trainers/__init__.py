from .base_trainer import BaseTrainer
from .ppo_trainer import PPOTrainer
from .transfer_trainer import TURRETTrainer

__all__ = ["BaseTrainer", "PPOTrainer", "TURRETTrainer"]

# 向后兼容的函数
def get_PPOTrainer():
    """获取PPO训练器类（向后兼容）"""
    return PPOTrainer