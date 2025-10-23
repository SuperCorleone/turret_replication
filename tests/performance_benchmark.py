# tests/performance_benchmark.py
#!/usr/bin/env python3
"""性能基准测试"""

import time
import torch
import numpy as np

# 添加缺失的导入
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.base_config import TURRETConfig

def benchmark_training_step():
    """测试训练步骤性能"""
    from training.trainers.ppo_trainer import PPOTrainer
    from models.policies.structured_policy import StructuredPolicyNetwork
    
    # 创建测试数据
    policy_net = StructuredPolicyNetwork({
        "node_observation_dim": 10,
        "hidden_dim": 32, 
        "output_dim": 4,
        "device": "cpu"
    })
    
    # 修复：添加可训练参数
    class SimpleValueNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)  # 添加可训练参数
        
        def forward(self, x):
            return self.linear(x)
    
    value_net = SimpleValueNetwork()
    
    config = TURRETConfig(
        device="cpu",
        batch_size=32
    )
    
    trainer = PPOTrainer(policy_net, value_net, config)
    
    # 创建测试批次
    batch = {
        'observations': torch.randn(32, 10),
        'actions': torch.randn(32, 4),
        'rewards': torch.randn(32),
        'log_probs': torch.randn(32),
        'values': torch.randn(32)
    }
    
    # 性能测试
    start_time = time.time()
    for i in range(10):  # 减少到10次迭代用于测试
        stats = trainer.train_step(batch)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    print(f"✅ 平均训练步骤时间: {avg_time:.4f} 秒")
    print(f"✅ 训练步骤统计: {stats}")
    
if __name__ == "__main__":
    benchmark_training_step()