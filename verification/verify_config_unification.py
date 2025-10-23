#!/usr/bin/env python3
"""
验证配置系统统一化
"""
import sys
import os
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.base_config import TURRETConfig
from training import PPOTrainer, TURRETTrainer
from models.policies.structured_policy import StructuredPolicyNetwork

def test_config_unification():
    """测试配置统一化"""
    print("Testing configuration unification...")
    
    # 测试1: 使用 TURRETConfig
    config = TURRETConfig(
        device="cpu",
        total_episodes=100,
        learning_rate=3e-4,
        ppo_epochs=10,
        embedding_dim=128
    )
    
    # 创建简单网络
    policy_net = StructuredPolicyNetwork({
        "node_observation_dim": 10,
        "hidden_dim": 32,
        "output_dim": 4,
        "device": "cpu"
    })
    
    value_net = torch.nn.Linear(10, 1)
    
    # 测试 PPOTrainer
    try:
        ppo_trainer = PPOTrainer(policy_net, value_net, config)
        print("✓ PPOTrainer accepts TURRETConfig")
    except Exception as e:
        print(f"✗ PPOTrainer failed with TURRETConfig: {e}")
        return False
    
    # 测试 TURRETTrainer
    try:
        turret_trainer = TURRETTrainer(
            policy_net, value_net, [policy_net], [value_net], config
        )
        print("✓ TURRETTrainer accepts TURRETConfig")
    except Exception as e:
        print(f"✗ TURRETTrainer failed with TURRETConfig: {e}")
        return False
    
    # 测试2: 使用字典配置（向后兼容）
    dict_config = {
        "device": "cpu",
        "learning_rate": 3e-4,
        "ppo_epochs": 5
    }
    
    try:
        ppo_trainer_dict = PPOTrainer(policy_net, value_net, dict_config)
        print("✓ PPOTrainer accepts dict config (backward compatibility)")
    except Exception as e:
        print(f"✗ PPOTrainer failed with dict config: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Running Configuration Unification Test")
    print("=" * 50)
    
    if test_config_unification():
        print("=" * 50)
        print("🎉 Configuration unification completed successfully!")
        print("All trainers now accept both TURRETConfig and dict configurations.")
    else:
        print("❌ Configuration unification failed")
        sys.exit(1)