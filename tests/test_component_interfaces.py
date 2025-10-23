# tests/test_component_interfaces.py
#!/usr/bin/env python3
"""测试各组件接口一致性"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.base_config import TURRETConfig

def test_training_components():
    """测试训练组件接口"""
    config = TURRETConfig(
        device="cpu",
        total_episodes=5,
        batch_size=32
    )
    
    # 测试PPO训练器
    try:
        from training.trainers.ppo_trainer import PPOTrainer
        from models.policies.structured_policy import StructuredPolicyNetwork
        
        # 创建简单网络
        policy_net = StructuredPolicyNetwork({
            "node_observation_dim": 10,
            "hidden_dim": 32,
            "output_dim": 4,
            "device": "cpu"
        })
        
        # 创建简单值网络
        class SimpleValueNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(10, 1)
            def forward(self, x):
                return self.net(x)
        
        value_net = SimpleValueNetwork()
        
        # 创建训练器
        trainer = PPOTrainer(policy_net, value_net, config)
        assert trainer is not None
        print("✅ PPO训练器接口测试通过")
        
    except Exception as e:
        print(f"❌ PPO训练器接口测试失败: {e}")
        raise

def test_transfer_components():
    """测试迁移学习组件接口"""
    config = TURRETConfig(
        device="cpu",
        embedding_dim=64
    )
    
    try:
        from transfer.semantic_space import SemanticSpaceManager
        from transfer.weight_calculator import AdaptiveWeightCalculator
        
        # 测试语义空间管理器
        semantic_space = SemanticSpaceManager({
            "embedding_dim": 64,
            "device": "cpu"
        })
        
        # 测试状态投影
        test_state = torch.randn(64)
        embedding = semantic_space.project_state(test_state)
        assert embedding.shape == (64,)
        print("✅ 语义空间管理器接口测试通过")
        
        # 测试权重计算器
        weight_calculator = AdaptiveWeightCalculator({
            "embedding_dim": 64,
            "device": "cpu"
        })
        
        target_state = torch.randn(64)
        source_states = [torch.randn(64), torch.randn(64)]
        weights = weight_calculator.compute_transfer_weights(target_state, source_states)
        
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-6
        print("✅ 权重计算器接口测试通过")
        
    except Exception as e:
        print(f"❌ 迁移学习组件接口测试失败: {e}")
        raise

def test_environment_components():
    """测试环境组件接口"""
    try:
        from environments import get_standard_robot_env
        
        # 修复：检查返回的是类还是实例
        StandardRobotEnv = get_standard_robot_env()
        
        # 如果是实例，尝试获取其类
        if not isinstance(StandardRobotEnv, type):
            # 可能是已经实例化的对象，尝试其他方式创建环境
            try:
                # 方法1：尝试使用现有的实例
                env = StandardRobotEnv
                print("ℹ️  使用现有环境实例进行测试")
            except:
                # 方法2：尝试从模块直接导入
                from environments.tasks.standard_robots import StandardRobotEnv as EnvClass
                env = EnvClass({"robot_type": "HalfCheetah", "max_episode_steps": 100})
        else:
            # 如果是类，正常实例化
            env = StandardRobotEnv({
                "robot_type": "HalfCheetah",
                "max_episode_steps": 100
            })
        
        # 测试基本功能
        obs, info = env.reset()
        assert obs.dtype == np.float32
        
        action = np.random.uniform(-1, 1, env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        
        env.close()
        print("✅ 环境组件接口测试通过")
        
    except Exception as e:
        print(f"❌ 环境组件接口测试失败: {e}")
        raise

if __name__ == "__main__":
    print("运行组件接口测试...")
    test_training_components()
    test_transfer_components()
    test_environment_components()
    print("🎉 所有组件接口测试通过！")