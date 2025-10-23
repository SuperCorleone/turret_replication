#!/usr/bin/env python3
"""
Script for managing source policies for TURRET
- 加载已有的预训练模型
- 只在需要时训练新模型
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.base_config import TURRETConfig
from environments import get_standard_robot_env
from training.trainers.ppo_trainer import PPOTrainer
from models.policies.structured_policy import StructuredPolicyNetwork
from utils import setup_logging, save_checkpoint, load_checkpoint


class SimpleValueNetwork(torch.nn.Module):
    """Simple value network for source policy training"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(torch.nn.Linear(prev_dim, 1))
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_source_policy(env, config: Union[TURRETConfig, Dict[str, Any]]) -> StructuredPolicyNetwork:
    """Create a structured policy network for source task"""
    # 统一配置处理
    if hasattr(config, 'hidden_dim'):
        hidden_dim = config.hidden_dim
        device = config.device
    else:
        hidden_dim = config.get("hidden_dim", 256)
        device = config.get("device", "cpu")
    
    policy_config = {
        "node_observation_dim": env.observation_space.shape[0],
        "hidden_dim": hidden_dim,
        "output_dim": env.action_space.shape[0],
        "shared_across_nodes": True,
        "device": device
    }
    
    return StructuredPolicyNetwork(policy_config)


def find_model_file(robot_type: str, checkpoint_dir: str) -> Optional[str]:
    """查找模型文件 - 支持多种命名约定"""
    # 可能的文件名格式
    possible_filenames = [
        f"{robot_type.lower()}_policy.pth",  # ant_policy.pth
        f"{robot_type}_policy.pth",          # Ant_policy.pth  
        f"best_{robot_type}.pth",            # best_Ant.pth
        f"best_{robot_type.lower()}.pth",    # best_ant.pth
        f"{robot_type.lower()}.pth",         # ant.pth
        f"{robot_type}.pth",                 # Ant.pth
    ]
    
    for filename in possible_filenames:
        filepath = os.path.join(checkpoint_dir, filename)
        if os.path.exists(filepath):
            return filepath
    
    return None


def check_pretrained_models(robot_types: List[str], checkpoint_dir: str) -> Dict[str, bool]:
    """检查预训练模型是否存在"""
    status = {}
    print("\n🔍 检查预训练模型状态:")
    
    for robot_type in robot_types:
        model_file = find_model_file(robot_type, checkpoint_dir)
        if model_file:
            file_size = os.path.getsize(model_file) / 1024 / 1024  # MB
            status[robot_type] = True
            print(f"  ✅ {robot_type}: 找到 {os.path.basename(model_file)} ({file_size:.1f} MB)")
        else:
            status[robot_type] = False
            print(f"  ❌ {robot_type}: 未找到预训练模型")
    
    # 列出目录中所有.pth文件
    all_pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if all_pth_files:
        print(f"\n📁 目录中所有模型文件: {all_pth_files}")
    
    return status


def load_source_policies(robot_types: List[str], 
                        checkpoint_dir: str,
                        config: Union[TURRETConfig, Dict[str, Any]]) -> List[StructuredPolicyNetwork]:
    """加载预训练源策略 - 主要使用这个函数"""
    source_policies = []
    
    print(f"\n🔄 加载源策略...")
    
    # 统一配置处理
    if hasattr(config, 'device'):
        device = config.device
        hidden_dim = config.hidden_dim
    else:
        device = config.get("device", "cpu")
        hidden_dim = config.get("hidden_dim", 256)
    
    for robot_type in robot_types:
        model_file = find_model_file(robot_type, checkpoint_dir)
        
        if model_file:
            # 加载已有的预训练模型
            try:
                checkpoint = load_checkpoint(model_file, device)
                policy_net = create_fallback_policy(config)
                
                # 尝试不同的权重加载方式
                if 'policy_state' in checkpoint:
                    policy_net.load_state_dict(checkpoint['policy_state'])
                elif 'model_state_dict' in checkpoint:
                    policy_net.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    policy_net.load_state_dict(checkpoint['state_dict'])
                else:
                    # 直接加载
                    policy_net.load_state_dict(checkpoint)
                
                source_policies.append(policy_net)
                print(f"  ✅ 加载预训练策略: {robot_type}")
                
            except Exception as e:
                print(f"  ⚠️ 加载 {robot_type} 失败: {e}，使用未训练策略")
                source_policies.append(create_fallback_policy(config))
        else:
            # 没有预训练模型，使用未训练的策略
            print(f"  ⚠️ {robot_type}: 无预训练模型，使用未训练策略")
            source_policies.append(create_fallback_policy(config))
    
    print(f"✅ 成功加载 {len([p for p in source_policies if p is not None])}/{len(robot_types)} 个策略")
    return source_policies


def train_source_policy(robot_type: str, 
                       config: TURRETConfig,
                       save_path: str) -> bool:
    """训练源策略 - 只在需要时调用"""
    
    logger = setup_logging()
    logger.info(f"Training source policy for {robot_type}")
    
    try:
        # Create environment
        StandardRobotEnv = get_standard_robot_env()
        env = StandardRobotEnv({
            "robot_type": robot_type,
            "max_episode_steps": config.max_episode_steps
        })
        
        # Create policy and value networks
        policy_net = create_source_policy(env, config)
        value_net = SimpleValueNetwork(env.observation_space.shape[0])
        
        # Create trainer
        trainer = PPOTrainer(policy_net, value_net, config)
        
        # Training loop
        best_reward = -float('inf')
        episode_rewards = []
        
        for episode in range(config.total_episodes):
            # Collect experience
            observation, info = env.reset()
            episode_reward = 0
            
            while True:
                # Get action
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                    mean, std = policy_net(obs_tensor)
                    action_dist = torch.distributions.Normal(mean, std)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
                    value = value_net(obs_tensor).squeeze(-1)
                
                # Take action
                next_observation, reward, terminated, truncated, info = env.step(
                    action.squeeze(0).numpy()
                )
                
                # Store experience
                trainer.experience_buffer.add(
                    observation, action.squeeze(0).numpy(), reward,
                    next_observation, terminated, truncated,
                    log_prob.item(), value.item()
                )
                
                # Update
                observation = next_observation
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            # Log episode
            trainer.log_episode(episode_reward, 0)
            episode_rewards.append(episode_reward)
            
            # Train periodically
            if episode % 10 == 0 and len(trainer.experience_buffer) >= 64:
                batch = trainer.experience_buffer.sample(64)
                stats = trainer.train_step(batch)
                
                if episode % 100 == 0:
                    logger.info(f"Episode {episode}: reward={episode_reward:.2f}, "
                               f"policy_loss={stats['policy_loss']:.4f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                checkpoint = {
                    'policy_state': policy_net.state_dict(),
                    'value_state': value_net.state_dict(),
                    'episode': episode,
                    'reward': episode_reward,
                    'config': config.__dict__
                }
                save_checkpoint(checkpoint, os.path.join(save_path, f"best_{robot_type}.pth"))
            
            # Early stopping
            if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) > 3000:
                logger.info(f"Early stopping: achieved target reward for {robot_type}")
                break
        
        env.close()
        
        # Save final model
        final_checkpoint = {
            'policy_state': policy_net.state_dict(),
            'value_state': value_net.state_dict(),
            'final_episode': episode,
            'final_reward': episode_reward,
            'config': config.__dict__
        }
        save_checkpoint(final_checkpoint, os.path.join(save_path, f"final_{robot_type}.pth"))
        
        logger.info(f"Completed training for {robot_type}. Best reward: {best_reward:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to train {robot_type}: {e}")
        return False


def create_fallback_policy(config: Union[TURRETConfig, Dict[str, Any]]) -> StructuredPolicyNetwork:
    """创建回退策略（未训练）"""
    # 统一配置处理
    if hasattr(config, 'hidden_dim'):
        hidden_dim = config.hidden_dim
        device = config.device
    else:
        hidden_dim = config.get("hidden_dim", 256)
        device = config.get("device", "cpu")
    
    return StructuredPolicyNetwork({
        "node_observation_dim": 10,  # 默认值
        "hidden_dim": hidden_dim,
        "output_dim": 6,  # 默认值
        "shared_across_nodes": True,
        "device": device
    })


def create_simple_value_network(input_dim: int) -> torch.nn.Module:
    """Create a simple value network for experiments"""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1)
    )


def main():
    """主函数 - 检查模型状态并提供训练选项"""
    
    # Configuration
    config = TURRETConfig(
        device="cpu",
        total_episodes=500,
        batch_size=64,
        learning_rate=3e-4,
        max_episode_steps=1000,
        hidden_dim=256,
        ppo_epochs=10,
        mini_batch_size=64,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        buffer_size=10000,
        gamma=0.99,
        gae_lambda=0.95,
        observation_shape=(17,),
        action_shape=(6,)
    )
    
    # Robot types
    robot_types = ["HalfCheetah", "Ant", "Hopper", "Walker2d"]
    save_dir = "data/pretrained/source_policies"
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查模型状态
    model_status = check_pretrained_models(robot_types, save_dir)
    
    # 统计
    trained_count = sum(1 for status in model_status.values() if status)
    untrained_count = len(robot_types) - trained_count
    
    print(f"\n📊 模型状态统计:")
    print(f"  ✅ 已预训练: {trained_count} 个")
    print(f"  ❌ 需要训练: {untrained_count} 个")
    
    # 如果有缺失的模型，询问是否训练
    if untrained_count > 0:
        print(f"\n⚠️  有 {untrained_count} 个机器人没有预训练模型")
        response = input("是否现在训练缺失的模型? (y/n): ")
        
        if response.lower() == 'y':
            for robot_type, has_model in model_status.items():
                if not has_model:
                    print(f"\n🎯 开始训练: {robot_type}")
                    success = train_source_policy(robot_type, config, save_dir)
                    if success:
                        print(f"✅ 完成训练: {robot_type}")
                    else:
                        print(f"❌ 训练失败: {robot_type}")
        else:
            print("跳过训练，使用未训练的策略进行实验")
    else:
        print("\n🎉 所有预训练模型都已就绪！")


if __name__ == "__main__":
    main()