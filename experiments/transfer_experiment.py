#!/usr/bin/env python3
"""
Script for running TURRET transfer experiments
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, Any, List
import json

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.base_config import TURRETConfig
from environments import get_standard_robot_env
from training.trainers.transfer_trainer import TURRETTrainer
from models.policies.structured_policy import StructuredPolicyNetwork
from utils import setup_logging, load_checkpoint, save_checkpoint


def load_source_policies(robot_types: List[str], 
                        checkpoint_dir: str,
                        config: TURRETConfig) -> List[StructuredPolicyNetwork]:  # 改为TURRETConfig类型
    """Load pre-trained source policies - 适配下载的模型格式"""
    
    source_policies = []
    
    for robot_type in robot_types:
        # 尝试多种可能的文件名
        possible_paths = [
            os.path.join(checkpoint_dir, f"{robot_type.lower()}_policy.pth"),
            os.path.join(checkpoint_dir, f"{robot_type.lower()}_compatible.pth"), 
            os.path.join(checkpoint_dir, f"best_{robot_type}.pth"),
            os.path.join(checkpoint_dir, f"final_{robot_type}.pth")
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path:
            print(f"Loading source policy from: {checkpoint_path}")
            try:
                # Load checkpoint
                checkpoint = load_checkpoint(checkpoint_path)
                
                # 创建策略网络
                policy_net = StructuredPolicyNetwork({
                    "node_observation_dim": 17,  # 使用实际观察维度
                    "hidden_dim": config.hidden_dim,
                    "output_dim": 6,  # 使用实际动作维度
                    "shared_across_nodes": True,
                    "device": config.device
                })
                
                # 适配不同的模型格式
                if 'policy_state_dict' in checkpoint:
                    # 我们的兼容格式
                    policy_net.load_state_dict(checkpoint['policy_state_dict'])
                elif 'actor' in checkpoint:
                    # Stable-Baselines3格式
                    # 需要适配权重名称
                    actor_weights = checkpoint['actor']
                    adapted_weights = adapt_sb3_weights(actor_weights)
                    policy_net.load_state_dict(adapted_weights)
                elif 'model_state_dict' in checkpoint:
                    # PyTorch格式
                    policy_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # 直接加载
                    policy_net.load_state_dict(checkpoint)
                
                source_policies.append(policy_net)
                print(f"✓ Successfully loaded {robot_type} policy")
                
            except Exception as e:
                print(f"✗ Failed to load {robot_type}: {e}")
                # 创建回退策略
                policy_net = create_fallback_policy(robot_type, config)
                source_policies.append(policy_net)
        else:
            print(f"✗ No checkpoint found for {robot_type}")
            # 创建回退策略
            policy_net = create_fallback_policy(robot_type, config)
            source_policies.append(policy_net)
    
    return source_policies

def adapt_sb3_weights(sb3_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """适配Stable-Baselines3权重到我们的网络结构"""
    adapted_weights = {}
    
    for key, value in sb3_weights.items():
        # 简化适配 - 实际需要根据具体网络结构调整
        if 'pi' in key:
            new_key = key.replace('pi_net.', '').replace('pi_features_extractor.', '')
            adapted_weights[new_key] = value
    
    return adapted_weights

def create_fallback_policy(robot_type: str, config: TURRETConfig) -> StructuredPolicyNetwork:  # 改为TURRETConfig类型
    """
    创建回退策略
    """
    print(f"Creating fallback policy for {robot_type}")
    
    policy_net = StructuredPolicyNetwork({
        "node_observation_dim": 17,
        "hidden_dim": 128,  # 更小的网络
        "output_dim": 6,
        "shared_across_nodes": True,
        "device": config.device
    })
    
    return policy_net


class SimpleValueNetwork(torch.nn.Module):
    """Simple value network for experiments"""
    
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


def run_transfer_experiment(source_robots: List[str],
                          target_robot: str,
                          config: TURRETConfig,  # 改为TURRETConfig类型
                          results_dir: str) -> Dict[str, Any]:
    """Run a single transfer learning experiment"""
    
    logger = setup_logging()
    logger.info(f"Running transfer experiment: {source_robots} -> {target_robot}")
    
    # Create environments
    StandardRobotEnv = get_standard_robot_env()
    target_env = StandardRobotEnv({
        "robot_type": target_robot,
        "max_episode_steps": config.max_episode_steps
    })
    
    # Load source policies
    source_policies = load_source_policies(
        source_robots, 
        "data/pretrained/source_policies",
        config
    )
    
    # Create source value networks (simplified)
    source_values = [
        SimpleValueNetwork(target_env.observation_space.shape[0])
        for _ in range(len(source_policies))
    ]
    
    # Create target networks
    target_policy = StructuredPolicyNetwork({
        "node_observation_dim": target_env.observation_space.shape[0],
        "hidden_dim": config.hidden_dim,
        "output_dim": target_env.action_space.shape[0],
        "shared_across_nodes": True,
        "device": config.device
    })
    
    target_value = SimpleValueNetwork(target_env.observation_space.shape[0])
    
    # Create TURRET trainer - 直接传递TURRETConfig对象
    trainer = TURRETTrainer(
        target_policy, target_value, source_policies, source_values, config
    )
    
    # Training loop
    results = {
        'episode_rewards': [],
        'transfer_statistics': [],
        'training_stats': []
    }
    
    for episode in range(config.total_episodes):
        # Collect experience with transfer
        trajectories = trainer.collect_transfer_experience(target_env, num_episodes=1)
        
        # Train on collected experience
        if len(trainer.experience_buffer) >= config.batch_size:
            batch = trainer.experience_buffer.sample(config.batch_size)
            stats = trainer.train_step(batch)
            
            # Store results
            results['episode_rewards'].append(trainer.episode_rewards[-1] if trainer.episode_rewards else 0)
            results['transfer_statistics'].append(trainer.get_transfer_statistics())
            results['training_stats'].append(stats)
        
        # Log progress
        if episode % 50 == 0:
            current_reward = trainer.episode_rewards[-1] if trainer.episode_rewards else 0
            p = trainer.independence_scheduler.get_current_p()
            logger.info(f"Episode {episode}: reward={current_reward:.2f}, p={p:.3f}")
        
        # Save checkpoint periodically
        if episode % 100 == 0:
            checkpoint_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            trainer.save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth"))
    
    target_env.close()
    
    # Save final results
    results_path = os.path.join(results_dir, f"results_{target_robot}.json")
    with open(results_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        serializable_results = {
            'episode_rewards': [float(r) for r in results['episode_rewards']],
            'transfer_statistics': results['transfer_statistics'],
            'final_training_stats': results['training_stats'][-1] if results['training_stats'] else {}
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Completed transfer experiment for {target_robot}")
    
    return results


def main():
    """Main function for running transfer experiments"""
    
    # Experiment configuration - 使用TURRETConfig
    config = TURRETConfig(
        device="cpu",
        total_episodes=500,
        batch_size=64,
        learning_rate=3e-4,
        max_episode_steps=1000,
        hidden_dim=256,
        # 添加迁移学习相关参数
        embedding_dim=128,
        temperature=1.0,
        min_weight=0.01,
        initial_p=0.0,
        final_p=1.0,
        independence_steps=500000
    )
    
    # Define transfer scenarios
    transfer_scenarios = [
        {
            "name": "size_transfer_small_to_large",
            "sources": ["HalfCheetah", "Ant"],
            "target": "Humanoid"
        },
        {
            "name": "morphology_transfer_quad_to_biped", 
            "sources": ["HalfCheetah", "Ant"],
            "target": "Walker2d"
        },
        {
            "name": "similar_morphology",
            "sources": ["Hopper", "Walker2d"],
            "target": "Humanoid"
        }
    ]
    
    # Create results directory
    results_dir = "data/results/transfer_experiments"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments
    for scenario in transfer_scenarios:
        scenario_dir = os.path.join(results_dir, scenario["name"])
        os.makedirs(scenario_dir, exist_ok=True)
        
        print(f"Running scenario: {scenario['name']}")
        
        try:
            results = run_transfer_experiment(
                scenario["sources"],
                scenario["target"],
                config,
                scenario_dir
            )
            
            print(f"Completed {scenario['name']}")
            
        except Exception as e:
            print(f"Failed to run {scenario['name']}: {e}")
            continue


if __name__ == "__main__":
    main()