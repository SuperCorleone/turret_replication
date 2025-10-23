import torch
import numpy as np
from typing import Dict, Any, List
import os

# 添加导入路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from configs.base_config import TURRETConfig
from experiments.runners.base_runner import ExperimentRunner
from training.trainers.transfer_trainer import TURRETTrainer
from models.policies.structured_policy import StructuredPolicyNetwork
from experiments.pretrain_source import load_source_policies, create_simple_value_network

# 修复：添加环境导入
from environments import get_standard_robot_env
from environments.tasks.standard_robots import StandardRobotEnv
from environments.tasks.centipede import CentipedeEnv

class SizeTransferRunner(ExperimentRunner):
    """
    Runner for size transfer experiments (Centipede少足机器人 -> 多足机器人)
    与论文一致：从Centipede-4/6迁移到Centipede-12/16/20
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Size transfer specific configuration
        self.source_robots = self.config.get("source_robots", ["HalfCheetah", "Ant"])
        self.target_robot = self.config.get("target_robot", "Humanoid")
        self.total_episodes = self.config.get("total_episodes", 500)
        
        # Components
        self.env = None
        self.trainer = None
        self.source_policies = []
    
    def setup_experiment(self) -> None:
        """Setup size transfer experiment"""
        self.logger.info(f"Setting up size transfer: {self.source_robots} -> {self.target_robot}")
        
        # Create environment instance
        if self.target_robot.startswith("Centipede-"):
            # Centipede机器人
            try:
                num_legs = int(self.target_robot.split("-")[1])
                self.env = CentipedeEnv({
                    "num_torsos": num_legs // 2,
                    "max_episode_steps": self.config.get("max_episode_steps", 1000)
                })
            except (ValueError, IndexError):
                raise ValueError(f"Invalid Centipede format: {self.target_robot}")
        else:
            # 标准机器人
            self.env = StandardRobotEnv({
                "robot_type": self.target_robot,
                "max_episode_steps": self.config.get("max_episode_steps", 1000)
            })
        
        # Load source policies
        self.source_policies = load_source_policies(
            self.source_robots,
            "data/pretrained/source_policies",
            self.config
        )
        
        # Create source value networks
        source_values = [
            create_simple_value_network(self.env.observation_space.shape[0])
            for _ in range(len(self.source_policies))
        ]
        
        # Create target networks
        target_policy = StructuredPolicyNetwork({
            "node_observation_dim": self.env.observation_space.shape[0],
            "hidden_dim": self.config.get("hidden_dim", 256),
            "output_dim": self.env.action_space.shape[0],
            "shared_across_nodes": True,
            "device": self.config.get("device", "cpu")
        })
        
        target_value = create_simple_value_network(self.env.observation_space.shape[0])
        
        # Create TURRET trainer
        trainer_config = {
            **self.config,
            "embedding_dim": self.config.get("embedding_dim", 128),
            "temperature": self.config.get("temperature", 1.0),
            "min_weight": self.config.get("min_weight", 0.01),
            "initial_p": self.config.get("initial_p", 0.0),
            "final_p": self.config.get("final_p", 1.0),
            "independence_steps": self.total_episodes * 100,
        }
        
        self.trainer = TURRETTrainer(
            target_policy, target_value, self.source_policies, source_values, trainer_config
        )
        
        self.logger.info("Size transfer experiment setup completed")
    
    def run_experiment(self) -> Dict[str, Any]:
        """运行规模迁移实验"""
        self.logger.info("Starting size transfer experiment")
        
        training_curves = []
        transfer_stats_history = []
        
        for episode in range(self.total_episodes):
            # 收集经验
            episode_reward = self._collect_episode_experience()
            
            training_stats = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0,
                'independence_p': 0.0
            }

            # 训练
            if len(self.trainer.experience_buffer) >= self.config.get("batch_size", 64):
                batch = self.trainer.experience_buffer.sample(self.config.get("batch_size", 64))
                training_stats = self.trainer.train_step(batch)
                
                # 存储结果
                training_curves.append({
                    'episode': episode,
                    'reward': episode_reward,
                    **training_stats
                })
                
                transfer_stats = self.trainer.get_transfer_statistics()
                transfer_stats_history.append(transfer_stats)
            
            # 日志进度
            if episode % 50 == 0:
                self.log_progress(episode, {
                    'reward': episode_reward,
                    'policy_loss': training_stats.get('policy_loss', 0),
                    'independence_p': training_stats.get('independence_p', 0)
                })
        
        # 存储最终结果
        self.results['training_curves'] = training_curves
        self.results['transfer_statistics'] = transfer_stats_history
        self.results['final_performance'] = self.evaluate()
        self.results['experiment_type'] = 'size_transfer'
        self.results['scenario'] = f"{'_'.join(self.source_robots)}_to_{self.target_robot}"
        
        # 保存最终结果
        self.save_results()
        self.logger.info("Size transfer experiment completed")
        
        return self.results
    
    def _collect_episode_experience(self) -> float:
        """收集经验"""
        observation, info = self.env.reset()
        episode_reward = 0
        
        while True:
            # 获取动作
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                mean, std = self.trainer.policy_network(obs_tensor)
                action_dist = torch.distributions.Normal(mean, std)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
                value = self.trainer.value_network(obs_tensor).squeeze(-1)
            
            # 执行动作
            next_observation, reward, terminated, truncated, info = self.env.step(
                action.squeeze(0).numpy()
            )
            
            # 存储经验
            self.trainer.experience_buffer.add(
                observation, action.squeeze(0).numpy(), reward,
                next_observation, terminated, truncated,
                log_prob.item(), value.item()
            )
            
            # 更新
            observation = next_observation
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # 记录回合
        self.trainer.log_episode(episode_reward, 0)
        self.trainer.episode_rewards.append(episode_reward)
        
        return episode_reward
    
    def evaluate(self) -> Dict[str, float]:
        """评估最终性能"""
        self.logger.info("Evaluating final performance")
        
        if not self.trainer.episode_rewards:
            return {}
        
        # 计算性能指标
        recent_rewards = self.trainer.episode_rewards[-100:]
        transfer_stats = self.trainer.get_transfer_statistics()
        
        evaluation_metrics = {
            'mean_reward': float(np.mean(recent_rewards)),
            'std_reward': float(np.std(recent_rewards)),
            'max_reward': float(np.max(recent_rewards)),
            'min_reward': float(np.min(recent_rewards)),
            'final_independence_p': self.trainer.independence_scheduler.get_current_p(),
            'total_episodes': len(self.trainer.episode_rewards),
            'source_robots': self.source_robots,
            'target_robot': self.target_robot
        }
        
        # 添加迁移统计
        if transfer_stats:
            evaluation_metrics.update({
                f'transfer_{k}': v for k, v in transfer_stats.items()
            })
        
        self.logger.info(f"Evaluation results: {evaluation_metrics}")
        return evaluation_metrics
    
    def cleanup(self):
        """清理资源"""
        if self.env:
            self.env.close()