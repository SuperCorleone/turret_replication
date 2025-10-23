# import torch
# import numpy as np
# from typing import Dict, Any, List
# import os
# import json

# from training.trainers.ppo_trainer import PPOTrainer
# from models.policies.structured_policy import StructuredPolicyNetwork


# class BaselinePPO:
#     """
#     Baseline PPO implementation for comparison
#     """
    
#     def __init__(self, config: Dict[str, Any]):
#         self.config = config
#         self.env = None
#         self.trainer = None
    
#     def setup(self, env) -> None:
#         """Setup baseline PPO"""
#         self.env = env
        
#         # Create policy and value networks
#         policy_net = StructuredPolicyNetwork({
#             "node_observation_dim": env.observation_space.shape[0],
#             "hidden_dim": self.config.get("hidden_dim", 256),
#             "output_dim": env.action_space.shape[0],
#             "shared_across_nodes": True,
#             "device": self.config.get("device", "cpu")
#         })
        
#         value_net = self._create_value_network(env.observation_space.shape[0])
        
#         # Create PPO trainer
#         trainer_config = {
#             **self.config,
#             "observation_shape": env.observation_space.shape,
#             "action_shape": env.action_space.shape,
#         }
        
#         self.trainer = PPOTrainer(policy_net, value_net, trainer_config)
    
#     def _create_value_network(self, input_dim: int) -> torch.nn.Module:
#         """Create value network"""
#         return torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 1)
#         )
    
#     def train(self, total_episodes: int) -> Dict[str, Any]:
#         """Train baseline PPO"""
#         training_curves = []
        
#         for episode in range(total_episodes):
#             # Collect experience
#             episode_reward = self._collect_episode_experience()
            
#             # Train on collected experience
#             if len(self.trainer.experience_buffer) >= self.config.get("batch_size", 64):
#                 batch = self.trainer.experience_buffer.sample(
#                     self.config.get("batch_size", 64)
#                 )
#                 training_stats = self.trainer.train_step(batch)
                
#                 training_curves.append({
#                     'episode': episode,
#                     'reward': episode_reward,
#                     **training_stats
#                 })
            
#             if episode % 100 == 0:
#                 print(f"Baseline PPO Episode {episode}: reward={episode_reward:.2f}")
        
#         return {
#             'training_curves': training_curves,
#             'final_performance': self._evaluate()
#         }
    
#     def _collect_episode_experience(self) -> float:
#         """Collect experience for one episode"""
#         observation, info = self.env.reset()
#         episode_reward = 0
        
#         while True:
#             with torch.no_grad():
#                 obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
#                 mean, std = self.trainer.policy_network(obs_tensor)
#                 action_dist = torch.distributions.Normal(mean, std)
#                 action = action_dist.sample()
#                 log_prob = action_dist.log_prob(action).sum(dim=-1)
#                 value = self.trainer.value_network(obs_tensor).squeeze(-1)
            
#             next_observation, reward, terminated, truncated, info = self.env.step(
#                 action.squeeze(0).numpy()
#             )
            
#             self.trainer.experience_buffer.add(
#                 observation, action.squeeze(0).numpy(), reward,
#                 next_observation, terminated, truncated,
#                 log_prob.item(), value.item()
#             )
            
#             observation = next_observation
#             episode_reward += reward
            
#             if terminated or truncated:
#                 break
        
#         self.trainer.log_episode(episode_reward, 0)
#         return episode_reward
    
#     def _evaluate(self) -> Dict[str, float]:
#         """Evaluate baseline performance"""
#         if not hasattr(self.trainer, 'training_stats') or not self.trainer.training_stats['episode_rewards']:
#             return {}
        
#         rewards = self.trainer.training_stats['episode_rewards']
#         recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        
#         return {
#             'mean_reward': float(np.mean(recent_rewards)),
#             'std_reward': float(np.std(recent_rewards)),
#             'max_reward': float(np.max(recent_rewards)),
#             'min_reward': float(np.min(recent_rewards))
#         }


# def load_baseline_results(baseline_dir: str = "data/baselines") -> Dict[str, Any]:
#     """Load pre-computed baseline results"""
#     baseline_results = {}
    
#     if not os.path.exists(baseline_dir):
#         print(f"Baseline directory not found: {baseline_dir}")
#         return {}
    
#     # Look for baseline result files
#     for filename in os.listdir(baseline_dir):
#         if filename.endswith('_results.json'):
#             baseline_name = filename.replace('_results.json', '')
#             filepath = os.path.join(baseline_dir, filename)
            
#             with open(filepath, 'r') as f:
#                 baseline_results[baseline_name] = json.load(f)
    
#     return baseline_results


# def create_baseline_comparison(experiment_results: Dict[str, Any],
#                               baseline_results: Dict[str, Any]) -> Dict[str, Any]:
#     """Create comparison between experiment and baseline results"""
#     comparison = {}
    
#     for exp_name, exp_data in experiment_results.items():
#         exp_final = exp_data.get('final_performance', {})
#         exp_reward = exp_final.get('mean_reward', 0)
        
#         comparison[exp_name] = {
#             'experiment_performance': exp_final,
#             'baseline_comparisons': {}
#         }
        
#         for baseline_name, baseline_data in baseline_results.items():
#             baseline_final = baseline_data.get('final_performance', {})
#             baseline_reward = baseline_final.get('mean_reward', 0)
            
#             if baseline_reward != 0:
#                 improvement = (exp_reward - baseline_reward) / abs(baseline_reward)
#             else:
#                 improvement = exp_reward - baseline_reward
            
#             comparison[exp_name]['baseline_comparisons'][baseline_name] = {
#                 'baseline_performance': baseline_final,
#                 'improvement_ratio': improvement,
#                 'absolute_improvement': exp_reward - baseline_reward
#             }
    
#     return comparison
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import os
import json

from training.trainers.ppo_trainer import PPOTrainer
from models.policies.structured_policy import StructuredPolicyNetwork
from environments import get_standard_robot_env


class BaselinePPO:
    """
    Baseline PPO implementation for comparison with TURRET
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = None
        self.trainer = None
        
    def setup(self, robot_type: str) -> None:
        """Setup baseline PPO for a specific robot"""
        StandardRobotEnv = get_standard_robot_env()
        self.env = StandardRobotEnv({
            "robot_type": robot_type,
            "max_episode_steps": self.config.get("max_episode_steps", 1000)
        })
        
        # Create policy and value networks
        policy_net = StructuredPolicyNetwork({
            "node_observation_dim": self.env.observation_space.shape[0],
            "hidden_dim": self.config.get("hidden_dim", 256),
            "output_dim": self.env.action_space.shape[0],
            "shared_across_nodes": True,
            "device": self.config.get("device", "cpu")
        })
        
        value_net = self._create_value_network(self.env.observation_space.shape[0])
        
        # Create PPO trainer
        trainer_config = {
            "device": self.config.get("device", "cpu"),
            "ppo_epochs": self.config.get("ppo_epochs", 10),
            "mini_batch_size": self.config.get("mini_batch_size", 64),
            "clip_epsilon": self.config.get("clip_epsilon", 0.2),
            "value_coef": self.config.get("value_coef", 0.5),
            "entropy_coef": self.config.get("entropy_coef", 0.01),
            "max_grad_norm": self.config.get("max_grad_norm", 0.5),
            "learning_rate": self.config.get("learning_rate", 3e-4),
            "buffer_size": self.config.get("buffer_size", 10000),
            "observation_shape": self.env.observation_space.shape,
            "action_shape": self.env.action_space.shape,
            "gamma": self.config.get("gamma", 0.99),
            "gae_lambda": self.config.get("gae_lambda", 0.95),
        }
        
        self.trainer = PPOTrainer(policy_net, value_net, trainer_config)
    
    def _create_value_network(self, input_dim: int) -> torch.nn.Module:
        """Create value network for baseline"""
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    
    def train(self, total_episodes: int = 500) -> List[Dict[str, Any]]:
        """Train baseline PPO"""
        training_curves = []
        
        for episode in range(total_episodes):
            episode_reward = self._collect_episode_experience()
            
            # Train periodically
            if episode % 10 == 0 and len(self.trainer.experience_buffer) >= 64:
                batch = self.trainer.experience_buffer.sample(64)
                stats = self.trainer.train_step(batch)
                
                training_curves.append({
                    'episode': episode,
                    'reward': episode_reward,
                    **stats
                })
            
            if episode % 100 == 0:
                print(f"Baseline PPO Episode {episode}: reward={episode_reward:.2f}")
        
        return training_curves
    
    def _collect_episode_experience(self) -> float:
        """Collect experience for one episode"""
        observation, info = self.env.reset()
        episode_reward = 0
        
        while True:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                mean, std = self.trainer.policy_network(obs_tensor)
                action_dist = torch.distributions.Normal(mean, std)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
                value = self.trainer.value_network(obs_tensor).squeeze(-1)
            
            # Take action
            next_observation, reward, terminated, truncated, info = self.env.step(
                action.squeeze(0).numpy()
            )
            
            # Store experience
            self.trainer.experience_buffer.add(
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
        self.trainer.log_episode(episode_reward, 0)
        return episode_reward
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.env:
            self.env.close()


def load_baseline_results(results_dir: str = "data/results/baselines") -> Dict[str, Any]:
    """Load baseline results from file"""
    baseline_results = {}
    
    if not os.path.exists(results_dir):
        return {}
    
    # Look for baseline result files
    for filename in os.listdir(results_dir):
        if filename.endswith("_baseline.json"):
            robot_type = filename.replace("_baseline.json", "")
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'r') as f:
                baseline_results[robot_type] = json.load(f)
    
    return baseline_results


def run_baseline_experiment(robot_type: str, 
                          config: Dict[str, Any],
                          save_dir: str = "data/results/baselines") -> Dict[str, Any]:
    """Run baseline PPO experiment for a robot type"""
    os.makedirs(save_dir, exist_ok=True)
    
    baseline = BaselinePPO(config)
    baseline.setup(robot_type)
    
    training_curves = baseline.train(config.get("total_episodes", 500))
    
    # Calculate final metrics
    rewards = [curve['reward'] for curve in training_curves]
    final_metrics = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'max_reward': float(np.max(rewards)),
        'min_reward': float(np.min(rewards)),
        'total_episodes': len(rewards)
    }
    
    # Save results
    results = {
        'training_curves': training_curves,
        'final_metrics': final_metrics,
        'config': config
    }
    
    results_path = os.path.join(save_dir, f"{robot_type}_baseline.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    baseline.cleanup()
    
    print(f"Baseline experiment for {robot_type} completed")
    return results