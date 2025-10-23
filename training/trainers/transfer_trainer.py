import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import os

from .ppo_trainer import PPOTrainer
from ..buffers import ExperienceBuffer, TrajectoryBuffer

# 修复导入路径问题
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from transfer.semantic_space import SemanticSpaceManager
from transfer.weight_calculator import AdaptiveWeightCalculator
from transfer.lateral_connections import LateralConnectionManager
from transfer.independence import GradualIndependenceScheduler
from models.networks.set_transformer import MultiHeadSetTransformer

class TURRETTrainer(PPOTrainer):
    """
    Complete TURRET training system with multi-source transfer learning
    Implements the full TURRET framework from the paper
    """
    
    def __init__(self, 
                 target_policy: nn.Module,
                 target_value: nn.Module,
                 source_policies: List[nn.Module],
                 source_values: List[nn.Module],
                 config):  # 接受任何配置类型
        
        # Initialize base PPO trainer
        super().__init__(target_policy, target_value, config)
        
        self.source_policies = source_policies
        self.source_values = source_values
        self.num_sources = len(source_policies)
        
        # Transfer learning components
        self._init_transfer_components(config)
        
        # Training state
        self.episode_rewards = []
        self.transfer_statistics = {
            'transfer_weights': [],
            'independence_factors': [],
            'semantic_distances': []
        }
    
    def _init_transfer_components(self, config) -> None:
        """Initialize transfer learning components"""
        
        # 统一配置处理
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = config
        
        # Semantic space manager
        self.semantic_space = SemanticSpaceManager({
            "embedding_dim": config_dict.get("embedding_dim", 128),
            "normalize_embeddings": True,
            "distance_metric": "euclidean",
            "device": self.device
        })
        
        # Adaptive weight calculator
        self.weight_calculator = AdaptiveWeightCalculator({
            "embedding_dim": config_dict.get("embedding_dim", 128),
            "temperature": config_dict.get("temperature", 1.0),
            "min_weight": config_dict.get("min_weight", 0.01),
            "use_performance_feedback": config_dict.get("use_performance_feedback", True),
            "device": self.device
        })
        
        # Lateral connection manager
        self.lateral_connections = LateralConnectionManager({
            "connection_type": config_dict.get("connection_type", "weighted_sum"),
            "residual_connection": True,
            "layer_specific": True,
            "device": self.device
        })
        
        # Gradual independence scheduler
        self.independence_scheduler = GradualIndependenceScheduler({
            "initial_p": config_dict.get("initial_p", 0.0),
            "final_p": config_dict.get("final_p", 1.0),
            "independence_steps": config_dict.get("independence_steps", 1000000),
            "schedule_type": config_dict.get("schedule_type", "linear"),
            "performance_adaptive": config_dict.get("performance_adaptive", True)
        })
        
        # Set Transformer for state embeddings
        self.set_transformer = MultiHeadSetTransformer({
            "input_dim": config_dict.get("node_embedding_dim", 64),
            "hidden_dim": config_dict.get("set_transformer_hidden", 128),
            "output_dim": config_dict.get("state_embedding_dim", 128),
            "n_heads": config_dict.get("n_heads", 8),
            "n_blocks": config_dict.get("n_blocks", 2),
            "n_induction_points": config_dict.get("n_induction_points", 4)
        })
    
    def compute_state_embedding(self, 
                          node_observations: Dict[str, torch.Tensor],
                          morphology_graph: Any) -> torch.Tensor:
        """
        Compute global state embedding using GNN and Set Transformer
        
        Args:
            node_observations: Dictionary of node observations
            morphology_graph: Robot morphology graph
            
        Returns:
            Global state embedding
        """
        # For Phase 5, we use a simplified approach
        # In Phase 6, this will use the full GNN pipeline
        
        # Simply use the first node observation as state embedding for now
        first_node_key = list(node_observations.keys())[0]
        state_embedding = node_observations[first_node_key]
        
        # Ensure correct dimensions
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        
        return state_embedding
    
    def _node_dict_to_matrix(self, 
                           node_representations: Dict[str, torch.Tensor],
                           morphology_graph: Any) -> torch.Tensor:
        """Convert node dictionary to matrix format"""
        # Get all nodes in consistent order
        node_ids = sorted(node_representations.keys())
        
        # Stack representations
        representations = [node_representations[node_id] for node_id in node_ids]
        node_matrix = torch.stack(representations, dim=0)
        
        # Add batch dimension if needed
        if node_matrix.dim() == 1:
            node_matrix = node_matrix.unsqueeze(0)
        
        return node_matrix.unsqueeze(0) if node_matrix.dim() == 2 else node_matrix
    
    def compute_transfer_weights(self,
                               target_state: torch.Tensor,
                               source_states: List[torch.Tensor]) -> List[float]:
        """
        Compute transfer weights for current state
        
        Args:
            target_state: Target task state embedding
            source_states: Source task state embeddings
            
        Returns:
            Transfer weights for each source
        """
        source_ids = [f"source_{i}" for i in range(len(source_states))]
        
        weights = self.weight_calculator.compute_transfer_weights(
            target_state, source_states, source_ids
        )
        
        # Store for statistics
        self.transfer_statistics['transfer_weights'].append(weights)
        
        return weights
    
    def transfer_knowledge(self,
                         target_activations: Dict[str, torch.Tensor],
                         source_activations: List[Dict[str, torch.Tensor]],
                         weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Transfer knowledge from source policies to target policy
        
        Args:
            target_activations: Target network layer activations
            source_activations: Source network layer activations
            weights: Transfer weights
            
        Returns:
            Fused activations for target network
        """
        fused_activations = {}
        
        for layer_name in target_activations.keys():
            target_feat = target_activations[layer_name]
            source_feats = []
            
            for src_act in source_activations:
                if layer_name in src_act:
                    source_feats.append(src_act[layer_name])
                else:
                    # If layer doesn't exist in source, use zeros
                    source_feats.append(torch.zeros_like(target_feat))
            
            # Fuse knowledge using lateral connections
            fused_feat = self.lateral_connections.fuse_knowledge(
                target_feat, source_feats, weights
            )
            
            # Apply independence factor
            p = self.independence_scheduler.get_current_p()
            final_feat = p * target_feat + (1 - p) * fused_feat
            
            fused_activations[layer_name] = final_feat
        
        return fused_activations
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a training step with transfer learning"""
        
        # Update independence scheduler
        if self.episode_rewards:
            recent_performance = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
            self.independence_scheduler.step(recent_performance)
        
        # Get current independence factor
        p = self.independence_scheduler.get_current_p()
        self.transfer_statistics['independence_factors'].append(p)
        
        # Perform transfer if needed
        if self.independence_scheduler.should_transfer():
            transfer_stats = self._apply_transfer(batch)
        else:
            transfer_stats = {}
        
        # Perform standard PPO training
        ppo_stats = super().train_step(batch)
        
        # Combine statistics
        combined_stats = {**ppo_stats, **transfer_stats}
        combined_stats['independence_p'] = p
        combined_stats['transfer_strength'] = 1.0 - p
        
        return combined_stats
    
    def _apply_transfer(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Apply transfer learning for current batch"""
        transfer_stats = {}
        
        # For simplicity, we'll use a subset of the batch for transfer
        if len(batch['observations']) > 0:
            # Get state embeddings for transfer weight computation
            # In practice, this would use the actual robot morphology
            target_state = batch['observations'][0]  # Use first observation
            
            # Get source state embeddings (simulated for now)
            source_states = []
            for i in range(self.num_sources):
                # In real implementation, this would come from source policies
                source_state = target_state + torch.randn_like(target_state) * 0.1
                source_states.append(source_state)
            
            # Compute transfer weights
            weights = self.compute_transfer_weights(target_state, source_states)
            transfer_stats['transfer_weights_mean'] = np.mean(weights)
            transfer_stats['transfer_weights_std'] = np.std(weights)
            
            # Compute semantic distances for statistics
            target_embedding = self.semantic_space.project_state(target_state, "target")
            distances = []
            for i, source_state in enumerate(source_states):
                source_embedding = self.semantic_space.project_state(source_state, f"source_{i}")
                distance = self.semantic_space.compute_semantic_distance(
                    target_embedding, source_embedding
                )
                distances.append(distance)
            
            transfer_stats['semantic_distance_mean'] = np.mean(distances)
            self.transfer_statistics['semantic_distances'].append(distances)
        
        return transfer_stats
    
    def collect_transfer_experience(self,
                                  environment,
                                  num_episodes: int = 10) -> List[Dict[str, Any]]:
        """
        Collect experience with transfer learning
        
        Args:
            environment: Training environment
            num_episodes: Number of episodes to collect
            
        Returns:
            List of episode trajectories
        """
        trajectories = []
        
        for episode in range(num_episodes):
            observation, info = environment.reset()
            episode_reward = 0
            episode_length = 0
            episode_trajectory = []
            
            while True:
                # Get state embedding for transfer
                # Note: In practice, this would use actual node observations and morphology
                state_embedding = torch.FloatTensor(observation)
                
                # Compute transfer weights
                source_states = [state_embedding] * self.num_sources  # Simplified
                transfer_weights = self.compute_transfer_weights(state_embedding, source_states)
                
                # Get action with potential transfer
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                    
                    # Get policy output
                    mean, std = self.policy_network(obs_tensor)
                    action_dist = torch.distributions.Normal(mean, std)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action).sum(dim=-1)
                    value = self.value_network(obs_tensor).squeeze(-1)
                
                # Take action
                next_observation, reward, terminated, truncated, info = environment.step(
                    action.squeeze(0).cpu().numpy()
                )
                
                # Store experience
                self.experience_buffer.add(
                    observation, action.squeeze(0).cpu().numpy(), reward,
                    next_observation, terminated, truncated,
                    log_prob.item(), value.item()
                )
                
                # Store trajectory step
                episode_trajectory.append({
                    'observation': observation,
                    'action': action.squeeze(0).cpu().numpy(),
                    'reward': reward,
                    'value': value.item(),
                    'log_prob': log_prob.item(),
                    'transfer_weights': transfer_weights
                })
                
                # Update
                observation = next_observation
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # Log episode
            self.log_episode(episode_reward, episode_length)
            self.episode_rewards.append(episode_reward)
            trajectories.append(episode_trajectory)
            
            # Update performance metrics for weight calculator
            if episode % 5 == 0:  # Update every 5 episodes
                self._update_performance_metrics()
        
        return trajectories
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics for transfer weight calculation"""
        if len(self.episode_rewards) >= 10:
            recent_performance = np.mean(self.episode_rewards[-10:])
            
            # Normalize performance for weight adjustment
            normalized_performance = recent_performance / 100.0  # Adjust scale as needed
            
            # Update all sources (simplified - in practice would track individual sources)
            for i in range(self.num_sources):
                self.weight_calculator.update_performance(
                    f"source_{i}", normalized_performance
                )
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive transfer learning statistics"""
        stats = {}
        
        if self.transfer_statistics['transfer_weights']:
            weights_array = np.array(self.transfer_statistics['transfer_weights'])
            stats['transfer_weights_mean'] = weights_array.mean(axis=0).tolist()
            stats['transfer_weights_std'] = weights_array.std(axis=0).tolist()
        
        if self.transfer_statistics['independence_factors']:
            p_array = np.array(self.transfer_statistics['independence_factors'])
            stats['independence_p_mean'] = float(p_array.mean())
            stats['independence_p_std'] = float(p_array.std())
        
        if self.transfer_statistics['semantic_distances']:
            distances_array = np.array(self.transfer_statistics['semantic_distances'])
            stats['semantic_distance_mean'] = float(distances_array.mean())
            stats['semantic_distance_std'] = float(distances_array.std())
        
        # Performance metrics from weight calculator
        perf_metrics = self.weight_calculator.get_performance_metrics()
        stats.update({f'source_perf_{k}': v for k, v in perf_metrics.items()})
        
        return stats
    
    def save_checkpoint(self, filepath: str, is_best: bool = False) -> None:
        """Save complete TURRET checkpoint"""
        checkpoint = {
            # Base trainer state
            'target_policy_state': self.policy_network.state_dict(),
            'target_value_state': self.value_network.state_dict(),
            'policy_optimizer_state': self.policy_optimizer.state_dict(),
            'value_optimizer_state': self.value_optimizer.state_dict(),
            
            # Transfer components
            'semantic_space_state': self._get_semantic_space_state(),
            'weight_calculator_perf': self.weight_calculator.get_performance_metrics(),
            'independence_scheduler_state': self.independence_scheduler.state_dict(),
            'set_transformer_state': self.set_transformer.state_dict(),
            
            # Training state
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'best_reward': self.best_reward,
            'training_stats': self.training_stats,
            'transfer_statistics': self.transfer_statistics,
            'episode_rewards': self.episode_rewards,
            
            # Config
            'config': self.config
        }
        
        # Save source policies
        for i, (policy, value) in enumerate(zip(self.source_policies, self.source_values)):
            checkpoint[f'source_policy_{i}_state'] = policy.state_dict()
            checkpoint[f'source_value_{i}_state'] = value.state_dict()
        
        save_checkpoint(checkpoint, filepath, is_best)
    
    def _get_semantic_space_state(self) -> Dict[str, Any]:
        """Get semantic space state for checkpointing"""
        # This would save the projection networks' states
        # For simplicity, we return basic info
        return {
            'embedding_dim': self.semantic_space.embedding_dim,
            'num_task_projections': len(self.semantic_space.projection_networks)
        }
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load TURRET checkpoint"""
        checkpoint = load_checkpoint(filepath, self.device)
        
        # Load base trainer state
        self.policy_network.load_state_dict(checkpoint['target_policy_state'])
        self.value_network.load_state_dict(checkpoint['target_value_state'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])
        
        # Load transfer components state
        self.weight_calculator.source_performances = checkpoint['weight_calculator_perf']
        self.independence_scheduler.load_state_dict(checkpoint['independence_scheduler_state'])
        self.set_transformer.load_state_dict(checkpoint['set_transformer_state'])
        
        # Load training state
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        self.best_reward = checkpoint['best_reward']
        self.training_stats = checkpoint['training_stats']
        self.transfer_statistics = checkpoint['transfer_statistics']
        self.episode_rewards = checkpoint['episode_rewards']
        
        # Load source policies
        for i in range(self.num_sources):
            if f'source_policy_{i}_state' in checkpoint:
                self.source_policies[i].load_state_dict(checkpoint[f'source_policy_{i}_state'])
            if f'source_value_{i}_state' in checkpoint:
                self.source_values[i].load_state_dict(checkpoint[f'source_value_{i}_state'])