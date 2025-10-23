import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from training.trainers.transfer_trainer import TURRETTrainer
from models.policies.structured_policy import StructuredPolicyNetwork


class SimpleValueNetwork(torch.nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.network = torch.nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.network(x)


class TestTURRETTrainer:
    """Test cases for TURRETTrainer"""
    
    @pytest.fixture
    def trainer_config(self):
        return {
            "device": "cpu",
            "embedding_dim": 64,
            "temperature": 1.0,
            "min_weight": 0.01,
            "initial_p": 0.0,
            "final_p": 1.0,
            "independence_steps": 100,
            "node_observation_dim": 10,
            "hidden_dim": 32,
            "output_dim": 4
        }
    
    @pytest.fixture
    def sample_networks(self, trainer_config):
        # Create target networks
        target_policy = StructuredPolicyNetwork({
            "node_observation_dim": trainer_config["node_observation_dim"],
            "hidden_dim": trainer_config["hidden_dim"],
            "output_dim": trainer_config["output_dim"],
            "shared_across_nodes": True,
            "device": "cpu"
        })
        target_value = SimpleValueNetwork(10)
        
        # Create source networks
        source_policies = [
            StructuredPolicyNetwork({
                "node_observation_dim": trainer_config["node_observation_dim"],
                "hidden_dim": trainer_config["hidden_dim"],
                "output_dim": trainer_config["output_dim"],
                "shared_across_nodes": True,
                "device": "cpu"
            }) for _ in range(2)
        ]
        source_values = [SimpleValueNetwork(10) for _ in range(2)]
        
        return target_policy, target_value, source_policies, source_values
    
    def test_initialization(self, trainer_config, sample_networks):
        """Test TURRET trainer initialization"""
        target_policy, target_value, source_policies, source_values = sample_networks
        
        trainer = TURRETTrainer(
            target_policy, target_value, source_policies, source_values, trainer_config
        )
        
        assert trainer is not None
        assert trainer.num_sources == 2
        assert hasattr(trainer, 'semantic_space')
        assert hasattr(trainer, 'weight_calculator')
        assert hasattr(trainer, 'lateral_connections')
        assert hasattr(trainer, 'independence_scheduler')
    
    def test_transfer_weight_computation(self, trainer_config, sample_networks):
        """Test transfer weight computation"""
        target_policy, target_value, source_policies, source_values = sample_networks
        trainer = TURRETTrainer(
            target_policy, target_value, source_policies, source_values, trainer_config
        )
        
        target_state = torch.randn(64)
        source_states = [torch.randn(64), torch.randn(64)]
        
        weights = trainer.compute_transfer_weights(target_state, source_states)
        
        assert len(weights) == 2
        assert all(w >= 0 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-6
    
    def test_training_step(self, trainer_config, sample_networks):
        """Test training step with transfer"""
        target_policy, target_value, source_policies, source_values = sample_networks
        trainer = TURRETTrainer(
            target_policy, target_value, source_policies, source_values, trainer_config
        )
        
        # Create dummy batch
        batch = {
            'observations': torch.randn(32, 10),
            'actions': torch.randn(32, 4),
            'rewards': torch.randn(32),
            'next_observations': torch.randn(32, 10),
            'terminated': torch.zeros(32, dtype=torch.bool),
            'truncated': torch.zeros(32, dtype=torch.bool),
            'log_probs': torch.randn(32),
            'values': torch.randn(32),
        }
        
        stats = trainer.train_step(batch)
        
        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'independence_p' in stats
        assert 'transfer_strength' in stats
    
    def test_transfer_statistics(self, trainer_config, sample_networks):
        """Test transfer statistics collection"""
        target_policy, target_value, source_policies, source_values = sample_networks
        trainer = TURRETTrainer(
            target_policy, target_value, source_policies, source_values, trainer_config
        )
        
        # Add some dummy data to statistics
        trainer.transfer_statistics['transfer_weights'].append([0.6, 0.4])
        trainer.transfer_statistics['independence_factors'].append(0.5)
        trainer.transfer_statistics['semantic_distances'].append([0.1, 0.2])
        
        stats = trainer.get_transfer_statistics()
        
        assert 'transfer_weights_mean' in stats
        assert 'independence_p_mean' in stats
        assert 'semantic_distance_mean' in stats