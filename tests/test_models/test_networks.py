import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.networks import InputNetwork, OutputNetwork
from models.components import GaussianDistribution


class TestInputNetwork:
    """Test cases for InputNetwork"""
    
    @pytest.fixture
    def input_network_config(self):
        return {
            "node_observation_dim": 10,
            "hidden_dim": 32,
            "shared_across_nodes": True,
            "device": "cpu"
        }
    
    def test_initialization(self, input_network_config):
        """Test input network initialization"""
        network = InputNetwork(input_network_config)
        assert network.node_observation_dim == 10
        assert network.hidden_dim == 32
        assert network.shared_across_nodes == True
    
    def test_forward_pass(self, input_network_config):
        """Test input network forward pass"""
        network = InputNetwork(input_network_config)
        
        # Create dummy node observations
        node_observations = {
            "node1": torch.randn(10),
            "node2": torch.randn(10),
            "node3": torch.randn(10),
        }
        
        # Process observations
        representations = network(node_observations)
        
        # Check output
        for node_id, representation in representations.items():
            assert representation.shape == (32,)  # hidden_dim
            assert representation.dtype == torch.float32
    
    def test_batch_processing(self, input_network_config):
        """Test batch processing"""
        network = InputNetwork(input_network_config)
        
        # Create batch of observations
        batch_observations = torch.randn(8, 5, 10)  # [batch_size, num_nodes, obs_dim]
        
        # Process batch
        representations = network.process_batch(batch_observations)
        
        # Check output shape
        assert representations.shape == (8, 5, 32)  # [batch_size, num_nodes, hidden_dim]


class TestOutputNetwork:
    """Test cases for OutputNetwork"""
    
    @pytest.fixture
    def output_network_config(self):
        return {
            "state_embedding_dim": 64,
            "total_action_dim": 8,
            "learn_std": True,
            "device": "cpu"
        }
    
    def test_initialization(self, output_network_config):
        """Test output network initialization"""
        network = OutputNetwork(output_network_config)
        assert network.state_embedding_dim == 64
        assert network.total_action_dim == 8
        assert network.learn_std == True
    
    def test_forward_pass(self, output_network_config):
        """Test output network forward pass"""
        network = OutputNetwork(output_network_config)
        
        # Create dummy state embedding
        state_embedding = torch.randn(16, 64)  # [batch_size, embedding_dim]
        
        # Get action distribution
        mean, std = network(state_embedding)
        
        # Check output shapes
        assert mean.shape == (16, 8)  # [batch_size, total_action_dim]
        assert std.shape == (16, 8)
        
        # Check that std is positive
        assert torch.all(std > 0)
    
    def test_action_splitting(self, output_network_config):
        """Test action splitting by nodes"""
        network = OutputNetwork(output_network_config)
        
        # Create dummy actions
        actions = torch.randn(4, 8)  # [batch_size, total_action_dim]
        
        # Define node action dimensions
        node_action_dims = {
            "node1": 2,
            "node2": 3,
            "node3": 3,
        }
        
        # Split actions
        node_actions = network.split_actions_by_nodes(actions, node_action_dims)
        
        # Check splitting
        assert "node1" in node_actions
        assert "node2" in node_actions
        assert "node3" in node_actions
        assert node_actions["node1"].shape == (4, 2)
        assert node_actions["node2"].shape == (4, 3)
        assert node_actions["node3"].shape == (4, 3)


class TestDistributions:
    """Test cases for distributions"""
    
    def test_gaussian_distribution(self):
        """Test Gaussian distribution functionality"""
        distribution = GaussianDistribution(action_dim=4)
        
        # Create parameters
        mean = torch.randn(8, 4)  # [batch_size, action_dim]
        std = torch.ones(8, 4) * 0.1
        
        # Test sampling
        samples = distribution.sample(mean, std)
        assert samples.shape == (8, 4)
        
        # Test log probability
        log_prob = distribution.log_prob(samples, mean, std)
        assert log_prob.shape == (8,)
        
        # Test entropy
        entropy = distribution.entropy(mean, std)
        assert entropy.shape == (8,)