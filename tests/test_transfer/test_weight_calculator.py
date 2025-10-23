import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from transfer.weight_calculator import AdaptiveWeightCalculator


class TestAdaptiveWeightCalculator:
    """Test cases for AdaptiveWeightCalculator"""
    
    @pytest.fixture
    def weight_config(self):
        return {
            "embedding_dim": 64,
            "temperature": 1.0,
            "min_weight": 0.01,
            "use_performance_feedback": False,
            "device": "cpu"
        }
    
    def test_initialization(self, weight_config):
        """Test weight calculator initialization"""
        calculator = AdaptiveWeightCalculator(weight_config)
        assert calculator.temperature == 1.0
        assert calculator.min_weight == 0.01
    
    def test_weight_computation(self, weight_config):
        """Test transfer weight computation"""
        calculator = AdaptiveWeightCalculator(weight_config)
        
        target_state = torch.randn(32)
        source_states = [torch.randn(32), torch.randn(32), torch.randn(32)]
        
        weights = calculator.compute_transfer_weights(target_state, source_states)
        
        assert len(weights) == 3
        assert all(w >= 0 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-6  # Should sum to 1
    
    def test_batch_weight_computation(self, weight_config):
        """Test batch weight computation"""
        calculator = AdaptiveWeightCalculator(weight_config)
        
        batch_size = 4
        target_states = torch.randn(batch_size, 32)
        source_states_list = [
            torch.randn(batch_size, 32),
            torch.randn(batch_size, 32)
        ]
        
        weight_matrix = calculator.compute_batch_weights(
            target_states, source_states_list
        )
        
        assert weight_matrix.shape == (batch_size, 2)
        assert torch.all(weight_matrix >= 0)
        assert torch.allclose(weight_matrix.sum(dim=1), torch.ones(batch_size))
    
    def test_performance_tracking(self, weight_config):
        """Test performance tracking"""
        calculator = AdaptiveWeightCalculator(weight_config)
        
        calculator.update_performance("source1", 0.8)
        calculator.update_performance("source2", 0.5)
        
        metrics = calculator.get_performance_metrics()
        assert "source1" in metrics
        assert "source2" in metrics
        assert metrics["source1"] == 0.8