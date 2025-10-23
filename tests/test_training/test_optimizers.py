import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from training.optimizers import LearningRateScheduler, GradientManager


class TestLearningRateScheduler:
    """Test cases for LearningRateScheduler"""
    
    @pytest.fixture
    def optimizer(self):
        model = torch.nn.Linear(10, 2)
        return torch.optim.Adam(model.parameters(), lr=0.001)
    
    def test_constant_scheduler(self, optimizer):
        """Test constant learning rate scheduler"""
        config = {
            "scheduler_type": "constant",
            "learning_rate": 0.001
        }
        
        scheduler = LearningRateScheduler(optimizer, config)
        initial_lr = scheduler.get_current_lr()
        
        # Step multiple times
        for _ in range(10):
            scheduler.step()
        
        # LR should remain constant
        assert scheduler.get_current_lr() == initial_lr
    
    def test_linear_scheduler(self, optimizer):
        """Test linear learning rate scheduler"""
        config = {
            "scheduler_type": "linear",
            "learning_rate": 0.001,
            "decay_steps": 10,
            "min_lr": 0.0001
        }
        
        scheduler = LearningRateScheduler(optimizer, config)
        initial_lr = scheduler.get_current_lr()
        
        # Step through decay
        for i in range(10):
            scheduler.step()
        
        # LR should have decayed
        final_lr = scheduler.get_current_lr()
        assert final_lr < initial_lr
        assert final_lr >= config["min_lr"]


class TestGradientManager:
    """Test cases for GradientManager"""
    
    @pytest.fixture
    def model_with_grads(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        # Create some gradients
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)
        loss = torch.nn.MSELoss()(model(x), y)
        loss.backward()
        
        return model
    
    def test_grad_norm_computation(self, model_with_grads):
        """Test gradient norm computation"""
        parameters = list(model_with_grads.parameters())
        grad_norm = GradientManager.compute_grad_norm(parameters)
        
        assert isinstance(grad_norm, float)
        assert grad_norm > 0
    
    def test_grad_clipping(self, model_with_grads):
        """Test gradient clipping"""
        parameters = list(model_with_grads.parameters())
        
        # Store original gradients
        original_grads = [p.grad.clone() for p in parameters]
        
        # Clip gradients
        max_norm = 0.1
        total_norm = GradientManager.clip_grad_norm(parameters, max_norm)
        
        # Check that gradients were clipped
        assert total_norm <= max_norm
        
        # Check that gradients are still valid
        for p in parameters:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()