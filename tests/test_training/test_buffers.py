import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from training.buffers import ExperienceBuffer, TrajectoryBuffer


class TestExperienceBuffer:
    """Test cases for ExperienceBuffer"""
    
    @pytest.fixture
    def buffer_config(self):
        return {
            "capacity": 100,
            "observation_shape": (10,),
            "action_shape": (4,)
        }
    
    def test_initialization(self, buffer_config):
        """Test buffer initialization"""
        buffer = ExperienceBuffer(**buffer_config)
        assert buffer.capacity == 100
        assert len(buffer) == 0
        assert not buffer.is_full()
    
    def test_add_experience(self, buffer_config):
        """Test adding experiences"""
        buffer = ExperienceBuffer(**buffer_config)
        
        # Add single experience
        obs = np.random.randn(10)
        action = np.random.randn(4)
        buffer.add(obs, action, 1.0, obs, False, False, 0.0, 0.5)
        
        assert len(buffer) == 1
        assert np.array_equal(buffer.observations[0], obs)
        assert buffer.rewards[0] == 1.0
    
    def test_add_batch(self, buffer_config):
        """Test adding batch of experiences"""
        buffer = ExperienceBuffer(**buffer_config)
        
        batch_size = 5
        observations = [np.random.randn(10) for _ in range(batch_size)]
        actions = [np.random.randn(4) for _ in range(batch_size)]
        
        buffer.add_batch(
            observations, actions,
            [1.0] * batch_size, observations,
            [False] * batch_size, [False] * batch_size,
            [0.0] * batch_size, [0.5] * batch_size
        )
        
        assert len(buffer) == batch_size
    
    def test_sampling(self, buffer_config):
        """Test experience sampling"""
        buffer = ExperienceBuffer(**buffer_config)
        
        # Add some experiences
        for i in range(10):
            obs = np.random.randn(10)
            action = np.random.randn(4)
            buffer.add(obs, action, float(i), obs, False, False, 0.0, 0.5)
        
        # Sample batch
        batch = buffer.sample(5)
        
        assert isinstance(batch, dict)
        assert batch['observations'].shape == (5, 10)
        assert batch['actions'].shape == (5, 4)
        assert batch['rewards'].shape == (5,)
        assert batch['log_probs'].shape == (5,)
    
    def test_clear(self, buffer_config):
        """Test buffer clearing"""
        buffer = ExperienceBuffer(**buffer_config)
        
        # Add experiences
        for i in range(5):
            obs = np.random.randn(10)
            action = np.random.randn(4)
            buffer.add(obs, action, 1.0, obs, False, False, 0.0, 0.5)
        
        assert len(buffer) == 5
        buffer.clear()
        assert len(buffer) == 0


class TestTrajectoryBuffer:
    """Test cases for TrajectoryBuffer"""
    
    @pytest.fixture
    def trajectory_buffer(self):
        return TrajectoryBuffer(gamma=0.99, gae_lambda=0.95)
    
    def test_add_steps(self, trajectory_buffer):
        """Test adding steps to trajectory"""
        obs = np.random.randn(10)
        action = np.random.randn(4)
        
        trajectory_buffer.add_step(obs, action, 1.0, 0.5, 0.0)
        trajectory_buffer.add_step(obs, action, 1.0, 0.5, 0.0)
        
        assert len(trajectory_buffer.current_trajectory) == 2
    
    def test_end_trajectory(self, trajectory_buffer):
        """Test ending trajectory and computing advantages"""
        # Add steps
        for i in range(3):
            obs = np.random.randn(10)
            action = np.random.randn(4)
            trajectory_buffer.add_step(obs, action, 1.0, 0.5, 0.0)
        
        # End trajectory
        trajectory = trajectory_buffer.end_trajectory(last_value=0.0)
        
        assert len(trajectory) == 3
        assert 'advantage' in trajectory[0]
        assert 'return' in trajectory[0]
        assert len(trajectory_buffer.trajectories) == 1
        assert len(trajectory_buffer.current_trajectory) == 0