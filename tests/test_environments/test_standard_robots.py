import pytest
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from environments import get_standard_robot_env
from environments.base_env import DummyEnv


class TestStandardRobotEnv:
    """Test cases for StandardRobotEnv"""
    
    def test_initialization(self, standard_robot_config):
        """Test environment initialization"""
        StandardRobotEnv = get_standard_robot_env()
        env = StandardRobotEnv(standard_robot_config)
        assert env._is_initialized
        assert hasattr(env, '_env')
        env.close()
        
    def test_supported_robots(self):
        """Test all supported robot types"""
        StandardRobotEnv = get_standard_robot_env()
        for robot_type in StandardRobotEnv.SUPPORTED_ROBOTS.keys():
            config = {"robot_type": robot_type}
            env = StandardRobotEnv(config)
            assert env._is_initialized
            env.close()
    
    def test_unsupported_robot(self):
        """Test error for unsupported robot type"""
        StandardRobotEnv = get_standard_robot_env()
        config = {"robot_type": "UnsupportedRobot"}
        with pytest.raises(ValueError):
            StandardRobotEnv(config)
    
    def test_reset_step(self, standard_robot_config):
        """Test reset and step functionality"""
        StandardRobotEnv = get_standard_robot_env()
        env = StandardRobotEnv(standard_robot_config)
        
        # Test reset
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        
        # Test step
        action = np.random.uniform(-1, 1, env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_observation_action_spaces(self, standard_robot_config):
        """Test observation and action spaces"""
        StandardRobotEnv = get_standard_robot_env()
        env = StandardRobotEnv(standard_robot_config)
        
        obs_space = env.observation_space
        action_space = env.action_space
        
        assert hasattr(obs_space, 'shape')
        assert hasattr(action_space, 'shape')
        
        # Test that we can sample from spaces
        obs_sample = obs_space.sample()
        action_sample = action_space.sample()
        
        assert obs_sample.shape == obs_space.shape
        assert action_sample.shape == action_space.shape
        
        env.close()
    
    def test_robot_info(self, standard_robot_config):
        """Test robot information retrieval"""
        StandardRobotEnv = get_standard_robot_env()
        env = StandardRobotEnv(standard_robot_config)
        info = env.get_robot_info()
        
        assert "robot_type" in info
        assert "observation_dim" in info
        assert "action_dim" in info
        assert "is_standard_robot" in info
        
        env.close()


class TestDummyEnv:
    """Test cases for DummyEnv (for basic interface testing)"""
    
    def test_dummy_env(self, dummy_env_config):
        """Test dummy environment functionality"""
        env = DummyEnv(dummy_env_config)
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (dummy_env_config["observation_dim"],)
        assert "reset" in info
        
        # Test step
        action = np.random.randn(dummy_env_config["action_dim"])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (dummy_env_config["observation_dim"],)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "step" in info
        
        env.close()