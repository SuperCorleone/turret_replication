import pytest
import numpy as np
from configs.base_config import Phase1Config


@pytest.fixture
def basic_config():
    """Fixture for basic configuration"""
    return Phase1Config(env_name="HalfCheetah-v4", seed=42)


@pytest.fixture
def dummy_env_config():
    """Fixture for dummy environment configuration"""
    return {
        "observation_dim": 10,
        "action_dim": 4,
        "max_steps": 50
    }


@pytest.fixture
def standard_robot_config():
    """Fixture for standard robot configuration"""
    return {
        "robot_type": "HalfCheetah",
        "max_episode_steps": 100
    }