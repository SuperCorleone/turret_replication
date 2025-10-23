from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class BaseEnv(ABC):
    """Abstract base class for all environments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._is_initialized = False
        
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the environment and release resources"""
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """Get observation space"""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Any:
        """Get action space"""
        pass
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment (optional)"""
        return None
    
    def seed(self, seed: int) -> None:
        """Set random seed"""
        pass


class DummyEnv(BaseEnv):
    """Dummy environment for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.observation_dim = config.get("observation_dim", 10)
        self.action_dim = config.get("action_dim", 4)
        self.step_count = 0
        self.max_steps = config.get("max_steps", 100)
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.step_count = 0
        obs = np.random.randn(self.observation_dim).astype(np.float32)
        info = {"reset": True}
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        obs = np.random.randn(self.observation_dim).astype(np.float32)
        reward = float(np.sum(action**2))  # Simple reward based on action magnitude
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {"step": self.step_count}
        return obs, reward, terminated, truncated, info
    
    def close(self) -> None:
        self._is_initialized = False
    
    @property
    def observation_space(self) -> Dict:
        return {
            "shape": (self.observation_dim,),
            "dtype": np.float32
        }
    
    @property
    def action_space(self) -> Dict:
        return {
            "shape": (self.action_dim,),
            "dtype": np.float32,
            "low": -1.0,
            "high": 1.0
        }