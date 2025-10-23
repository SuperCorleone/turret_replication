import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_env import BaseEnv


class MuJoCoWrapper(BaseEnv):
    """Wrapper for MuJoCo environments using Gymnasium"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.env_name = config["env_name"]
        self.max_episode_steps = config.get("max_episode_steps", 1000)
        self.render_mode = config.get("render_mode", "rgb_array")
        
        # Initialize the environment
        self._env = gym.make(
            self.env_name,
            max_episode_steps=self.max_episode_steps,
            render_mode=self.render_mode
        )
        self._is_initialized = True
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._env.reset(seed=seed)
        obs, info = self._env.reset()
        return obs.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def close(self) -> None:
        if self._is_initialized:
            self._env.close()
            self._is_initialized = False
    
    def render(self) -> Optional[np.ndarray]:
        return self._env.render()
    
    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space
    
    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space
    
    @property
    def unwrapped(self) -> gym.Env:
        """Get the underlying Gym environment"""
        return self._env