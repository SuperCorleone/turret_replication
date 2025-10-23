# training/buffers.py - 完整版本

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional
import random

class ExperienceBuffer:
    """
    Experience replay buffer for storing and sampling trajectories
    """
    def __init__(self, capacity: int, observation_shape: tuple, action_shape: tuple):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        
        # Storage for experiences
        self.observations = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.terminated = deque(maxlen=capacity)
        self.truncated = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        
        self.position = 0
        self.size = 0

    def add(self, 
            observation: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_observation: np.ndarray,
            terminated: bool,
            truncated: bool,
            log_prob: float,
            value: float) -> None:
        """Add a single experience to the buffer"""
        self.observations.append(observation.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.next_observations.append(next_observation.copy())
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, 
                 observations: List[np.ndarray],
                 actions: List[np.ndarray],
                 rewards: List[float],
                 next_observations: List[np.ndarray],
                 terminated: List[bool],
                 truncated: List[bool],
                 log_probs: List[float],
                 values: List[float]) -> None:
        """Add a batch of experiences to the buffer"""
        for i in range(len(observations)):
            self.add(
                observations[i], actions[i], rewards[i], next_observations[i],
                terminated[i], truncated[i], log_probs[i], values[i]
            )

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences from the buffer"""
        if self.size < batch_size:
            batch_size = self.size
            
        indices = random.sample(range(self.size), batch_size)
        
        batch = {
            'observations': torch.FloatTensor(np.array([self.observations[i] for i in indices])),
            'actions': torch.FloatTensor(np.array([self.actions[i] for i in indices])),
            'rewards': torch.FloatTensor(np.array([self.rewards[i] for i in indices])),
            'next_observations': torch.FloatTensor(np.array([self.next_observations[i] for i in indices])),
            'terminated': torch.BoolTensor(np.array([self.terminated[i] for i in indices])),
            'truncated': torch.BoolTensor(np.array([self.truncated[i] for i in indices])),
            'log_probs': torch.FloatTensor(np.array([self.log_probs[i] for i in indices])),
            'values': torch.FloatTensor(np.array([self.values[i] for i in indices])),
        }
        
        return batch

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """Get all data from buffer (for PPO updates)"""
        return {
            'observations': torch.FloatTensor(np.array(self.observations)),
            'actions': torch.FloatTensor(np.array(self.actions)),
            'rewards': torch.FloatTensor(np.array(self.rewards)),
            'next_observations': torch.FloatTensor(np.array(self.next_observations)),
            'terminated': torch.BoolTensor(np.array(self.terminated)),
            'truncated': torch.BoolTensor(np.array(self.truncated)),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)),
            'values': torch.FloatTensor(np.array(self.values)),
        }

    def clear(self) -> None:
        """Clear the buffer"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.log_probs.clear()
        self.values.clear()
        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        return self.size

    def is_full(self) -> bool:
        return self.size == self.capacity


class TrajectoryBuffer:
    """
    Buffer for storing complete trajectories (for advantage calculation)
    """
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.trajectories = []
        self.current_trajectory = []

    def add_step(self, 
                 observation: np.ndarray,
                 action: np.ndarray,
                 reward: float,
                 value: float,
                 log_prob: float) -> None:
        """Add a step to the current trajectory"""
        self.current_trajectory.append({
            'observation': observation,
            'action': action,
            'reward': reward,
            'value': value,
            'log_prob': log_prob
        })

    def end_trajectory(self, last_value: float = 0.0) -> List[Dict[str, Any]]:
        """End current trajectory and compute advantages/returns"""
        if not self.current_trajectory:
            return []
            
        trajectory = self._compute_advantages_and_returns(last_value)
        self.trajectories.append(trajectory)
        self.current_trajectory = []
        return trajectory

    def _compute_advantages_and_returns(self, last_value: float) -> List[Dict[str, Any]]:
        """Compute advantages and returns using GAE"""
        trajectory = self.current_trajectory.copy()
        advantages = []
        returns = []
        
        # Compute advantages using GAE
        gae = 0
        for i in reversed(range(len(trajectory))):
            if i == len(trajectory) - 1:
                next_value = last_value
                next_non_terminal = 1.0
            else:
                next_value = trajectory[i+1]['value']
                next_non_terminal = 1.0  # Assuming no early termination for simplicity
                
            delta = trajectory[i]['reward'] + self.gamma * next_value * next_non_terminal - trajectory[i]['value']
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        # Compute returns
        for i in range(len(trajectory)):
            returns.append(advantages[i] + trajectory[i]['value'])
            trajectory[i]['advantage'] = advantages[i]
            trajectory[i]['return'] = returns[i]
            
        return trajectory

    def get_all_trajectories(self) -> List[Dict[str, Any]]:
        """Get all completed trajectories"""
        return self.trajectories

    def clear(self) -> None:
        """Clear all trajectories"""
        self.trajectories.clear()
        self.current_trajectory.clear()