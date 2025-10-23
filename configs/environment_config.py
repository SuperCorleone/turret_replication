from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    
    # MuJoCo settings
    frame_skip: int = 1
    camera_name: str = "track"
    render_mode: str = "rgb_array"  # "human", "rgb_array", "depth_array"
    
    # Observation and action spaces
    normalize_observations: bool = False
    clip_actions: bool = True
    
    # Robot morphology (for future use)
    robot_types: List[str] = None
    
    def __post_init__(self):
        if self.robot_types is None:
            self.robot_types = ["HalfCheetah", "Ant", "Hopper", "Walker2d", "Humanoid"]


@dataclass 
class CentipedeConfig(EnvironmentConfig):
    """Centipede-specific configuration"""
    
    num_segments: int = 4
    num_legs: int = 8
    segment_mass: float = 1.0
    leg_length: float = 0.5