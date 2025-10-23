from typing import Dict, Any, List
from ..mujoco_wrapper import MuJoCoWrapper


class StandardRobotEnv(MuJoCoWrapper):
    """Wrapper for standard MuJoCo robot environments"""
    
    # Supported robot environments
    SUPPORTED_ROBOTS = {
        "HalfCheetah": "HalfCheetah-v4",
        "Ant": "Ant-v4", 
        "Hopper": "Hopper-v4",
        "Walker2d": "Walker2d-v4",
        "Humanoid": "Humanoid-v4"
    }
    
    def __init__(self, config: Dict[str, Any]):
        robot_type = config.get("robot_type", "HalfCheetah")
        
        if robot_type not in self.SUPPORTED_ROBOTS:
            raise ValueError(f"Unsupported robot type: {robot_type}. "
                           f"Supported: {list(self.SUPPORTED_ROBOTS.keys())}")
        
        # Convert robot type to Gymnasium environment name
        config["env_name"] = self.SUPPORTED_ROBOTS[robot_type]
        config["robot_type"] = robot_type
        
        super().__init__(config)
        
    def get_robot_info(self) -> Dict[str, Any]:
        """Get information about the robot morphology"""
        robot_type = self.config["robot_type"]
        
        # Basic robot information (will be extended in Phase 2)
        info = {
            "robot_type": robot_type,
            "observation_dim": self.observation_space.shape[0],
            "action_dim": self.action_space.shape[0],
            "is_standard_robot": True
        }
        
        return info