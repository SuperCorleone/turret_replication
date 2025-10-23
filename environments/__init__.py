# environments/__init__.py
from typing import Dict, Any
from .base_env import BaseEnv, DummyEnv
from .mujoco_wrapper import MuJoCoWrapper
from .tasks.standard_robots import StandardRobotEnv
from .tasks.centipede import CentipedeEnv, CentipedeRobotFactory

def get_robot_env(robot_type: str, config: Dict[str, Any] = None):
    """
    统一的机器人环境工厂函数
    支持标准机器人和Centipede机器人
    """
    if config is None:
        config = {}
    
    # 检查是否是Centipede机器人
    if robot_type.startswith("Centipede-"):
        try:
            num_legs = int(robot_type.split("-")[1])
            return CentipedeRobotFactory.create_centipede_env(num_legs, config)
        except (ValueError, IndexError):
            raise ValueError(f"Invalid Centipede format: {robot_type}. Use 'Centipede-4', 'Centipede-6', etc.")
    
    # 标准机器人
    return StandardRobotEnv({**config, "robot_type": robot_type})

# 保持向后兼容 - 返回实例而不是类
def get_standard_robot_env(config: Dict[str, Any] = None):
    """获取标准机器人环境实例"""
    if config is None:
        config = {}
    
    robot_type = config.get("robot_type", "HalfCheetah")
    return get_robot_env(robot_type, config)

__all__ = [
    'BaseEnv', 
    'DummyEnv',
    'MuJoCoWrapper',
    'StandardRobotEnv',
    'get_standard_robot_env',
    'get_robot_env',
    'CentipedeEnv',
    'CentipedeRobotFactory'
]