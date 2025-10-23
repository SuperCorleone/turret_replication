import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import gymnasium as gym

# 修复导入
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from environments.base_env import BaseEnv
from models.morphology import MorphologyGraph, Node, Edge, StandardRobotMorphology

class CentipedeEnv(BaseEnv):
    """Centipede-n机器人环境"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_torsos = config.get("num_torsos", 4)  # n/2个躯干
        self.num_legs = self.num_torsos * 2  # n条腿
        self.robot_type = f"Centipede-{self.num_legs}"
        
        # 创建形态学图
        self.morphology_graph = self._create_centipede_morphology()
        
        # 初始化Gym环境
        self._init_gym_environment()
        
        # 当前状态
        self._current_observation = None
        
    def _create_centipede_morphology(self) -> MorphologyGraph:
        """创建Centipede形态学图"""
        graph = MorphologyGraph(self.robot_type)
        
        # 添加节点
        nodes = []
        
        # 根节点
        nodes.append(Node("root", "root", observation_dim=13, action_dim=0))
        
        # 躯干节点
        for i in range(self.num_torsos):
            nodes.append(Node(f"torso_{i}", "torso", observation_dim=8, action_dim=0))
        
        # 髋关节节点
        for i in range(self.num_legs):
            nodes.append(Node(f"hip_{i}", "hip", observation_dim=6, action_dim=1))
        
        # 踝关节节点  
        for i in range(self.num_legs):
            nodes.append(Node(f"ankle_{i}", "ankle", observation_dim=6, action_dim=1))
        
        for node in nodes:
            graph.add_node(node)
        
        # 添加边（连接关系）
        # 根节点连接到第一个躯干
        graph.add_edge("root", "torso_0")
        
        # 躯干之间的连接（链式结构）
        for i in range(self.num_torsos - 1):
            graph.add_edge(f"torso_{i}", f"torso_{i+1}")
        
        # 躯干连接到髋关节
        for i in range(self.num_torsos):
            # 每个躯干连接两个髋关节
            left_hip_idx = i * 2
            right_hip_idx = i * 2 + 1
            graph.add_edge(f"torso_{i}", f"hip_{left_hip_idx}")
            graph.add_edge(f"torso_{i}", f"hip_{right_hip_idx}")
        
        # 髋关节连接到踝关节
        for i in range(self.num_legs):
            graph.add_edge(f"hip_{i}", f"ankle_{i}")
        
        return graph
    
    def _init_gym_environment(self):
        """初始化Gym环境"""
        # 计算总观察维度和动作维度
        total_obs_dim = sum(node.observation_dim for node in self.morphology_graph.nodes)
        total_action_dim = sum(node.action_dim for node in self.morphology_graph.nodes)
        
        # 创建自定义Gym环境
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(total_action_dim,), dtype=np.float32
        )
        
        self._is_initialized = True
    
    @property
    def observation_space(self) -> gym.Space:
        """获取观察空间"""
        return self._observation_space
    
    @property
    def action_space(self) -> gym.Space:
        """获取动作空间"""
        return self._action_space
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 生成随机初始状态
        self._current_observation = self._generate_initial_observation()
        info = {
            "morphology_graph": self.morphology_graph,
            "robot_type": self.robot_type,
            "num_nodes": len(self.morphology_graph.nodes)
        }
        
        return self._current_observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """环境步进"""
        # 简化物理模拟：基于动作计算新状态和奖励
        self._current_observation = self._simulate_physics(action)
        reward = self._compute_reward(self._current_observation, action)
        terminated = self._check_termination(self._current_observation)
        truncated = False
        
        info = {
            "action_magnitude": np.linalg.norm(action),
            "forward_progress": self._compute_forward_progress(self._current_observation)
        }
        
        return self._current_observation, reward, terminated, truncated, info
    
    def close(self) -> None:
        """关闭环境"""
        self._is_initialized = False
        self._current_observation = None
    
    def _generate_initial_observation(self) -> np.ndarray:
        """生成初始观察"""
        observations = []
        
        for node in self.morphology_graph.nodes:
            # 为每个节点生成随机初始状态
            node_obs = np.random.randn(node.observation_dim).astype(np.float32) * 0.1
            observations.append(node_obs)
        
        return np.concatenate(observations)
    
    def _simulate_physics(self, action: np.ndarray) -> np.ndarray:
        """简化物理模拟"""
        if self._current_observation is None:
            return self._generate_initial_observation()
            
        noise = np.random.randn(*self._current_observation.shape) * 0.01
        action_effect = action * 0.1  # 简化动作效果
        
        # 确保动作维度匹配
        if len(action_effect) < len(self._current_observation):
            action_effect = np.pad(action_effect, (0, len(self._current_observation) - len(action_effect)))
        
        new_obs = self._current_observation + action_effect + noise
        return new_obs.astype(np.float32)
    
    def _compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        """计算奖励"""
        # 前进奖励
        forward_reward = observation[0] * 10.0  # 假设第一个维度是前进方向
        
        # 存活奖励
        survival_reward = 0.1
        
        # 动作惩罚（能量效率）
        action_penalty = -0.01 * np.sum(np.square(action))
        
        # 平衡奖励（简化）
        balance_reward = -0.1 * np.sum(np.square(observation[1:3]))  # 假设第二、三维度是平衡相关
        
        total_reward = forward_reward + survival_reward + action_penalty + balance_reward
        return float(total_reward)
    
    def _check_termination(self, observation: np.ndarray) -> bool:
        """检查终止条件"""
        # 检查是否摔倒（某个关节角度过大）
        if np.max(np.abs(observation)) > 5.0:
            return True
        
        # 检查是否前进足够远
        if observation[0] > 100.0:  # 简化条件
            return True
        
        return False
    
    def _compute_forward_progress(self, observation: np.ndarray) -> float:
        """计算前进进度"""
        return float(observation[0])
    
    def get_morphology_graph(self) -> MorphologyGraph:
        """获取形态学图"""
        return self.morphology_graph
    
    def get_node_observations(self, global_observation: np.ndarray) -> Dict[str, np.ndarray]:
        """将全局观察分解为节点观察"""
        node_observations = {}
        start_idx = 0
        
        for node in self.morphology_graph.nodes:
            end_idx = start_idx + node.observation_dim
            node_observations[node.node_id] = global_observation[start_idx:end_idx]
            start_idx = end_idx
        
        return node_observations

class CentipedeRobotFactory:
    """Centipede机器人工厂"""
    
    @staticmethod
    def create_centipede_env(num_legs: int, config: Dict[str, Any] = None) -> CentipedeEnv:
        """创建Centipede环境"""
        if config is None:
            config = {}
        
        config["num_torsos"] = num_legs // 2
        return CentipedeEnv(config)
    
    @staticmethod
    def get_available_sizes() -> List[int]:
        """获取可用的Centipede大小"""
        return [4, 6, 8, 12, 16, 20]  # 论文中使用的尺寸