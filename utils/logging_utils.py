# utils/logging_utils.py - 修复 TrainingLogger

import logging
import os
import numpy as np
from typing import Optional, Dict, Any
import datetime
import json  # 添加导入

def setup_logging(log_dir: str = "logs", 
                 level: int = logging.INFO,
                 console: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger("turret")
    logger.setLevel(level)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"turret_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get logger with specific name"""
    return logging.getLogger(f"turret.{name}")

class TrainingLogger:
    """Logger for training statistics"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = {}
        
        # 创建CSV文件用于训练数据
        self.csv_file = os.path.join(log_dir, "training_data.csv")
        with open(self.csv_file, 'w') as f:
            f.write("episode,reward,length,total_steps,timestamp\n")
    
    def log_episode(self, 
                   episode: int,
                   reward: float,
                   length: int,
                   total_steps: int) -> None:
        """Log episode statistics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # 写入CSV
        timestamp = datetime.datetime.now().isoformat()
        with open(self.csv_file, 'a') as f:
            f.write(f"{episode},{reward},{length},{total_steps},{timestamp}\n")
    
    def log_losses(self, losses: Dict[str, float], step: int) -> None:
        """Log training losses"""
        for key, value in losses.items():
            if key not in self.losses:
                self.losses[key] = []
            self.losses[key].append((step, value))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
            'mean_length': float(np.mean(self.episode_lengths)),
            'total_episodes': len(self.episode_rewards),
        }
    
    def save_statistics(self, filepath: str) -> None:
        """保存统计信息到文件"""
        stats = self.get_statistics()
        if stats:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)