from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
import json
import torch
import numpy as np

# 添加导入
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from configs.base_config import TURRETConfig
from utils import setup_logging, save_checkpoint, load_checkpoint

class ExperimentRunner(ABC):
    """实验运行器基类 - 更新为使用 TURRETConfig"""
    
    def __init__(self, config: TURRETConfig):  # 明确类型注解
        # 统一配置处理
        if hasattr(config, '__dict__'):
            self.config = config.__dict__
        else:
            self.config = config
            
        self.logger = setup_logging()
        
        # 实验跟踪
        self.experiment_id = config.experiment_id if hasattr(config, 'experiment_id') else config.get("experiment_id", "default_experiment")
        self.results_dir = config.results_dir if hasattr(config, 'results_dir') else config.get("results_dir", "data/results")
        self.checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        
        # 创建目录
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 结果存储
        self.results = {
            'training_curves': [],
            'transfer_statistics': [],
            'final_performance': {},
            'config': self.config
        }
    
    @abstractmethod
    def setup_experiment(self) -> None:
        """Setup experiment environment and models"""
        pass
    
    @abstractmethod
    def run_experiment(self) -> Dict[str, Any]:
        """Run the main experiment"""
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate experiment results"""
        pass
    
    def save_results(self) -> None:
        """Save experiment results to file"""
        results_path = os.path.join(
            self.results_dir, 
            f"{self.experiment_id}_results.json"
        )
        
        # Convert any tensors to lists for JSON serialization
        serializable_results = self._make_results_serializable(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
    
    def _make_results_serializable(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to JSON-serializable format"""
        serializable = {}
        
        for key, value in results.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = self._make_results_serializable(value)
            elif isinstance(value, (list, tuple)):
                serializable[key] = [
                    item.tolist() if isinstance(item, (np.ndarray, torch.Tensor)) else item
                    for item in value
                ]
            else:
                serializable[key] = value
        
        return serializable
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load experiment checkpoint"""
        return load_checkpoint(checkpoint_path)
    
    def save_checkpoint(self, state: Dict[str, Any], filename: str) -> None:
        """Save experiment checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        save_checkpoint(state, checkpoint_path)
    
    def log_progress(self, episode: int, metrics: Dict[str, float]) -> None:
        """Log training progress"""
        log_message = f"Episode {episode}: "
        log_message += ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
        self.logger.info(log_message)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        return {
            'experiment_id': self.experiment_id,
            'config': self.config,
            'results_dir': self.results_dir,
            'total_episodes': len(self.results['training_curves'])
        }