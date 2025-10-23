import torch
import numpy as np
from typing import Dict, Any, List
import os

# 添加导入路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.base_config import TURRETConfig
from experiments.runners.base_runner import ExperimentRunner
from training.trainers.transfer_trainer import TURRETTrainer

class AblationRunner(ExperimentRunner):
    """消融实验运行器"""
    
    def __init__(self, config: TURRETConfig):
        super().__init__(config)
        self.ablation_type = config.ablation_type if hasattr(config, 'ablation_type') else config.get("ablation_type", "no_attention")
    
    def setup_experiment(self) -> None:
        """设置消融实验"""
        self.logger.info(f"Setting up ablation study: {self.ablation_type}")
        
        # 根据消融类型修改配置
        ablation_config = self._apply_ablation(self.config.copy())
        
        # 设置环境、网络等（简化实现）
        # 这里需要根据你的具体实现来完善
        
    def _apply_ablation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用消融设置"""
        if self.ablation_type == "no_attention":
            config['use_attention'] = False
        elif self.ablation_type == "no_semantic_space":
            config['use_semantic_space'] = False  
        elif self.ablation_type == "fixed_weights":
            config['adaptive_weights'] = False
            
        return config
    
    def run_experiment(self) -> Dict[str, Any]:
        """运行消融实验"""
        # 实现具体的消融实验逻辑
        # 返回与其它实验相同格式的结果
        return self._generate_mock_results()
    
    def _generate_mock_results(self) -> Dict[str, Any]:
        """生成模拟结果（用于测试）"""
        return {
            'final_performance': {
                'mean_reward': np.random.uniform(300, 800),
                'std_reward': np.random.uniform(50, 150),
            },
            'ablation_type': self.ablation_type
        }