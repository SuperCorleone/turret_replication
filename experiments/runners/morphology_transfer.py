import torch
import numpy as np
from typing import Dict, Any, List

from .size_transfer import SizeTransferRunner
from experiments.pretrain_source import load_source_policies


class MorphologyTransferRunner(SizeTransferRunner):
    """
    Runner for morphology transfer experiments (different robot types)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Morphology transfer specific configuration
        self.transfer_type = config.get("transfer_type", "quad_to_biped")
        self._setup_transfer_scenario()
    
    def _setup_transfer_scenario(self) -> None:
        """Setup morphology transfer scenario based on type"""
        scenarios = {
            "quad_to_biped": {
                "sources": ["HalfCheetah", "Ant"],
                "target": "Walker2d",
                "description": "Quadrupedal to Bipedal transfer"
            },
            "biped_to_humanoid": {
                "sources": ["Walker2d", "Hopper"], 
                "target": "Humanoid",
                "description": "Bipedal to Humanoid transfer"
            },
            "similar_morphology": {
                "sources": ["Ant", "HalfCheetah"],
                "target": "Humanoid", 
                "description": "Similar morphology transfer"
            }
        }
        
        if self.transfer_type in scenarios:
            scenario = scenarios[self.transfer_type]
            self.source_robots = scenario["sources"]
            self.target_robot = scenario["target"]
            self.experiment_id = f"morphology_{self.transfer_type}"
        else:
            # Use default from config
            self.experiment_id = f"morphology_{self.target_robot}"
    
    def setup_experiment(self) -> None:
        """Setup morphology transfer experiment"""
        self.logger.info(
            f"Setting up morphology transfer: {self.source_robots} -> {self.target_robot}"
        )
        self.logger.info(f"Transfer type: {self.transfer_type}")
        
        # Call parent setup
        super().setup_experiment()
        
        # Morphology-specific adjustments
        self.trainer.independence_scheduler.total_steps = self.total_episodes * 150
        self.logger.info("Morphology transfer experiment setup completed")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate morphology transfer performance with additional metrics"""
        basic_metrics = super().evaluate()
        
        # Add morphology-specific metrics
        morphology_metrics = {
            'transfer_type': self.transfer_type,
            'morphology_complexity': self._calculate_morphology_complexity(),
            'source_target_similarity': self._calculate_similarity_metric()
        }
        
        return {**basic_metrics, **morphology_metrics}
    
    def _calculate_morphology_complexity(self) -> float:
        """Calculate morphology complexity metric"""
        # Simplified complexity estimation based on robot type
        complexity_scores = {
            "Hopper": 1.0,
            "Walker2d": 2.0, 
            "HalfCheetah": 2.5,
            "Ant": 3.0,
            "Humanoid": 4.0
        }
        
        target_complexity = complexity_scores.get(self.target_robot, 2.0)
        source_complexity = np.mean([complexity_scores.get(r, 2.0) for r in self.source_robots])
        
        return float(target_complexity - source_complexity)
    
    def _calculate_similarity_metric(self) -> float:
        """Calculate similarity between source and target morphologies"""
        # Simplified similarity based on leg count and body type
        leg_counts = {
            "Hopper": 1,
            "Walker2d": 2,
            "HalfCheetah": 4, 
            "Ant": 4,
            "Humanoid": 2
        }
        
        target_legs = leg_counts.get(self.target_robot, 2)
        source_legs = np.mean([leg_counts.get(r, 2) for r in self.source_robots])
        
        # Similarity based on leg count difference
        leg_similarity = 1.0 / (1.0 + abs(target_legs - source_legs))
        
        return float(leg_similarity)