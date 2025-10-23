import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.evaluation.metrics import calculate_metrics, analyze_transfer_effectiveness


class TestEvaluationMetrics:
    """Test cases for evaluation metrics"""
    
    def test_calculate_metrics(self):
        """Test metric calculation"""
        # Create dummy training curves
        training_curves = [
            {'episode': 0, 'reward': 10.0, 'policy_loss': 1.0},
            {'episode': 1, 'reward': 20.0, 'policy_loss': 0.8},
            {'episode': 2, 'reward': 30.0, 'policy_loss': 0.6},
            {'episode': 3, 'reward': 40.0, 'policy_loss': 0.4},
            {'episode': 4, 'reward': 50.0, 'policy_loss': 0.2},
        ]
        
        metrics = calculate_metrics(training_curves)
        
        assert 'final_reward' in metrics
        assert 'mean_reward' in metrics
        assert 'max_reward' in metrics
        assert metrics['final_reward'] == 50.0
        assert metrics['mean_reward'] == 30.0
        assert metrics['max_reward'] == 50.0
    
    def test_analyze_transfer_effectiveness(self):
        """Test transfer effectiveness analysis"""
        # Create dummy transfer statistics
        transfer_stats = [
            {
                'independence_p_mean': 0.1,
                'transfer_strength': 0.9,
                'weight_entropy': 1.2
            },
            {
                'independence_p_mean': 0.5, 
                'transfer_strength': 0.5,
                'weight_entropy': 0.8
            },
            {
                'independence_p_mean': 0.9,
                'transfer_strength': 0.1,
                'weight_entropy': 0.4
            }
        ]
        
        effectiveness = analyze_transfer_effectiveness(transfer_stats)
        
        assert 'final_independence' in effectiveness
        assert 'avg_transfer_strength' in effectiveness
        assert 'knowledge_diversity' in effectiveness
        assert effectiveness['final_independence'] == 0.9
        assert effectiveness['avg_transfer_strength'] == 0.5