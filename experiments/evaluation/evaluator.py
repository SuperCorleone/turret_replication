import torch
import numpy as np
from typing import Dict, Any, List, Optional
import os
import json
import glob

from ..runners import SizeTransferRunner, MorphologyTransferRunner


class TransferEvaluator:
    """
    Evaluator for transfer learning experiments
    """
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = results_dir
        self.experiment_results = {}
    
    def load_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load results for a specific experiment"""
        results_path = os.path.join(self.results_dir, f"{experiment_id}_results.json")
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            self.experiment_results[experiment_id] = results
            return results
        else:
            print(f"Results not found for experiment: {experiment_id}")
            return None
    
    def load_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Load all experiment results from results directory"""
        pattern = os.path.join(self.results_dir, "*_results.json")
        result_files = glob.glob(pattern)
        
        for file_path in result_files:
            experiment_id = os.path.basename(file_path).replace("_results.json", "")
            self.load_experiment_results(experiment_id)
        
        return self.experiment_results
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments"""
        comparison = {}
        
        for exp_id in experiment_ids:
            if exp_id in self.experiment_results:
                results = self.experiment_results[exp_id]
                final_perf = results.get('final_performance', {})
                
                comparison[exp_id] = {
                    'mean_reward': final_perf.get('mean_reward', 0),
                    'std_reward': final_perf.get('std_reward', 0),
                    'max_reward': final_perf.get('max_reward', 0),
                    'total_episodes': final_perf.get('total_episodes', 0),
                    'final_independence_p': final_perf.get('final_independence_p', 0),
                }
        
        return comparison
    
    def analyze_transfer_patterns(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze transfer learning patterns from experiment results"""
        if experiment_id not in self.experiment_results:
            print(f"Experiment {experiment_id} not found")
            return {}
        
        results = self.experiment_results[experiment_id]
        training_curves = results.get('training_curves', [])
        transfer_stats = results.get('transfer_statistics', [])
        
        if not training_curves:
            return {}
        
        # Extract key metrics over time
        episodes = [curve['episode'] for curve in training_curves]
        rewards = [curve['reward'] for curve in training_curves]
        independence_factors = [curve.get('independence_p', 0) for curve in training_curves]
        
        # Calculate learning trends
        if len(rewards) > 10:
            recent_rewards = rewards[-10:]
            early_rewards = rewards[:10]
            
            improvement = np.mean(recent_rewards) - np.mean(early_rewards)
            learning_slope = self._calculate_slope(episodes, rewards)
        else:
            improvement = 0
            learning_slope = 0
        
        # Analyze transfer weight patterns
        transfer_patterns = self._analyze_transfer_weights(transfer_stats)
        
        analysis = {
            'total_improvement': float(improvement),
            'learning_slope': float(learning_slope),
            'final_independence': float(independence_factors[-1] if independence_factors else 0),
            'stable_performance': self._check_stability(rewards),
            'transfer_patterns': transfer_patterns
        }
        
        return analysis
    
    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate slope of learning curve"""
        if len(x) < 2:
            return 0.0
        
        x_array = np.array(x)
        y_array = np.array(y)
        
        # Simple linear regression
        slope = np.polyfit(x_array, y_array, 1)[0]
        return float(slope)
    
    def _analyze_transfer_weights(self, transfer_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in transfer weights"""
        if not transfer_stats:
            return {}
        
        # Extract weight statistics
        weight_means = []
        weight_stds = []
        
        for stats in transfer_stats:
            if 'transfer_weights_mean' in stats:
                weight_means.append(stats['transfer_weights_mean'])
            if 'transfer_weights_std' in stats:
                weight_stds.append(stats['transfer_weights_std'])
        
        if not weight_means:
            return {}
        
        # Calculate weight patterns
        final_weights = weight_means[-1] if weight_means else []
        weight_variability = np.mean(weight_stds) if weight_stds else 0
        
        patterns = {
            'final_weight_distribution': final_weights,
            'average_weight_variability': float(weight_variability),
            'dominant_source': int(np.argmax(final_weights)) if final_weights else 0,
            'weight_entropy': float(self._calculate_entropy(final_weights))
        }
        
        return patterns
    
    def _calculate_entropy(self, weights: List[float]) -> float:
        """Calculate entropy of weight distribution"""
        if not weights:
            return 0.0
        
        # Normalize weights
        weights_array = np.array(weights)
        weights_array = weights_array / np.sum(weights_array)
        
        # Calculate entropy
        entropy = -np.sum(weights_array * np.log(weights_array + 1e-8))
        return entropy
    
    def _check_stability(self, rewards: List[float], window: int = 50) -> bool:
        """Check if performance has stabilized"""
        if len(rewards) < window * 2:
            return False
        
        recent_mean = np.mean(rewards[-window:])
        previous_mean = np.mean(rewards[-window*2:-window])
        
        # Consider stable if change is less than 10%
        return abs(recent_mean - previous_mean) / (previous_mean + 1e-8) < 0.1
    
    def generate_report(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            'experiments_compared': experiment_ids,
            'performance_comparison': self.compare_experiments(experiment_ids),
            'detailed_analyses': {},
            'summary_metrics': {}
        }
        
        # Add detailed analysis for each experiment
        for exp_id in experiment_ids:
            report['detailed_analyses'][exp_id] = self.analyze_transfer_patterns(exp_id)
        
        # Calculate summary metrics
        comparison = report['performance_comparison']
        if comparison:
            best_exp = max(comparison.items(), key=lambda x: x[1]['mean_reward'])
            worst_exp = min(comparison.items(), key=lambda x: x[1]['mean_reward'])
            
            report['summary_metrics'] = {
                'best_performing': best_exp[0],
                'best_reward': best_exp[1]['mean_reward'],
                'worst_performing': worst_exp[0],
                'worst_reward': worst_exp[1]['mean_reward'],
                'average_reward': np.mean([exp['mean_reward'] for exp in comparison.values()]),
                'total_experiments': len(comparison)
            }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = "evaluation_report.json") -> None:
        """Save evaluation report to file"""
        report_path = os.path.join(self.results_dir, filename)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {report_path}")