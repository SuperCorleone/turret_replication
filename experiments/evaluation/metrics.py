import numpy as np
from typing import Dict, Any, List, Tuple
import scipy.stats as stats


def calculate_metrics(training_curves: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate comprehensive performance metrics from training curves"""
    if not training_curves:
        return {}
    
    # Extract rewards and episodes
    episodes = [curve['episode'] for curve in training_curves]
    rewards = [curve['reward'] for curve in training_curves]
    
    # Basic statistics
    metrics = {
        'final_reward': float(rewards[-1]),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'max_reward': float(np.max(rewards)),
        'min_reward': float(np.min(rewards)),
        'total_episodes': len(episodes)
    }
    
    # Learning efficiency metrics
    if len(rewards) > 10:
        # Time to threshold (first episode where reward exceeds mean)
        mean_reward = np.mean(rewards)
        threshold_episodes = [i for i, r in enumerate(rewards) if r > mean_reward]
        metrics['time_to_threshold'] = float(threshold_episodes[0]) if threshold_episodes else len(episodes)
        
        # Learning speed (slope of learning curve)
        if len(episodes) > 1:
            slope, _ = np.polyfit(episodes, rewards, 1)
            metrics['learning_speed'] = float(slope)
        
        # Performance stability (coefficient of variation of last 20% episodes)
        stable_start = int(0.8 * len(rewards))
        stable_rewards = rewards[stable_start:]
        if len(stable_rewards) > 1:
            cv = np.std(stable_rewards) / np.mean(stable_rewards)
            metrics['performance_stability'] = float(cv)
    
    return metrics


def compare_performance(experiment_results: Dict[str, Dict[str, Any]], 
                       baseline_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare experiment performance against baselines"""
    comparison = {}
    
    for exp_id, results in experiment_results.items():
        exp_metrics = calculate_metrics(results.get('training_curves', []))
        final_perf = results.get('final_performance', {})
        
        comparison[exp_id] = {
            'experiment_metrics': exp_metrics,
            'improvement_over_baseline': {},
            'statistical_significance': {}
        }
        
        # Compare with baselines
        for baseline_id, baseline_data in baseline_results.items():
            baseline_metrics = calculate_metrics(baseline_data.get('training_curves', []))
            
            # Calculate improvements
            improvement = {}
            for metric in ['mean_reward', 'max_reward', 'learning_speed']:
                if metric in exp_metrics and metric in baseline_metrics:
                    exp_val = exp_metrics[metric]
                    base_val = baseline_metrics[metric]
                    if base_val != 0:
                        improvement[metric] = (exp_val - base_val) / abs(base_val)
                    else:
                        improvement[metric] = exp_val - base_val
            
            comparison[exp_id]['improvement_over_baseline'][baseline_id] = improvement
            
            # Statistical significance testing (simplified)
            exp_rewards = [c['reward'] for c in results.get('training_curves', [])]
            base_rewards = [c['reward'] for c in baseline_data.get('training_curves', [])]
            
            if len(exp_rewards) > 10 and len(base_rewards) > 10:
                try:
                    # Use Mann-Whitney U test for non-parametric comparison
                    _, p_value = stats.mannwhitneyu(exp_rewards, base_rewards, alternative='two-sided')
                    comparison[exp_id]['statistical_significance'][baseline_id] = {
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
                except:
                    comparison[exp_id]['statistical_significance'][baseline_id] = {
                        'p_value': 1.0,
                        'significant': False
                    }
    
    return comparison


def calculate_sample_efficiency(training_curves: List[Dict[str, Any]], 
                              target_reward: float) -> Dict[str, float]:
    """Calculate sample efficiency metrics"""
    if not training_curves:
        return {}
    
    rewards = [curve['reward'] for curve in training_curves]
    episodes = [curve['episode'] for curve in training_curves]
    
    # Find first episode where target reward is achieved
    target_episode = None
    for i, reward in enumerate(rewards):
        if reward >= target_reward:
            target_episode = episodes[i]
            break
    
    efficiency_metrics = {
        'target_reward': target_reward,
        'episodes_to_target': target_episode if target_episode is not None else len(episodes),
        'achieved_target': target_episode is not None
    }
    
    # Calculate area under learning curve (AUC) as measure of overall efficiency
    if len(episodes) > 1:
        auc = np.trapz(rewards, episodes)
        efficiency_metrics['learning_auc'] = float(auc)
    
    return efficiency_metrics


def analyze_transfer_effectiveness(transfer_stats: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze effectiveness of transfer learning"""
    if not transfer_stats:
        return {}
    
    # Extract transfer-related metrics
    independence_factors = []
    transfer_strengths = []
    weight_entropies = []
    
    for stats in transfer_stats:
        if 'independence_p_mean' in stats:
            independence_factors.append(stats['independence_p_mean'])
        if 'transfer_strength' in stats:
            transfer_strengths.append(stats['transfer_strength'])
        if 'weight_entropy' in stats:
            weight_entropies.append(stats['weight_entropy'])
    
    effectiveness_metrics = {}
    
    if independence_factors:
        effectiveness_metrics.update({
            'final_independence': independence_factors[-1],
            'avg_independence': np.mean(independence_factors),
            'independence_trend': independence_factors[-1] - independence_factors[0] if len(independence_factors) > 1 else 0
        })
    
    if transfer_strengths:
        effectiveness_metrics['avg_transfer_strength'] = np.mean(transfer_strengths)
    
    if weight_entropies:
        effectiveness_metrics.update({
            'final_weight_entropy': weight_entropies[-1],
            'avg_weight_entropy': np.mean(weight_entropies),
            'knowledge_diversity': 1.0 / (1.0 + np.mean(weight_entropies))  # Inverse of entropy
        })
    
    return effectiveness_metrics