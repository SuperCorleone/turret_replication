import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
import os


class TrajectoryPlotter:
    """
    Plot training trajectories and performance curves
    """
    
    def __init__(self):
        self.style_config = {
            'figure.figsize': (12, 8),
            'font.size': 12,
            'lines.linewidth': 2,
            'axes.grid': True,
            'grid.alpha': 0.3
        }
        
        # Color scheme for different experiments
        self.color_scheme = {
            'TURRET': 'red',
            'PPO': 'blue',
            'CAT': 'green',
            'NerveNet': 'orange',
            'Snowflake': 'purple'
        }
    
    def plot_training_curves(self,
                           experiments: Dict[str, Dict[str, Any]],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training curves for multiple experiments
        
        Args:
            experiments: Dictionary of experiment results
            save_path: Optional path to save plot
            
        Returns:
            matplotlib Figure
        """
        plt.style.use('default')
        for key, value in self.style_config.items():
            plt.rcParams[key] = value
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for exp_name, results in experiments.items():
            training_curves = results.get('training_curves', [])
            if not training_curves:
                continue
            
            # Extract data
            episodes = [curve['episode'] for curve in training_curves]
            rewards = [curve['reward'] for curve in training_curves]
            policy_losses = [curve.get('policy_loss', 0) for curve in training_curves]
            value_losses = [curve.get('value_loss', 0) for curve in training_curves]
            independence_factors = [curve.get('independence_p', 0) for curve in training_curves]
            
            color = self.color_scheme.get(exp_name, 'gray')
            
            # Plot rewards
            axes[0].plot(episodes, rewards, label=exp_name, color=color, alpha=0.8)
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
            axes[0].set_title('Training Rewards')
            axes[0].legend()
            
            # Plot losses
            axes[1].plot(episodes, policy_losses, label=f'{exp_name} Policy', 
                        color=color, alpha=0.8, linestyle='-')
            axes[1].plot(episodes, value_losses, label=f'{exp_name} Value',
                        color=color, alpha=0.8, linestyle='--')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Losses')
            axes[1].legend()
            
            # Plot independence factors (for TURRET)
            if any(independence_factors):
                axes[2].plot(episodes, independence_factors, label=exp_name,
                           color=color, alpha=0.8)
                axes[2].set_xlabel('Episode')
                axes[2].set_ylabel('Independence Factor (p)')
                axes[2].set_title('Gradual Independence')
                axes[2].legend()
            
            # Plot smoothed rewards (moving average)
            if len(rewards) > 10:
                window = min(50, len(rewards) // 10)
                smoothed_rewards = self._moving_average(rewards, window)
                axes[3].plot(episodes[window-1:], smoothed_rewards, 
                           label=exp_name, color=color, alpha=0.8)
                axes[3].set_xlabel('Episode')
                axes[3].set_ylabel('Smoothed Reward')
                axes[3].set_title('Smoothed Training Rewards')
                axes[3].legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        return fig
    
    def plot_transfer_statistics(self,
                               experiments: Dict[str, Dict[str, Any]],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot transfer learning statistics
        
        Args:
            experiments: Dictionary of experiment results
            save_path: Optional path to save plot
        """
        plt.style.use('default')
        for key, value in self.style_config.items():
            plt.rcParams[key] = value
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for exp_name, results in experiments.items():
            transfer_stats = results.get('transfer_statistics', [])
            if not transfer_stats:
                continue
            
            color = self.color_scheme.get(exp_name, 'gray')
            episodes = list(range(len(transfer_stats)))
            
            # Extract transfer metrics
            transfer_weights = []
            semantic_distances = []
            independence_factors = []
            
            for stats in transfer_stats:
                # Average transfer weights
                if 'transfer_weights_mean' in stats:
                    avg_weight = np.mean(stats['transfer_weights_mean'])
                    transfer_weights.append(avg_weight)
                
                if 'semantic_distance_mean' in stats:
                    semantic_distances.append(stats['semantic_distance_mean'])
                
                if 'independence_p_mean' in stats:
                    independence_factors.append(stats['independence_p_mean'])
            
            # Plot transfer weights
            if transfer_weights:
                axes[0].plot(episodes, transfer_weights, label=exp_name,
                           color=color, alpha=0.8)
                axes[0].set_xlabel('Training Step')
                axes[0].set_ylabel('Average Transfer Weight')
                axes[0].set_title('Transfer Weight Evolution')
                axes[0].legend()
            
            # Plot semantic distances
            if semantic_distances:
                axes[1].plot(episodes, semantic_distances, label=exp_name,
                           color=color, alpha=0.8)
                axes[1].set_xlabel('Training Step')
                axes[1].set_ylabel('Semantic Distance')
                axes[1].set_title('Semantic Distance Evolution')
                axes[1].legend()
            
            # Plot independence factors
            if independence_factors:
                axes[2].plot(episodes, independence_factors, label=exp_name,
                           color=color, alpha=0.8)
                axes[2].set_xlabel('Training Step')
                axes[2].set_ylabel('Independence Factor')
                axes[2].set_title('Independence Schedule')
                axes[2].legend()
            
            # Plot weight entropy (if available)
            weight_entropies = [stats.get('weight_entropy', 0) for stats in transfer_stats]
            if any(weight_entropies):
                axes[3].plot(episodes, weight_entropies, label=exp_name,
                           color=color, alpha=0.8)
                axes[3].set_xlabel('Training Step')
                axes[3].set_ylabel('Weight Entropy')
                axes[3].set_title('Knowledge Diversity')
                axes[3].legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Transfer statistics saved to {save_path}")
        
        return fig
    
    def plot_performance_comparison(self,
                                  comparison_results: Dict[str, Any],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot performance comparison between methods
        
        Args:
            comparison_results: Results from TransferEvaluator.compare_experiments()
            save_path: Optional path to save plot
        """
        plt.style.use('default')
        for key, value in self.style_config.items():
            plt.rcParams[key] = value
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        methods = list(comparison_results.keys())
        mean_rewards = [comp['mean_reward'] for comp in comparison_results.values()]
        max_rewards = [comp['max_reward'] for comp in comparison_results.values()]
        std_rewards = [comp['std_reward'] for comp in comparison_results.values()]
        
        colors = [self.color_scheme.get(method, 'gray') for method in methods]
        
        # Plot mean rewards
        bars1 = axes[0].bar(methods, mean_rewards, color=colors, alpha=0.7)
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Average Performance')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mean_rewards):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.1f}', ha='center', va='bottom')
        
        # Plot max rewards
        bars2 = axes[1].bar(methods, max_rewards, color=colors, alpha=0.7)
        axes[1].set_ylabel('Max Reward')
        axes[1].set_title('Best Performance')
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, max_rewards):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.1f}', ha='center', va='bottom')
        
        # Plot standard deviations
        bars3 = axes[2].bar(methods, std_rewards, color=colors, alpha=0.7)
        axes[2].set_ylabel('Standard Deviation')
        axes[2].set_title('Performance Stability')
        axes[2].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, std_rewards):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison saved to {save_path}")
        
        return fig
    
    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """Calculate moving average"""
        return np.convolve(data, np.ones(window)/window, mode='valid')