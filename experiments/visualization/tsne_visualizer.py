import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, Any, List, Optional
import os
import torch


class TSNEVisualizer:
    """
    Visualize state embeddings using t-SNE for semantic space analysis
    """
    
    def __init__(self, n_components: int = 3, perplexity: float = 30.0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42
        )
        
        # Color scheme for different tasks/robots
        self.color_scheme = {
            'HalfCheetah': 'red',
            'Ant': 'blue', 
            'Hopper': 'green',
            'Walker2d': 'orange',
            'Humanoid': 'purple',
            'source': 'lightblue',
            'target': 'darkred'
        }
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit t-SNE and transform embeddings"""
        return self.tsne.fit_transform(embeddings)
    
    def visualize_embeddings(self, 
                           embeddings: Dict[str, np.ndarray],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize embeddings from multiple tasks/sources
        
        Args:
            embeddings: Dictionary mapping task names to embedding arrays
            save_path: Optional path to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        # Combine all embeddings
        all_embeddings = []
        labels = []
        colors = []
        
        for task_name, task_embeddings in embeddings.items():
            n_samples = len(task_embeddings)
            all_embeddings.extend(task_embeddings)
            labels.extend([task_name] * n_samples)
            colors.extend([self.color_scheme.get(task_name, 'gray')] * n_samples)
        
        all_embeddings = np.array(all_embeddings)
        
        # Apply t-SNE
        if self.n_components == 2:
            transformed = self.fit_transform(all_embeddings)
            fig = self._plot_2d(transformed, labels, colors)
        else:
            transformed = self.fit_transform(all_embeddings)
            fig = self._plot_3d(transformed, labels, colors)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def _plot_2d(self, points: np.ndarray, labels: List[str], colors: List[str]) -> plt.Figure:
        """Create 2D scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = list(set(labels))
        
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(points[mask, 0], points[mask, 1], 
                      c=self.color_scheme.get(label, 'gray'),
                      label=label, alpha=0.7, s=30)
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('Semantic Space Visualization (2D)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_3d(self, points: np.ndarray, labels: List[str], colors: List[str]) -> plt.Figure:
        """Create 3D scatter plot"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = list(set(labels))
        
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                      c=self.color_scheme.get(label, 'gray'),
                      label=label, alpha=0.7, s=30)
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2') 
        ax.set_zlabel('t-SNE Component 3')
        ax.set_title('Semantic Space Visualization (3D)')
        ax.legend()
        
        return fig
    
    def visualize_trajectory_similarity(self,
                                      source_trajectories: Dict[str, np.ndarray],
                                      target_trajectories: np.ndarray,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize similarity between source and target trajectories
        
        Args:
            source_trajectories: Dictionary of source task trajectories
            target_trajectories: Target task trajectories
            save_path: Optional path to save visualization
            
        Returns:
            matplotlib Figure
        """
        # Combine all trajectories
        all_trajectories = []
        labels = []
        
        # Add source trajectories
        for source_name, trajectories in source_trajectories.items():
            all_trajectories.extend(trajectories)
            labels.extend([f'source_{source_name}'] * len(trajectories))
        
        # Add target trajectories
        all_trajectories.extend(target_trajectories)
        labels.extend(['target'] * len(target_trajectories))
        
        all_trajectories = np.array(all_trajectories)
        
        # Apply t-SNE
        transformed = self.fit_transform(all_trajectories)
        
        # Create visualization
        if self.n_components == 2:
            fig = self._plot_trajectory_2d(transformed, labels)
        else:
            fig = self._plot_trajectory_3d(transformed, labels)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_trajectory_2d(self, points: np.ndarray, labels: List[str]) -> plt.Figure:
        """Plot trajectory similarity in 2D"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot source trajectories
        source_mask = np.array([l.startswith('source_') for l in labels])
        target_mask = np.array([l == 'target' for l in labels])
        
        ax.scatter(points[source_mask, 0], points[source_mask, 1],
                  c='lightblue', label='Source Tasks', alpha=0.6, s=40)
        ax.scatter(points[target_mask, 0], points[target_mask, 1],
                  c='darkred', label='Target Task', alpha=0.8, s=50)
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('Trajectory Similarity Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def calculate_trajectory_distances(self,
                                    source_trajectories: Dict[str, np.ndarray],
                                    target_trajectories: np.ndarray) -> Dict[str, float]:
        """
        Calculate distances between source and target trajectories
        
        Returns:
            Dictionary of average distances for each source
        """
        distances = {}
        
        for source_name, trajectories in source_trajectories.items():
            source_dists = []
            for source_traj in trajectories:
                for target_traj in target_trajectories:
                    dist = np.linalg.norm(source_traj - target_traj)
                    source_dists.append(dist)
            
            distances[source_name] = np.mean(source_dists)
        
        return distances