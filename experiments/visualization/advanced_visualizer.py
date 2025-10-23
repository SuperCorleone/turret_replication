import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

class AdvancedVisualizer:
    """高级可视化工具"""
    
    def __init__(self, results_dir: str = "data/visualizations"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # 颜色配置
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.transfer_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 迁移相关颜色
    
    def plot_transfer_dynamics(self, 
                             experiment_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> plt.Figure:
        """绘制迁移学习动态"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        training_curves = experiment_results.get('training_curves', [])
        transfer_stats = experiment_results.get('transfer_statistics', [])
        
        if not training_curves:
            return fig
        
        # 1. 学习曲线
        episodes = [curve['episode'] for curve in training_curves]
        rewards = [curve['reward'] for curve in training_curves]
        
        axes[0].plot(episodes, rewards, 'b-', alpha=0.7, linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Performance')
        axes[0].grid(True, alpha=0.3)
        
        # 计算移动平均
        if len(rewards) > 10:
            window = min(50, len(rewards) // 4)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, 
                        label=f'Moving Avg (window={window})')
            axes[0].legend()
        
        # 2. 独立性因子p
        if transfer_stats:
            p_values = [stats.get('independence_p_mean', 0) for stats in transfer_stats]
            episodes_p = list(range(len(p_values)))
            
            axes[1].plot(episodes_p, p_values, 'g-', linewidth=2)
            axes[1].set_xlabel('Training Step')
            axes[1].set_ylabel('Independence Factor (p)')
            axes[1].set_title('Gradual Independence Schedule')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
        
        # 3. 迁移权重
        if transfer_stats and 'transfer_weights_mean' in transfer_stats[0]:
            weights_data = [stats['transfer_weights_mean'] for stats in transfer_stats]
            weights_array = np.array(weights_data)
            
            for i in range(weights_array.shape[1]):
                axes[2].plot(episodes_p[:len(weights_array)], weights_array[:, i], 
                           label=f'Source {i+1}', linewidth=2)
            
            axes[2].set_xlabel('Training Step')
            axes[2].set_ylabel('Transfer Weight')
            axes[2].set_title('Source Policy Weights Over Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # 4. 语义距离
        if transfer_stats and 'semantic_distance_mean' in transfer_stats[0]:
            distances = [stats.get('semantic_distance_mean', 0) for stats in transfer_stats]
            axes[3].plot(episodes_p, distances, 'purple', linewidth=2)
            axes[3].set_xlabel('Training Step')
            axes[3].set_ylabel('Semantic Distance')
            axes[3].set_title('Average Semantic Distance to Sources')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Transfer dynamics plot saved to {save_path}")
        
        return fig
    
    def create_interactive_embedding_plot(self,
                                        embeddings: Dict[str, np.ndarray],
                                        save_path: Optional[str] = None) -> go.Figure:
        """创建交互式嵌入可视化"""
        # 合并所有嵌入
        all_embeddings = []
        all_labels = []
        all_sizes = []
        
        for task_name, task_embeddings in embeddings.items():
            all_embeddings.extend(task_embeddings)
            all_labels.extend([task_name] * len(task_embeddings))
            all_sizes.extend([8] * len(task_embeddings))  # 统一大小
        
        all_embeddings = np.array(all_embeddings)
        
        # 降维到3D
        if all_embeddings.shape[1] > 3:
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(all_embeddings)
        else:
            embeddings_3d = all_embeddings
        
        # 创建交互式图表
        fig = go.Figure()
        
        unique_labels = list(set(all_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in all_labels]
            color = f'rgb({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)})'
            
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1],
                z=embeddings_3d[mask, 2],
                mode='markers',
                name=label,
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7
                ),
                hovertemplate=f'<b>{label}</b><br>' +
                            'X: %{x:.2f}<br>' +
                            'Y: %{y:.2f}<br>' +
                            'Z: %{z:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Interactive State Embedding Space",
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
        
        return fig
    
    def plot_attention_heatmap(self,
                             attention_weights: np.ndarray,
                             node_names: List[str],
                             save_path: Optional[str] = None) -> plt.Figure:
        """绘制注意力热力图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 平均注意力权重
        mean_attention = attention_weights.mean(axis=0)
        
        # 1. 热力图
        im = axes[0].imshow(mean_attention, cmap='viridis', aspect='auto')
        axes[0].set_xticks(range(len(node_names)))
        axes[0].set_yticks(range(len(node_names)))
        axes[0].set_xticklabels(node_names, rotation=45)
        axes[0].set_yticklabels(node_names)
        axes[0].set_title('Average Attention Weights')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[0])
        
        # 2. 节点重要性条形图
        node_importance = mean_attention.sum(axis=0)  # 列求和，表示节点作为源的重要性
        axes[1].bar(range(len(node_names)), node_importance, color='skyblue')
        axes[1].set_xticks(range(len(node_names)))
        axes[1].set_xticklabels(node_names, rotation=45)
        axes[1].set_title('Node Importance (Sum of Attention Weights)')
        axes[1].set_ylabel('Importance Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to {save_path}")
        
        return fig
    
    def create_comparison_dashboard(self,
                                  experiment_results: Dict[str, Dict[str, Any]],
                                  save_path: Optional[str] = None) -> go.Figure:
        """创建实验对比仪表板"""
        methods = list(experiment_results.keys())
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Final Performance', 'Learning Speed', 
                          'Sample Efficiency', 'Transfer Effectiveness'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. 最终性能
        final_performances = []
        for method in methods:
            results = experiment_results[method]
            if 'final_performance' in results:
                final_performances.append(results['final_performance'].get('mean_reward', 0))
            else:
                final_performances.append(0)
        
        fig.add_trace(
            go.Bar(x=methods, y=final_performances, name='Final Reward'),
            row=1, col=1
        )
        
        # 2. 学习曲线
        for method in methods:
            results = experiment_results[method]
            training_curves = results.get('training_curves', [])
            if training_curves:
                episodes = [curve['episode'] for curve in training_curves]
                rewards = [curve['reward'] for curve in training_curves]
                fig.add_trace(
                    go.Scatter(x=episodes, y=rewards, name=method, mode='lines'),
                    row=1, col=2
                )
        
        # 3. 样本效率（达到阈值所需步数）
        sample_efficiencies = []
        threshold = 100  # 示例阈值
        for method in methods:
            results = experiment_results[method]
            training_curves = results.get('training_curves', [])
            efficiency = len(training_curves)  # 简化计算
            sample_efficiencies.append(efficiency)
        
        fig.add_trace(
            go.Bar(x=methods, y=sample_efficiencies, name='Sample Efficiency'),
            row=2, col=1
        )
        
        # 4. 迁移效果
        transfer_effects = []
        for method in methods:
            results = experiment_results[method]
            # 简化计算迁移效果
            effect = results.get('final_performance', {}).get('mean_reward', 0) / 100
            transfer_effects.append(effect)
        
        fig.add_trace(
            go.Bar(x=methods, y=transfer_effects, name='Transfer Effect'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Experiment Comparison Dashboard",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Comparison dashboard saved to {save_path}")
        
        return fig