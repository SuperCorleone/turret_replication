import torch
import torch.nn as nn
import time
import os
import sys
import numpy as np
from typing import Dict, Any

# 使用现有工具路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.file_utils import ensure_dir


class PerformanceOptimizer:
    """性能优化器：监控并优化 GNN 前向传播性能"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_history = []

    def optimize_gnn_forward(
        self,
        model: nn.Module,
        node_observations: Dict[str, torch.Tensor],
        morphology_graph: Any
    ) -> Dict[str, Any]:
        """优化 GNN 前向传播过程"""
        # 1. 节点观察批处理优化
        optimized_observations = self._batch_node_observations(node_observations)

        # 2. 邻接矩阵预计算
        adjacency_matrix = self._precompute_adjacency(morphology_graph)

        # 3. 启用自动混合精度 (AMP)
        use_amp = self.config.get('use_amp', False) and torch.cuda.is_available()

        with torch.no_grad():
            start_time = time.time()

            if use_amp:
                with torch.cuda.amp.autocast():
                    mean, std = model(optimized_observations, morphology_graph)
            else:
                mean, std = model(optimized_observations, morphology_graph)

            end_time = time.time()

        # 4. 记录性能统计
        stats = {
            'forward_time': end_time - start_time,
            'memory_usage': self._get_memory_usage(),
            'gpu_available': torch.cuda.is_available(),
            'num_nodes': len(node_observations),
            'used_amp': use_amp
        }

        self.optimization_history.append(stats)
        return stats

    def _batch_node_observations(self, node_observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """批处理节点观测向量，自动对齐长度"""
        node_ids = sorted(node_observations.keys())
        max_obs_dim = max(obs.shape[0] for obs in node_observations.values())

        batched_observations = []
        for node_id in node_ids:
            obs = node_observations[node_id]
            if obs.shape[0] < max_obs_dim:
                # 填充到相同维度
                padded_obs = torch.cat([obs, torch.zeros(max_obs_dim - obs.shape[0])])
            else:
                padded_obs = obs
            batched_observations.append(padded_obs)

        return torch.stack(batched_observations)

    def _precompute_adjacency(self, morphology_graph: Any) -> torch.Tensor:
        """预计算邻接矩阵"""
        num_nodes = len(morphology_graph.nodes)
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        node_to_idx = {node.node_id: idx for idx, node in enumerate(morphology_graph.nodes)}

        for edge in morphology_graph.edges:
            source_idx = node_to_idx.get(edge.source.node_id)
            target_idx = node_to_idx.get(edge.target.node_id)
            if source_idx is not None and target_idx is not None:
                adj_matrix[target_idx, source_idx] = 1.0

        return adj_matrix

    def _get_memory_usage(self) -> float:
        """获取当前进程内存使用量（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0

    def save_optimization_report(self, report_dir: str = "data/optimization_reports"):
        """保存优化报告为 JSON 文件"""
        ensure_dir(report_dir)

        report = {
            'optimization_history': self.optimization_history,
            'summary': self._generate_optimization_summary(),
            'config': self.config
        }

        import json
        report_file = os.path.join(report_dir, 'performance_optimization_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Optimization report saved to: {report_file}")

    def _generate_optimization_summary(self) -> Dict[str, Any]:
        """生成优化摘要"""
        if not self.optimization_history:
            return {}

        forward_times = [stats['forward_time'] for stats in self.optimization_history]
        memory_usages = [stats['memory_usage'] for stats in self.optimization_history]

        return {
            'average_forward_time': np.mean(forward_times),
            'std_forward_time': np.std(forward_times),
            'average_memory_usage': np.mean(memory_usages),
            'total_optimization_steps': len(self.optimization_history)
        }
