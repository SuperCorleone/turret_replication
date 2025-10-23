# transfer/weight_calculator.py

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np

# 修复导入
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transfer.semantic_space import SemanticSpaceManager

class AdaptiveWeightCalculator:
    """
    Computes adaptive transfer weights based on state semantic similarities
    Implements the weight calculation from the paper
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cpu")
        self.semantic_space = SemanticSpaceManager(config)
        self.temperature = config.get("temperature", 1.0)
        self.min_weight = config.get("min_weight", 0.01)
        self.use_performance_feedback = config.get("use_performance_feedback", False)
        
        # Performance tracking
        self.source_performances = {}
        self.performance_decay = config.get("performance_decay", 0.99)
        
        # Weight smoothing
        self.previous_weights = None
        self.smoothing_factor = config.get("smoothing_factor", 0.1)
    
    def to_device(self, tensor: Any) -> Any:
        """Move tensor to appropriate device - 修复缺失的方法"""
        if hasattr(tensor, 'to'):
            return tensor.to(self.device)
        return tensor
    
    def compute_transfer_weights(self, 
                               target_state: torch.Tensor,
                               source_states: List[torch.Tensor],
                               source_ids: List[str] = None) -> List[float]:
        """
        Compute transfer weights based on semantic distances
        
        Args:
            target_state: Current target task state
            source_states: List of corresponding source task states
            source_ids: Optional list of source identifiers
            
        Returns:
            List of weights for each source
        """
        if not source_states:
            return []
        
        # Ensure states have correct dimensions for projection
        target_state = self._ensure_correct_dimensions(target_state)
        source_states = [self._ensure_correct_dimensions(state) for state in source_states]
        
        # Project states to semantic space
        target_embedding = self.semantic_space.project_state(target_state, "target")
        source_embeddings = []
        for i, source_state in enumerate(source_states):
            source_id = source_ids[i] if source_ids else f"source_{i}"
            source_embedding = self.semantic_space.project_state(source_state, source_id)
            source_embeddings.append(source_embedding)
        
        # Compute distances and convert to similarities
        similarities = []
        for source_embedding in source_embeddings:
            distance = self.semantic_space.compute_semantic_distance(
                target_embedding, source_embedding
            )
            similarity = 1.0 / (1.0 + distance)
            similarities.append(similarity)
        
        # Apply temperature scaling
        similarities_tensor = torch.tensor(similarities) / self.temperature
        
        # Apply softmax to get probabilities
        weights = torch.softmax(similarities_tensor, dim=0)
        
        # Apply minimum weight threshold
        weights = torch.clamp(weights, min=self.min_weight)
        
        # Renormalize
        weights = weights / weights.sum()
        
        # Apply performance feedback if enabled
        if self.use_performance_feedback and source_ids:
            weights = self._apply_performance_feedback(weights, source_ids)
        
        # Apply smoothing
        if self.previous_weights is not None and len(weights) == len(self.previous_weights):
            weights = (1 - self.smoothing_factor) * weights + \
            self.smoothing_factor * torch.tensor(self.previous_weights)
        
        self.previous_weights = weights.tolist()
        
        return weights.tolist()
    
    def _ensure_correct_dimensions(self, state: torch.Tensor) -> torch.Tensor:
        """Ensure state has correct dimensions for processing"""
        if state.dim() == 1:
            # Single state vector
            return state
        elif state.dim() == 2 and state.shape[0] == 1:
            # Batch with single element
            return state.squeeze(0)
        else:
            # For batches, we'll process each element separately
            return state
    
    def _apply_performance_feedback(self, 
                                 weights: torch.Tensor, 
                                 source_ids: List[str]) -> torch.Tensor:
        """Adjust weights based on source performance history"""
        adjusted_weights = weights.clone()
        for i, source_id in enumerate(source_ids):
            if source_id in self.source_performances:
                performance = self.source_performances[source_id]
                # Boost weights for better performing sources
                adjustment = 1.0 + performance
                adjusted_weights[i] = adjusted_weights[i] * adjustment
        
        # Renormalize
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        return adjusted_weights
    
    def update_performance(self, 
                         source_id: str, 
                         performance: float) -> None:
        """Update performance metric for a source"""
        if source_id not in self.source_performances:
            self.source_performances[source_id] = performance
        else:
            # Exponential moving average
            self.source_performances[source_id] = (
                self.performance_decay * self.source_performances[source_id] +
                (1 - self.performance_decay) * performance
            )
    
    def compute_batch_weights(self,
                            target_states: torch.Tensor,
                            source_states_list: List[torch.Tensor],
                            source_ids: List[str] = None) -> torch.Tensor:
        """
        Compute transfer weights for a batch of states
        
        Args:
            target_states: Batch of target states [batch_size, state_dim]
            source_states_list: List of source state batches [batch_size, state_dim]
            source_ids: Optional source identifiers
            
        Returns:
            Weight matrix [batch_size, num_sources]
        """
        batch_size = target_states.shape[0]
        num_sources = len(source_states_list)
        weight_matrix = torch.zeros(batch_size, num_sources)
        
        for i in range(batch_size):
            target_state = target_states[i]
            source_states = [source_states[i] for source_states in source_states_list]
            weights = self.compute_transfer_weights(
                target_state, source_states, source_ids
            )
            weight_matrix[i] = torch.tensor(weights)
        
        return weight_matrix
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics for all sources"""
        return self.source_performances.copy()