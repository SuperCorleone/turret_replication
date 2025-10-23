import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np

from .base_transfer import BaseTransferModule


class LateralConnectionManager(BaseTransferModule):
    """
    Manages lateral connections between source and target networks
    Implements the knowledge fusion mechanism from the paper
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.connection_type = config.get("connection_type", "weighted_sum")
        self.residual_connection = config.get("residual_connection", True)
        self.layer_specific = config.get("layer_specific", True)
        
        # Connection parameters
        self.connection_strength = config.get("connection_strength", 0.5)
        self.learnable_connections = config.get("learnable_connections", False)
        
        if self.learnable_connections:
            self.connection_weights = nn.ParameterDict()
    
    def compute_transfer_weights(self, 
                               target_state: torch.Tensor,
                               source_states: List[torch.Tensor]) -> List[float]:
        """
        Compute transfer weights - 实现抽象方法
        
        Args:
            target_state: Target state tensor
            source_states: List of source state tensors
            
        Returns:
            List of transfer weights
        """
        # 简化实现：均匀权重
        if not source_states:
            return []
        
        # 返回均匀分布的权重
        num_sources = len(source_states)
        return [1.0 / num_sources] * num_sources
    
    def fuse_knowledge(self,
                      target_features: torch.Tensor,
                      source_features: List[torch.Tensor],
                      weights: List[float]) -> torch.Tensor:
        """
        Fuse knowledge from multiple sources using lateral connections
        
        Args:
            target_features: Features from target network
            source_features: List of features from source networks
            weights: Transfer weights for each source
            
        Returns:
            Fused features
        """
        if not source_features:
            return target_features
        
        batch_size = target_features.shape[0]
        
        if self.connection_type == "weighted_sum":
            fused_features = self._weighted_sum_fusion(
                target_features, source_features, weights, batch_size
            )
        elif self.connection_type == "concatenate":
            fused_features = self._concatenate_fusion(
                target_features, source_features, weights, batch_size
            )
        elif self.connection_type == "attention":
            fused_features = self._attention_fusion(
                target_features, source_features, weights, batch_size
            )
        else:
            raise ValueError(f"Unknown connection type: {self.connection_type}")
        
        # Apply residual connection
        if self.residual_connection:
            fused_features = target_features + self.connection_strength * fused_features
        
        return fused_features
    
    def _weighted_sum_fusion(self,
                           target_features: torch.Tensor,
                           source_features: List[torch.Tensor],
                           weights: List[float],
                           batch_size: int) -> torch.Tensor:
        """Weighted sum fusion method"""
        # Initialize with zeros
        fused = torch.zeros_like(target_features)
        
        # Apply weighted sum
        for i, (source_feat, weight) in enumerate(zip(source_features, weights)):
            # Ensure source features have same shape as target
            if source_feat.shape != target_features.shape:
                source_feat = self._adapt_features(source_feat, target_features.shape)
            
            fused += weight * source_feat
        
        return fused
    
    def _concatenate_fusion(self,
                          target_features: torch.Tensor,
                          source_features: List[torch.Tensor],
                          weights: List[float],
                          batch_size: int) -> torch.Tensor:
        """Concatenation fusion method"""
        # Adapt and weight source features
        weighted_sources = []
        for i, (source_feat, weight) in enumerate(zip(source_features, weights)):
            if source_feat.shape != target_features.shape:
                source_feat = self._adapt_features(source_feat, target_features.shape)
            weighted_sources.append(weight * source_feat)
        
        # Concatenate along feature dimension
        all_features = [target_features] + weighted_sources
        fused = torch.cat(all_features, dim=-1)
        
        # Project back to original dimension if needed
        if fused.shape[-1] != target_features.shape[-1]:
            fused = nn.Linear(fused.shape[-1], target_features.shape[-1]).to(
                fused.device
            )(fused)
        
        return fused
    
    def _attention_fusion(self,
                        target_features: torch.Tensor,
                        source_features: List[torch.Tensor],
                        weights: List[float],
                        batch_size: int) -> torch.Tensor:
        """Attention-based fusion method"""
        # Use weights as attention scores
        attention_weights = torch.softmax(torch.tensor(weights), dim=0)
        
        fused = torch.zeros_like(target_features)
        for i, (source_feat, attn_weight) in enumerate(zip(source_features, attention_weights)):
            if source_feat.shape != target_features.shape:
                source_feat = self._adapt_features(source_feat, target_features.shape)
            fused += attn_weight * source_feat
        
        return fused
    
    def _adapt_features(self, 
                       features: torch.Tensor, 
                       target_shape: torch.Size) -> torch.Tensor:
        """Adapt feature dimensions to match target"""
        if features.dim() != len(target_shape):
            raise ValueError("Feature dimensions don't match")
        
        # Simple adaptation: use linear projection if feature dimensions differ
        if features.shape[-1] != target_shape[-1]:
            adapter = nn.Linear(features.shape[-1], target_shape[-1]).to(features.device)
            features = adapter(features)
        
        return features
    
    def create_layer_connections(self,
                               layer_name: str,
                               target_dim: int,
                               source_dims: List[int]) -> None:
        """Create learnable connections for a specific layer"""
        if self.learnable_connections:
            # Create parameter for each source
            for i, source_dim in enumerate(source_dims):
                param_name = f"{layer_name}_source_{i}"
                self.connection_weights[param_name] = nn.Parameter(
                    torch.randn(target_dim, source_dim) * 0.01
                )
    
    def get_connection_strength(self, 
                              layer_name: str = None,
                              source_idx: int = None) -> float:
        """Get connection strength for specific layer/source"""
        base_strength = self.connection_strength
        
        if self.learnable_connections and layer_name and source_idx is not None:
            param_name = f"{layer_name}_source_{source_idx}"
            if param_name in self.connection_weights:
                # Use Frobenius norm of weight matrix as strength indicator
                weight_norm = torch.norm(self.connection_weights[param_name])
                base_strength *= weight_norm.item()
        
        return base_strength
    
    def update_transfer_metrics(self, 
                              performance_metrics: Dict[str, float]) -> None:
        """
        Update transfer metrics based on performance
        """
        # 这里可以更新基于性能的连接强度等
        pass