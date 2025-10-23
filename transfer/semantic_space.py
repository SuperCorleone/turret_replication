import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np

from .base_transfer import BaseTransferModule


class SemanticSpaceManager(BaseTransferModule):
    """
    Manages the unified semantic space for state representations
    Projects states from different tasks into a common embedding space
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.embedding_dim = config.get("embedding_dim", 128)
        self.normalize_embeddings = config.get("normalize_embeddings", True)
        self.distance_metric = config.get("distance_metric", "euclidean")
        
        # Projection networks for different task types
        self.projection_networks = nn.ModuleDict()
        
        # Store input dimensions for different tasks
        self.task_input_dims = {}
        
        # Initialize with default projection (will be created on first use)
        self.default_projection = None
        self.default_input_dim = None
    
    def _create_projection_network(self, input_dim: int) -> nn.Module:
        """Create a projection network for given input dimension"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
    
    def add_task_projection(self, task_id: str, input_dim: int) -> None:
        """Add task-specific projection network"""
        projection_net = self._create_projection_network(input_dim)
        self.projection_networks[task_id] = projection_net
        self.task_input_dims[task_id] = input_dim
    
    def _get_projection_network(self, task_id: str, input_dim: int) -> nn.Module:
        """Get or create projection network for task"""
        if task_id in self.projection_networks:
            # Check if input dimension matches
            if self.task_input_dims[task_id] != input_dim:
                raise ValueError(
                    f"Input dimension mismatch for task {task_id}: "
                    f"expected {self.task_input_dims[task_id]}, got {input_dim}"
                )
            return self.projection_networks[task_id]
        else:
            # Create default projection if not exists
            if self.default_projection is None:
                self.default_projection = self._create_projection_network(input_dim)
                self.default_input_dim = input_dim
                return self.default_projection
            else:
                # Check if input dimension matches default
                if self.default_input_dim != input_dim:
                    raise ValueError(
                        f"Input dimension mismatch for default projection: "
                        f"expected {self.default_input_dim}, got {input_dim}"
                    )
                return self.default_projection
    
    def project_state(self, 
                     state: torch.Tensor, 
                     task_id: str = "default") -> torch.Tensor:
        """
        Project state into the unified semantic space
        
        Args:
            state: Input state tensor
            task_id: Identifier for the task/source
            
        Returns:
            Projected state embedding in semantic space
        """
        # Get input dimension
        if state.dim() == 1:
            input_dim = state.shape[0]
            state = state.unsqueeze(0)  # Add batch dimension
        else:
            input_dim = state.shape[-1]
        
        # Get appropriate projection network
        projection_net = self._get_projection_network(task_id, input_dim)
        
        # Project to semantic space
        embedding = projection_net(state)
        
        # Normalize if required
        if self.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding.squeeze(0) if state.dim() == 2 and state.shape[0] == 1 else embedding
    
    def compute_semantic_distance(self, 
                                embedding1: torch.Tensor,
                                embedding2: torch.Tensor) -> float:
        """
        Compute distance between two state embeddings in semantic space
        
        Args:
            embedding1: First state embedding
            embedding2: Second state embedding
            
        Returns:
            Distance between embeddings
        """
        # Ensure both embeddings have same dimensions
        if embedding1.dim() == 1:
            embedding1 = embedding1.unsqueeze(0)
        if embedding2.dim() == 1:
            embedding2 = embedding2.unsqueeze(0)
        
        if self.distance_metric == "euclidean":
            distance = torch.norm(embedding1 - embedding2, p=2, dim=-1).mean()
        elif self.distance_metric == "cosine":
            distance = 1 - torch.nn.functional.cosine_similarity(
                embedding1, embedding2, dim=-1
            ).mean()
        elif self.distance_metric == "manhattan":
            distance = torch.norm(embedding1 - embedding2, p=1, dim=-1).mean()
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distance.item()
    
    def compute_similarity_matrix(self,
                                target_embeddings: List[torch.Tensor],
                                source_embeddings: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Compute similarity matrix between target and source state embeddings
        
        Args:
            target_embeddings: List of target state embeddings
            source_embeddings: List of lists, each containing source state embeddings
            
        Returns:
            Similarity matrix of shape [num_target_states, num_sources]
        """
        num_target = len(target_embeddings)
        num_sources = len(source_embeddings)
        
        similarity_matrix = torch.zeros(num_target, num_sources)
        
        for i, target_emb in enumerate(target_embeddings):
            for j, source_embs in enumerate(source_embeddings):
                if source_embs:
                    # Compute average distance to all source embeddings
                    distances = []
                    for source_emb in source_embs:
                        dist = self.compute_semantic_distance(target_emb, source_emb)
                        distances.append(dist)
                    
                    # Convert distance to similarity (inverse)
                    avg_distance = np.mean(distances)
                    similarity = 1.0 / (1.0 + avg_distance)
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def get_common_representation_space(self) -> Dict[str, Any]:
        """Get information about the common representation space"""
        return {
            "embedding_dim": self.embedding_dim,
            "distance_metric": self.distance_metric,
            "normalize_embeddings": self.normalize_embeddings,
            "num_task_projections": len(self.projection_networks),
            "default_input_dim": self.default_input_dim
        }
    
    # Implement abstract methods from BaseTransferModule
    def compute_transfer_weights(self, 
                               target_state: torch.Tensor,
                               source_states: List[torch.Tensor]) -> List[float]:
        """
        Compute transfer weights based on semantic distances
        
        Note: This is a simplified implementation. For full functionality,
        use AdaptiveWeightCalculator which builds upon SemanticSpaceManager.
        """
        if not source_states:
            return []
        
        # Project states to semantic space
        target_embedding = self.project_state(target_state, "target")
        
        source_embeddings = []
        for source_state in source_states:
            source_embedding = self.project_state(source_state, "source")
            source_embeddings.append(source_embedding)
        
        # Compute distances and convert to similarities
        similarities = []
        for source_embedding in source_embeddings:
            distance = self.compute_semantic_distance(target_embedding, source_embedding)
            similarity = 1.0 / (1.0 + distance)
            similarities.append(similarity)
        
        # Apply softmax to get probabilities
        similarities_tensor = torch.tensor(similarities)
        weights = torch.softmax(similarities_tensor, dim=0)
        
        return weights.tolist()
    
    def fuse_knowledge(self,
                      target_features: torch.Tensor,
                      source_features: List[torch.Tensor],
                      weights: List[float]) -> torch.Tensor:
        """
        Fuse knowledge from multiple sources using weighted sum
        
        Note: This is a basic implementation. For more sophisticated fusion,
        use LateralConnectionManager.
        """
        if not source_features:
            return target_features
        
        # Simple weighted sum fusion
        fused = torch.zeros_like(target_features)
        
        for source_feat, weight in zip(source_features, weights):
            # Ensure compatible shapes
            if source_feat.shape != target_features.shape:
                # Simple adaptation: use interpolation if dimensions differ
                if source_feat.dim() == target_features.dim():
                    # For 1D features, use linear interpolation
                    if source_feat.dim() == 2:  # [batch, features]
                        source_feat_resized = torch.nn.functional.interpolate(
                            source_feat.unsqueeze(1),
                            size=target_features.shape[1],
                            mode='linear',
                            align_corners=False
                        ).squeeze(1)
                    else:
                        # For other cases, use adaptive avg pool
                        source_feat_resized = torch.nn.functional.adaptive_avg_pool1d(
                            source_feat.unsqueeze(1) if source_feat.dim() == 1 else source_feat,
                            target_features.shape[-1]
                        )
                        if source_feat.dim() == 1:
                            source_feat_resized = source_feat_resized.squeeze(1)
                else:
                    raise ValueError(f"Feature dimension mismatch: {source_feat.shape} vs {target_features.shape}")
            else:
                source_feat_resized = source_feat
            
            fused += weight * source_feat_resized
        
        return fused
    
    def update_transfer_metrics(self, 
                              performance_metrics: Dict[str, float]) -> None:
        """
        Update transfer metrics based on performance
        
        Note: SemanticSpaceManager doesn't track performance by default.
        This method is provided for interface compatibility.
        """
        # SemanticSpaceManager doesn't inherently track performance metrics
        # Subclasses or composed classes can implement this functionality
        pass