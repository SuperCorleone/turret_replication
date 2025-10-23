import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from transfer.semantic_space import SemanticSpaceManager


class TestSemanticSpaceManager:
    """Test cases for SemanticSpaceManager"""
    
    @pytest.fixture
    def semantic_config(self):
        return {
            "embedding_dim": 64,
            "normalize_embeddings": True,
            "distance_metric": "euclidean",
            "device": "cpu"
        }
    
    def test_initialization(self, semantic_config):
        """Test semantic space initialization"""
        manager = SemanticSpaceManager(semantic_config)
        assert manager.embedding_dim == 64
        assert manager.normalize_embeddings == True
    
    def test_state_projection(self, semantic_config):
        """Test state projection to semantic space"""
        manager = SemanticSpaceManager(semantic_config)
        
        # Test single state
        state = torch.randn(32)
        embedding = manager.project_state(state)
        
        assert embedding.shape == (64,)
        assert not torch.isnan(embedding).any()
        
        # Test batch of states
        batch_states = torch.randn(8, 32)
        batch_embeddings = manager.project_state(batch_states)
        
        assert batch_embeddings.shape == (8, 64)
    
    def test_semantic_distance(self, semantic_config):
        """Test semantic distance computation"""
        manager = SemanticSpaceManager(semantic_config)
        
        embedding1 = torch.randn(64)
        embedding2 = torch.randn(64)
        
        distance = manager.compute_semantic_distance(embedding1, embedding2)
        
        assert isinstance(distance, float)
        assert distance >= 0
    
    def test_task_specific_projection(self, semantic_config):
        """Test task-specific projection networks"""
        manager = SemanticSpaceManager(semantic_config)
        
        # Add task-specific projection
        manager.add_task_projection("task1", 32)
        
        state = torch.randn(32)
        embedding = manager.project_state(state, "task1")
        
        assert embedding.shape == (64,)
        assert "task1" in manager.projection_networks