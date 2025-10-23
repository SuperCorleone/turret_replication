from abc import ABC, abstractmethod
from typing import Dict, Any
import torch

class BaseTransferModule(ABC):
    """Base class for transfer learning components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cpu")
    
    def to_device(self, tensor: Any) -> Any:
        """Move tensor to appropriate device"""
        if hasattr(tensor, 'to'):
            return tensor.to(self.device)
        return tensor
    
    @abstractmethod
    def compute_transfer_weights(self, target_state, source_states):
        pass
    
    @abstractmethod  
    def fuse_knowledge(self, target_features, source_features, weights):
        pass