import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional


class BaseNetwork(nn.Module):
    """Base class for all neural networks in TURRET"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = config.get("device", "cpu")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def save(self, filepath: str) -> None:
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP(BaseNetwork):
    """Multi-Layer Perceptron with configurable architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.hidden_dims = config.get("hidden_dims", [256, 256])
        self.activation = config.get("activation", "relu")
        
        # Build layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self) -> nn.Module:
        """Get activation function"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }
        return activations.get(self.activation, nn.ReLU())
    
    def _initialize_weights(self) -> None:
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)