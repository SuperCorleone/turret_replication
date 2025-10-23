import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for Set Transformer"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(output)


class SetAttentionBlock(nn.Module):
    """Set Attention Block from Set Transformer"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class MultiHeadSetTransformer(nn.Module):
    """
    Multi-head Set Transformer for learning global state representations
    Implements the readout model from the TURRET paper
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.input_dim = config.get("input_dim", 128)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.output_dim = config.get("output_dim", 128)
        self.n_heads = config.get("n_heads", 8)
        self.n_blocks = config.get("n_blocks", 2)
        self.n_induction_points = config.get("n_induction_points", 4)
        self.dropout = config.get("dropout", 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Encoder: stack of Set Attention Blocks
        self.encoder_blocks = nn.ModuleList([
            SetAttentionBlock(self.hidden_dim, self.n_heads, self.dropout)
            for _ in range(self.n_blocks)
        ])
        
        # Induction points (learnable queries for decoder)
        self.induction_points = nn.Parameter(
            torch.randn(self.n_induction_points, self.hidden_dim)
        )
        
        # Decoder: cross-attention between induction points and encoded features
        self.decoder_attention = MultiHeadAttention(self.hidden_dim, self.n_heads, self.dropout)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
    
    def encoder(self, H: torch.Tensor) -> torch.Tensor:
        """Encoder part of Set Transformer"""
        # Input projection
        x = self.input_projection(H)
        x = self.layer_norm(x)
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        return x
    
    def decoder(self, encoded_H: torch.Tensor) -> torch.Tensor:
        """Decoder part of Set Transformer"""
        batch_size = encoded_H.size(0)
        
        # Expand induction points to batch size
        queries = self.induction_points.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Cross-attention: induction points attend to encoded features
        decoded = self.decoder_attention(queries, encoded_H, encoded_H)
        
        return decoded
    
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Set Transformer
        
        Args:
            H: Node representation matrix [batch_size, num_nodes, input_dim]
            
        Returns:
            State embedding [batch_size, output_dim]
        """
        batch_size = H.size(0)
        
        # Encoder
        encoded_H = self.encoder(H)
        
        # Decoder  
        decoded = self.decoder(encoded_H)
        
        # Pool across induction points (multi-head pooling)
        # Average pooling as in the paper formula
        state_embedding = decoded.mean(dim=1)
        
        # Final projection
        state_embedding = self.output_projection(state_embedding)
        
        return state_embedding
    
    def multi_head_pooling(self, decoded: torch.Tensor) -> torch.Tensor:
        """Multi-head pooling as described in the paper"""
        # Simple average pooling (can be extended to more sophisticated methods)
        return decoded.mean(dim=1)