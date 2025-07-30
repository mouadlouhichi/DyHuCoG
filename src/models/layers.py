"""Neural network layers for recommendation models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import defaultdict  # ADD THIS IMPORT


class NGCFLayer(nn.Module):
    """Single NGCF propagation layer
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        dropout: Dropout rate
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Weight matrices
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        
    def forward(self, emb: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        """Forward propagation
        
        Args:
            emb: Node embeddings
            adj: Normalized adjacency matrix
            
        Returns:
            Updated embeddings
        """
        # Message passing
        neighbor_emb = torch.sparse.mm(adj, emb)
        
        # Feature transformation
        emb_i = self.W1(emb)
        emb_j = self.W2(neighbor_emb)
        
        # Aggregate
        output = self.activation(emb_i + emb_j)
        output = self.dropout(output)
        
        return output


class AttentionLayer(nn.Module):
    """Attention mechanism for dynamic weighting
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        dropout: Dropout rate
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention weights
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights
        """
        return self.attention(x)


class BilinearDecoder(nn.Module):
    """Bilinear decoder for rating prediction
    
    Args:
        input_dim: Input dimension
        num_classes: Number of rating classes (for explicit feedback)
    """
    
    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__()
        
        self.num_classes = num_classes
        
        if num_classes > 1:
            # For explicit ratings (multi-class)
            self.W = nn.Parameter(torch.randn(num_classes, input_dim, input_dim))
            self.b = nn.Parameter(torch.zeros(num_classes))
            nn.init.xavier_uniform_(self.W)
        else:
            # For implicit feedback (binary)
            self.W = nn.Parameter(torch.randn(input_dim, input_dim))
            self.b = nn.Parameter(torch.zeros(1))
            nn.init.xavier_uniform_(self.W.unsqueeze(0))
            
    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Decode user-item embeddings to ratings
        
        Args:
            user_emb: User embeddings
            item_emb: Item embeddings
            
        Returns:
            Predicted ratings/scores
        """
        if self.num_classes > 1:
            # Multi-class rating prediction
            scores = []
            for i in range(self.num_classes):
                score = torch.sum(user_emb * torch.matmul(item_emb, self.W[i]), dim=1)
                scores.append(score + self.b[i])
            return torch.stack(scores, dim=1)
        else:
            # Binary prediction
            score = torch.sum(user_emb * torch.matmul(item_emb, self.W), dim=1)
            return score + self.b


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for advanced message passing
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        dropout: Dropout rate
        alpha: LeakyReLU negative slope
        concat: Whether to concatenate or average
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Weight matrix
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention parameters
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        """Forward propagation
        
        Args:
            h: Node features
            adj: Adjacency matrix
            
        Returns:
            Updated node features
        """
        # Linear transformation
        Wh = torch.mm(h, self.W)
        N = Wh.size()[0]
        
        # Attention mechanism
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """Prepare input for attention mechanism"""
        N = Wh.size()[0]
        
        # Repeat for all pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # Concatenate
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        return all_combinations_matrix.view(N, N, 2 * self.out_features)