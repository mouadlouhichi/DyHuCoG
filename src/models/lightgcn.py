"""LightGCN implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LightGCN(nn.Module):
    """LightGCN: Simplifying and Powering Graph Convolution Network
    
    Reference:
        He et al. "LightGCN: Simplifying and Powering Graph Convolution Network 
        for Recommendation." SIGIR 2020.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        latent_dim: Dimension of embeddings
        n_layers: Number of GCN layers
        dropout: Dropout rate (not used in original LightGCN)
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int, 
                 n_layers: int, dropout: float = 0.0):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, adj: torch.sparse.Tensor) -> tuple:
        """Forward propagation
        
        Args:
            adj: Normalized adjacency matrix
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Concatenate user and item embeddings
        emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [emb]
        
        # Multi-layer propagation
        for layer in range(self.n_layers):
            emb = torch.sparse.mm(adj, emb)
            embs.append(emb)
        
        # Layer combination - average pooling
        emb = torch.stack(embs, dim=1).mean(dim=1)
        
        # Split back to users and items
        user_emb = emb[:self.n_users]
        item_emb = emb[self.n_users:self.n_users + self.n_items]
        
        return user_emb, item_emb
    
    def predict(self, users: torch.Tensor, items: torch.Tensor, 
                adj: torch.sparse.Tensor) -> torch.Tensor:
        """Predict ratings for user-item pairs
        
        Args:
            users: User IDs (1-indexed)
            items: Item IDs (1-indexed)
            adj: Normalized adjacency matrix
            
        Returns:
            Predicted scores
        """
        user_emb, item_emb = self.forward(adj)
        
        # Convert to 0-indexed
        users = users - 1
        items = items - 1
        
        # Get embeddings for specific users and items
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        
        # Inner product
        scores = (u_emb * i_emb).sum(dim=1)
        
        return scores
    
    def get_all_predictions(self, users: torch.Tensor, 
                           adj: torch.sparse.Tensor) -> torch.Tensor:
        """Get predictions for all items for given users
        
        Args:
            users: User IDs (1-indexed)
            adj: Normalized adjacency matrix
            
        Returns:
            Prediction matrix [n_users, n_items]
        """
        user_emb, item_emb = self.forward(adj)
        
        # Convert to 0-indexed
        users = users - 1
        
        # Get user embeddings
        u_emb = user_emb[users]
        
        # Compute scores for all items
        scores = torch.matmul(u_emb, item_emb.t())
        
        return scores
    
    def get_embedding(self, adj: torch.sparse.Tensor) -> tuple:
        """Get final embeddings (for visualization or analysis)
        
        Args:
            adj: Normalized adjacency matrix
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        return self.forward(adj)