"""NGCF implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .layers import NGCFLayer


class NGCF(nn.Module):
    """Neural Graph Collaborative Filtering
    
    Reference:
        Wang et al. "Neural Graph Collaborative Filtering." SIGIR 2019.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        latent_dim: Dimension of embeddings
        n_layers: Number of propagation layers
        dropout: Dropout rate
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int,
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        
        # Propagation layers
        self.propagation_layers = nn.ModuleList()
        for i in range(n_layers):
            self.propagation_layers.append(
                NGCFLayer(latent_dim, latent_dim, dropout)
            )
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, adj: torch.sparse.Tensor) -> tuple:
        """Forward propagation through GCN layers
        
        Args:
            adj: Normalized adjacency matrix
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Concatenate embeddings
        emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [emb]
        
        # Message passing through layers
        for layer in self.propagation_layers:
            emb = layer(emb, adj)
            embs.append(emb)
        
        # Concatenate embeddings from all layers
        emb = torch.cat(embs, dim=1)
        
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
        
        # Get embeddings
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
        """Get final embeddings
        
        Args:
            adj: Normalized adjacency matrix
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        return self.forward(adj)