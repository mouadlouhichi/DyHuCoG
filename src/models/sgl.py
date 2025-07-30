"""
SGL: Self-supervised Graph Learning for Recommendation
Reference: Wu et al., "Self-supervised Graph Learning for Recommendation", SIGIR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SGL(nn.Module):
    """Self-supervised Graph Learning for Recommendation
    
    Args:
        n_users: Number of users
        n_items: Number of items
        latent_dim: Dimension of embeddings
        n_layers: Number of GCN layers
        dropout: Dropout rate for edge dropout augmentation
        ssl_temp: Temperature for contrastive loss
        ssl_reg: Weight for SSL loss
        aug_type: Augmentation type ('ed', 'nd', 'rw')
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int,
                 n_layers: int, dropout: float = 0.1,
                 ssl_temp: float = 0.2, ssl_reg: float = 0.1,
                 aug_type: str = 'ed'):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.aug_type = aug_type
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def graph_augmentation(self, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        """Apply graph augmentation for SSL
        
        Args:
            adj: Original adjacency matrix
            
        Returns:
            Augmented adjacency matrix
        """
        if self.aug_type == 'ed':  # Edge dropout
            # Get indices and values
            indices = adj._indices()
            values = adj._values()
            
            # Random dropout
            dropout_mask = torch.rand(values.size()) > self.dropout
            dropout_mask = dropout_mask.to(adj.device)
            
            new_values = values * dropout_mask.float()
            
            # Create new sparse tensor
            aug_adj = torch.sparse_coo_tensor(
                indices, new_values, adj.shape
            ).coalesce()
            
            return aug_adj
            
        elif self.aug_type == 'nd':  # Node dropout
            n_nodes = adj.shape[0]
            node_mask = torch.rand(n_nodes) > self.dropout
            node_mask = node_mask.to(adj.device)
            
            # Apply node mask to adjacency
            indices = adj._indices()
            values = adj._values()
            
            # Keep edges where both nodes are not dropped
            edge_mask = node_mask[indices[0]] & node_mask[indices[1]]
            new_values = values * edge_mask.float()
            
            aug_adj = torch.sparse_coo_tensor(
                indices, new_values, adj.shape
            ).coalesce()
            
            return aug_adj
            
        elif self.aug_type == 'rw':  # Random walk
            # Simplified random walk augmentation
            return adj  # Placeholder - implement if needed
            
        else:
            return adj
    
    def forward(self, adj: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation
        
        Args:
            adj: Normalized adjacency matrix
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Concatenate embeddings
        emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [emb]
        
        # Multi-layer propagation
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(adj, emb)
            embs.append(emb)
        
        # Layer combination
        emb = torch.stack(embs, dim=1).mean(dim=1)
        
        # Split
        user_emb = emb[:self.n_users]
        item_emb = emb[self.n_users:self.n_users + self.n_items]
        
        return user_emb, item_emb
    
    def ssl_loss(self, users: torch.Tensor, items: torch.Tensor,
                 adj: torch.sparse.Tensor) -> torch.Tensor:
        """Calculate SSL loss using InfoNCE
        
        Args:
            users: User IDs
            items: Item IDs
            adj: Original adjacency matrix
            
        Returns:
            SSL loss value
        """
        # Generate two augmented views
        adj1 = self.graph_augmentation(adj)
        adj2 = self.graph_augmentation(adj)
        
        # Get embeddings from both views
        user_emb1, item_emb1 = self.forward(adj1)
        user_emb2, item_emb2 = self.forward(adj2)
        
        # Get embeddings for batch
        users_idx = users - 1
        items_idx = items - 1
        
        u_emb1 = user_emb1[users_idx]
        u_emb2 = user_emb2[users_idx]
        i_emb1 = item_emb1[items_idx]
        i_emb2 = item_emb2[items_idx]
        
        # Normalize embeddings
        u_emb1 = F.normalize(u_emb1, dim=1)
        u_emb2 = F.normalize(u_emb2, dim=1)
        i_emb1 = F.normalize(i_emb1, dim=1)
        i_emb2 = F.normalize(i_emb2, dim=1)
        
        # User SSL loss
        pos_score_u = (u_emb1 * u_emb2).sum(dim=1) / self.ssl_temp
        ttl_score_u = torch.matmul(u_emb1, user_emb2.T) / self.ssl_temp
        ssl_loss_u = -torch.log(torch.exp(pos_score_u) / torch.exp(ttl_score_u).sum(dim=1)).mean()
        
        # Item SSL loss
        pos_score_i = (i_emb1 * i_emb2).sum(dim=1) / self.ssl_temp
        ttl_score_i = torch.matmul(i_emb1, item_emb2.T) / self.ssl_temp
        ssl_loss_i = -torch.log(torch.exp(pos_score_i) / torch.exp(ttl_score_i).sum(dim=1)).mean()
        
        return self.ssl_reg * (ssl_loss_u + ssl_loss_i)
    
    def predict(self, users: torch.Tensor, items: torch.Tensor,
                adj: torch.sparse.Tensor) -> torch.Tensor:
        """Predict ratings for user-item pairs"""
        user_emb, item_emb = self.forward(adj)
        
        users = users - 1
        items = items - 1
        
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        
        return (u_emb * i_emb).sum(dim=1)
    
    def get_all_predictions(self, users: torch.Tensor,
                           adj: torch.sparse.Tensor) -> torch.Tensor:
        """Get predictions for all items for given users"""
        user_emb, item_emb = self.forward(adj)
        users = users - 1
        u_emb = user_emb[users]
        
        return torch.matmul(u_emb, item_emb.t())