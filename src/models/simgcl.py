"""
SimGCL: Simple Contrastive Learning for Graph Collaborative Filtering
Reference: Yu et al., "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation", SIGIR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimGCL(nn.Module):
    """SimGCL: Contrastive learning without explicit augmentation
    
    Args:
        n_users: Number of users
        n_items: Number of items
        latent_dim: Dimension of embeddings
        n_layers: Number of GCN layers
        noise_eps: Noise epsilon for embedding perturbation
        ssl_temp: Temperature for contrastive loss
        ssl_reg: Weight for SSL loss
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int,
                 n_layers: int, noise_eps: float = 0.1,
                 ssl_temp: float = 0.2, ssl_reg: float = 0.1):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.noise_eps = noise_eps
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, adj: torch.sparse.Tensor,
                perturb: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation with optional perturbation
        
        Args:
            adj: Normalized adjacency matrix
            perturb: Whether to add noise perturbation
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Get base embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Add perturbation if needed
        if perturb:
            user_noise = torch.randn_like(user_emb) * self.noise_eps
            item_noise = torch.randn_like(item_emb) * self.noise_eps
            user_emb = user_emb + user_noise
            item_emb = item_emb + item_noise
        
        # Concatenate
        emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [emb]
        
        # Propagation
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(adj, emb)
            embs.append(emb)
        
        # Aggregate
        emb = torch.stack(embs, dim=1).mean(dim=1)
        
        # Split
        user_emb_final = emb[:self.n_users]
        item_emb_final = emb[self.n_users:self.n_users + self.n_items]
        
        return user_emb_final, item_emb_final
    
    def ssl_loss(self, users: torch.Tensor, items: torch.Tensor,
                 adj: torch.sparse.Tensor) -> torch.Tensor:
        """Calculate SSL loss with embedding perturbation"""
        # Get two perturbed views
        user_emb1, item_emb1 = self.forward(adj, perturb=True)
        user_emb2, item_emb2 = self.forward(adj, perturb=True)
        
        # Get embeddings for batch
        users_idx = users - 1
        items_idx = items - 1
        
        u1 = F.normalize(user_emb1[users_idx], dim=1)
        u2 = F.normalize(user_emb2[users_idx], dim=1)
        i1 = F.normalize(item_emb1[items_idx], dim=1)
        i2 = F.normalize(item_emb2[items_idx], dim=1)
        
        # InfoNCE loss for users
        pos_score_u = (u1 * u2).sum(dim=1) / self.ssl_temp
        ttl_score_u = torch.matmul(u1, user_emb2.T) / self.ssl_temp
        ssl_loss_u = -torch.log(
            torch.exp(pos_score_u) / torch.exp(ttl_score_u).sum(dim=1)
        ).mean()
        
        # InfoNCE loss for items
        pos_score_i = (i1 * i2).sum(dim=1) / self.ssl_temp
        ttl_score_i = torch.matmul(i1, item_emb2.T) / self.ssl_temp
        ssl_loss_i = -torch.log(
            torch.exp(pos_score_i) / torch.exp(ttl_score_i).sum(dim=1)
        ).mean()
        
        return self.ssl_reg * (ssl_loss_u + ssl_loss_i)
    
    def predict(self, users: torch.Tensor, items: torch.Tensor,
                adj: torch.sparse.Tensor) -> torch.Tensor:
        """Predict ratings"""
        user_emb, item_emb = self.forward(adj, perturb=False)
        
        users = users - 1
        items = items - 1
        
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        
        return (u_emb * i_emb).sum(dim=1)
    
    def get_all_predictions(self, users: torch.Tensor,
                           adj: torch.sparse.Tensor) -> torch.Tensor:
        """Get all predictions"""
        user_emb, item_emb = self.forward(adj, perturb=False)
        users = users - 1
        u_emb = user_emb[users]
        
        return torch.matmul(u_emb, item_emb.t())