"""
DyHuCoG: Dynamic Hybrid Recommender via Graph-based Cooperative Games
Main model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .cooperative_game import CooperativeGameDAE, ShapleyValueNetwork
from ..utils.graph_builder import GraphBuilder


class DyHuCoG(nn.Module):
    """Dynamic Hybrid Recommender via Graph-based Cooperative Games
    
    Args:
        n_users: Number of users
        n_items: Number of items  
        n_genres: Number of context features (genres)
        config: Model configuration dictionary
    """
    
    def __init__(self, n_users: int, n_items: int, n_genres: int, config: Dict):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_genres = n_genres
        self.n_nodes = n_users + n_items + n_genres
        self.config = config
        
        # Extract config parameters
        self.latent_dim = config['latent_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.use_attention = config.get('use_attention', True)
        self.use_genres = config.get('use_genres', True)
        
        # Node embeddings
        self.embedding = nn.Embedding(self.n_nodes, self.latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        
        # Cooperative game components
        self.dae = CooperativeGameDAE(
            n_items=n_items,
            hidden_dim=config['dae_hidden'],
            dropout=self.dropout
        )
        
        self.shapley_net = ShapleyValueNetwork(
            n_items=n_items,
            hidden_dim=config['shapley_hidden'],
            n_samples=config.get('n_shapley_samples', 10)
        )
        
        # Attention mechanism
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim),
                nn.Tanh(),
                nn.Linear(self.latent_dim, 1),
                nn.Sigmoid()
            )
        
        # Edge weights dictionary
        self.edge_weights = {}
        self.adj = None
        
    def compute_shapley_weights(self, train_mat: torch.Tensor) -> Dict[Tuple[int, int], float]:
        """Compute Shapley value-based edge weights
        
        Args:
            train_mat: User-item interaction matrix
            
        Returns:
            Dictionary mapping (user, item) pairs to edge weights
        """
        edge_weights = {}
        
        with torch.no_grad():
            for user_id in range(1, self.n_users + 1):
                user_items = train_mat[user_id]
                
                if user_items.sum() == 0:
                    continue
                
                # Compute Shapley values
                shapley_values = self.shapley_net(user_items.unsqueeze(0)).squeeze()
                
                # Update weights for user-item edges
                item_indices = torch.where(user_items > 0)[0]
                for item_idx in item_indices:
                    item_id = item_idx.item() + 1
                    weight = shapley_values[item_idx].item()
                    
                    # Ensure positive weight
                    weight = max(weight, 0.1) + 1.0
                    edge_weights[(user_id, item_id)] = weight
        
        return edge_weights
    
    def build_hypergraph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                        item_genres: Dict[int, list]) -> None:
        """Build weighted hypergraph with Shapley values
        
        Args:
            edge_index: Initial edge indices
            edge_weight: Initial edge weights
            item_genres: Dictionary mapping items to genres
        """
        # Update edge weights with Shapley values
        updated_weights = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[:, i]
            if u < self.n_users and v >= self.n_users:
                # User-item edge
                user_id = u.item() + 1
                item_id = v.item() - self.n_users + 1
                weight = self.edge_weights.get((user_id, item_id), 1.0)
                updated_weights.append(weight)
            else:
                # Other edges (item-genre) keep original weight
                updated_weights.append(edge_weight[i].item())
        
        edge_weight = torch.tensor(updated_weights, device=edge_index.device)
        
        # Create adjacency matrix
        self.adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, (self.n_nodes, self.n_nodes)
        )
        
        # Normalize adjacency matrix
        self.adj = GraphBuilder.normalize_adj(self.adj)
    
    def forward(self) -> torch.Tensor:
        """Graph neural network forward propagation
        
        Returns:
            Node embeddings after multi-layer propagation
        """
        emb = self.embedding.weight
        embs = [emb]
        
        # Multi-layer propagation
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.adj, emb)
            emb = F.dropout(emb, p=self.dropout, training=self.training)
            embs.append(emb)
        
        # Combine embeddings from all layers
        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        
        return final_emb
    
    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Predict ratings for user-item pairs
        
        Args:
            users: User IDs (1-indexed)
            items: Item IDs (1-indexed)
            
        Returns:
            Predicted scores
        """
        emb = self.forward()
        
        # Get embeddings
        user_emb = emb[users - 1]
        item_emb = emb[self.n_users + items - 1]
        
        # Basic inner product
        base_score = (user_emb * item_emb).sum(dim=1)
        
        if self.use_attention:
            # Attention-based adjustment
            concat_emb = torch.cat([user_emb, item_emb], dim=1)
            attention_weight = self.attention(concat_emb).squeeze()
            score = base_score * (1 + attention_weight)
        else:
            score = base_score
        
        return score
    
    def get_user_embeddings(self) -> torch.Tensor:
        """Get user embeddings after propagation"""
        emb = self.forward()
        return emb[:self.n_users]
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get item embeddings after propagation"""
        emb = self.forward()
        return emb[self.n_users:self.n_users + self.n_items]
    
    def get_all_predictions(self, users: torch.Tensor) -> torch.Tensor:
        """Get predictions for all items for given users
        
        Args:
            users: User IDs (1-indexed)
            
        Returns:
            Predictions matrix [n_users, n_items]
        """
        user_emb = self.get_user_embeddings()[users - 1]
        item_emb = self.get_item_embeddings()
        
        # Compute all scores
        scores = torch.matmul(user_emb, item_emb.t())
        
        if self.use_attention:
            # Apply attention to all pairs
            n_users = user_emb.shape[0]
            n_items = item_emb.shape[0]
            
            # Expand embeddings for broadcasting
            user_emb_exp = user_emb.unsqueeze(1).expand(n_users, n_items, -1)
            item_emb_exp = item_emb.unsqueeze(0).expand(n_users, n_items, -1)
            
            # Concatenate and compute attention
            concat_emb = torch.cat([user_emb_exp, item_emb_exp], dim=-1)
            attention_weights = self.attention(concat_emb.view(-1, concat_emb.shape[-1])).view(n_users, n_items)
            
            scores = scores * (1 + attention_weights)
        
        return scores