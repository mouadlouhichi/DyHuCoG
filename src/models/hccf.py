"""
HCCF: Hypergraph Contrastive Collaborative Filtering
Reference: Xia et al., "Hypergraph Contrastive Collaborative Filtering", SIGIR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class HCCF(nn.Module):
    """Hypergraph Contrastive Collaborative Filtering
    
    Args:
        n_users: Number of users
        n_items: Number of items
        latent_dim: Embedding dimension
        n_layers: Number of propagation layers
        n_hyperedges: Number of hyperedges to construct
        ssl_temp: Temperature for contrastive loss
        ssl_reg: SSL loss weight
    """
    
    def __init__(self, n_users: int, n_items: int, latent_dim: int,
                 n_layers: int, n_hyperedges: int = 1000,
                 ssl_temp: float = 0.2, ssl_reg: float = 0.1):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.n_hyperedges = n_hyperedges
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Hypergraph will be built from data
        self.H = None  # Incidence matrix
        
    def build_hypergraph(self, train_mat: torch.sparse.Tensor) -> None:
        """Build hypergraph from training data
        
        Args:
            train_mat: Training interaction matrix
        """
        # Convert to dense for processing
        train_dense = train_mat.to_dense()
        
        # Sample hyperedges (user groups with common items)
        hyperedges = []
        
        # Method 1: Users who rated same items
        for item in range(self.n_items):
            users = torch.where(train_dense[:, item] > 0)[0]
            if len(users) > 1:
                hyperedges.append(users.tolist())
        
        # Sample if too many
        if len(hyperedges) > self.n_hyperedges:
            indices = np.random.choice(len(hyperedges), self.n_hyperedges, replace=False)
            hyperedges = [hyperedges[i] for i in indices]
        
        # Build incidence matrix H
        n_nodes = self.n_users + self.n_items
        n_edges = len(hyperedges)
        
        H_indices = []
        H_values = []
        
        for edge_idx, nodes in enumerate(hyperedges):
            for node in nodes:
                H_indices.append([node, edge_idx])
                H_values.append(1.0 / np.sqrt(len(nodes)))
        
        if H_indices:
            H_indices = torch.tensor(H_indices).T
            H_values = torch.tensor(H_values)
            self.H = torch.sparse_coo_tensor(
                H_indices, H_values, (n_nodes, n_edges)
            )
        
    def hypergraph_propagation(self, emb: torch.Tensor) -> torch.Tensor:
        """Propagate embeddings through hypergraph
        
        Args:
            emb: Node embeddings
            
        Returns:
            Propagated embeddings
        """
        if self.H is None:
            return emb
            
        # H * H^T gives the hypergraph Laplacian
        # Simplified propagation: emb' = H @ H^T @ emb
        H = self.H.to(emb.device)
        
        # Step 1: H^T @ emb
        hyperedge_emb = torch.sparse.mm(H.t(), emb)
        
        # Step 2: H @ hyperedge_emb
        propagated = torch.sparse.mm(H, hyperedge_emb)
        
        return propagated
    
    def forward(self, adj: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation combining graph and hypergraph
        
        Args:
            adj: Standard adjacency matrix
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Initial embeddings
        emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [emb]
        
        # Multi-layer propagation
        for _ in range(self.n_layers):
            # Graph propagation
            graph_emb = torch.sparse.mm(adj, emb)
            
            # Hypergraph propagation
            if self.H is not None:
                hyper_emb = self.hypergraph_propagation(emb)
                # Combine graph and hypergraph
                emb = 0.8 * graph_emb + 0.2 * hyper_emb
            else:
                emb = graph_emb
                
            embs.append(emb)
        
        # Aggregate
        emb = torch.stack(embs, dim=1).mean(dim=1)
        
        # Split
        user_emb = emb[:self.n_users]
        item_emb = emb[self.n_users:self.n_users + self.n_items]
        
        return user_emb, item_emb
    
    def ssl_loss(self, users: torch.Tensor, items: torch.Tensor,
                 adj: torch.sparse.Tensor) -> torch.Tensor:
        """Contrastive loss between graph and hypergraph views"""
        # Graph view
        user_emb_g, item_emb_g = self.forward(adj)
        
        # Create perturbed hypergraph for second view
        # (In practice, could use different hyperedge sampling)
        user_emb_h, item_emb_h = self.forward(adj)  # Simplified
        
        # Get batch embeddings
        users_idx = users - 1
        items_idx = items - 1
        
        u_g = F.normalize(user_emb_g[users_idx], dim=1)
        u_h = F.normalize(user_emb_h[users_idx], dim=1)
        i_g = F.normalize(item_emb_g[items_idx], dim=1)
        i_h = F.normalize(item_emb_h[items_idx], dim=1)
        
        # Contrastive loss
        pos_u = (u_g * u_h).sum(dim=1) / self.ssl_temp
        neg_u = torch.matmul(u_g, user_emb_h.T) / self.ssl_temp
        loss_u = -torch.log(torch.exp(pos_u) / torch.exp(neg_u).sum(dim=1)).mean()
        
        pos_i = (i_g * i_h).sum(dim=1) / self.ssl_temp
        neg_i = torch.matmul(i_g, item_emb_h.T) / self.ssl_temp
        loss_i = -torch.log(torch.exp(pos_i) / torch.exp(neg_i).sum(dim=1)).mean()
        
        return self.ssl_reg * (loss_u + loss_i)
    
    def predict(self, users: torch.Tensor, items: torch.Tensor,
                adj: torch.sparse.Tensor) -> torch.Tensor:
        """Predict ratings"""
        user_emb, item_emb = self.forward(adj)
        
        users = users - 1
        items = items - 1
        
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        
        return (u_emb * i_emb).sum(dim=1)
    
    def get_all_predictions(self, users: torch.Tensor,
                           adj: torch.sparse.Tensor) -> torch.Tensor:
        """Get all predictions"""
        user_emb, item_emb = self.forward(adj)
        users = users - 1
        u_emb = user_emb[users]
        
        return torch.matmul(u_emb, item_emb.t())