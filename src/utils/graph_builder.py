"""Graph construction utilities for recommendation models"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import scipy.sparse as sp


class GraphBuilder:
    """Build various graph structures for recommendation models"""
    
    @staticmethod
    def build_user_item_graph(dataset) -> Tuple[torch.sparse.Tensor, int]:
        """Build user-item bipartite graph
        
        Args:
            dataset: RecommenderDataset object
            
        Returns:
            Tuple of (adjacency matrix, number of nodes)
        """
        edge_index = []
        edge_weight = []
        
        # Add edges from training data
        for _, row in dataset.train_ratings.iterrows():
            u = row['user'] - 1  # Convert to 0-indexed
            i = row['item'] - 1 + dataset.n_users  # Item nodes come after user nodes
            
            # Bidirectional edges
            edge_index.append([u, i])
            edge_index.append([i, u])
            edge_weight.extend([1.0, 1.0])
            
        # Convert to tensors
        edge_index = torch.LongTensor(edge_index).t()
        edge_weight = torch.FloatTensor(edge_weight)
        
        # Create sparse adjacency matrix
        n_nodes = dataset.n_users + dataset.n_items
        adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, (n_nodes, n_nodes)
        )
        
        return adj, n_nodes
    
    @staticmethod
    def build_hypergraph(dataset) -> Tuple[torch.sparse.Tensor, int]:
        """Build hypergraph with user-item-genre connections
        
        Args:
            dataset: RecommenderDataset object
            
        Returns:
            Tuple of (adjacency matrix, number of nodes)
        """
        edge_index = []
        edge_weight = []
        
        # User-item edges
        for _, row in dataset.train_ratings.iterrows():
            u = row['user'] - 1
            i = row['item'] - 1 + dataset.n_users
            
            edge_index.append([u, i])
            edge_index.append([i, u])
            edge_weight.extend([1.0, 1.0])
        
        # Item-genre edges
        if hasattr(dataset, 'item_genres'):
            for item_id, genres in dataset.item_genres.items():
                if genres:  # If item has genres
                    item_node = item_id - 1 + dataset.n_users
                    for genre in genres:
                        genre_node = dataset.n_users + dataset.n_items + genre
                        
                        edge_index.append([item_node, genre_node])
                        edge_index.append([genre_node, item_node])
                        edge_weight.extend([1.0, 1.0])
        
        # Convert to tensors
        edge_index = torch.LongTensor(edge_index).t()
        edge_weight = torch.FloatTensor(edge_weight)
        
        # Create sparse adjacency matrix
        n_nodes = dataset.n_users + dataset.n_items
        if hasattr(dataset, 'n_genres'):
            n_nodes += dataset.n_genres
            
        adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, (n_nodes, n_nodes)
        )
        
        return adj, n_nodes
    
    @staticmethod
    def normalize_adj(adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        """Symmetric normalization of adjacency matrix
        
        D^(-1/2) * A * D^(-1/2)
        
        Args:
            adj: Sparse adjacency matrix
            
        Returns:
            Normalized adjacency matrix
        """
        # Get dimensions
        n_nodes = adj.shape[0]
        
        # Add self-loops
        indices = adj._indices()
        values = adj._values()
        
        # Add self-loop indices
        self_loop_indices = torch.arange(n_nodes, device=adj.device).unsqueeze(0).repeat(2, 1)
        indices = torch.cat([indices, self_loop_indices], dim=1)
        values = torch.cat([values, torch.ones(n_nodes, device=adj.device)])
        
        # Create new adjacency with self-loops
        adj_with_self_loops = torch.sparse_coo_tensor(
            indices, values, adj.shape
        ).coalesce()
        
        # Compute degree
        degree = torch.sparse.sum(adj_with_self_loops, dim=1).to_dense()
        degree = torch.clamp(degree, min=1e-6)  # Avoid division by zero
        
        # D^(-1/2)
        d_inv_sqrt = torch.pow(degree, -0.5)
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        indices = adj_with_self_loops._indices()
        values = adj_with_self_loops._values()
        
        row, col = indices
        norm_values = values * d_inv_sqrt[row] * d_inv_sqrt[col]
        
        normalized_adj = torch.sparse_coo_tensor(
            indices, norm_values, adj.shape
        )
        
        return normalized_adj
    
    @staticmethod
    def get_edge_list(dataset, edge_weights: Optional[Dict[Tuple[int, int], float]] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get edge list representation of the graph
        
        Args:
            dataset: RecommenderDataset object
            edge_weights: Optional dictionary of edge weights
            
        Returns:
            Tuple of (edge_index, edge_weight)
        """
        edge_list = []
        weight_list = []
        
        for _, row in dataset.train_ratings.iterrows():
            u = row['user'] - 1
            i = row['item'] - 1 + dataset.n_users
            
            # Get weight
            if edge_weights:
                weight = edge_weights.get((row['user'], row['item']), 1.0)
            else:
                weight = 1.0
                
            edge_list.append([u, i])
            edge_list.append([i, u])
            weight_list.extend([weight, weight])
            
        edge_index = torch.LongTensor(edge_list).t()
        edge_weight = torch.FloatTensor(weight_list)
        
        return edge_index, edge_weight
    
    @staticmethod
    def build_social_graph(dataset, social_data: pd.DataFrame) -> torch.sparse.Tensor:
        """Build social network graph
        
        Args:
            dataset: RecommenderDataset object
            social_data: DataFrame with columns ['user1', 'user2'] for social connections
            
        Returns:
            Social adjacency matrix
        """
        edge_index = []
        edge_weight = []
        
        for _, row in social_data.iterrows():
            u1 = row['user1'] - 1
            u2 = row['user2'] - 1
            
            # Bidirectional edges
            edge_index.append([u1, u2])
            edge_index.append([u2, u1])
            edge_weight.extend([1.0, 1.0])
            
        edge_index = torch.LongTensor(edge_index).t()
        edge_weight = torch.FloatTensor(edge_weight)
        
        social_adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, (dataset.n_users, dataset.n_users)
        )
        
        return social_adj
    
    @staticmethod
    def combine_graphs(adj1: torch.sparse.Tensor, adj2: torch.sparse.Tensor,
                      alpha: float = 0.5) -> torch.sparse.Tensor:
        """Combine two graphs with weighted sum
        
        Args:
            adj1: First adjacency matrix
            adj2: Second adjacency matrix
            alpha: Weight for first graph (1-alpha for second)
            
        Returns:
            Combined adjacency matrix
        """
        # Ensure same dimensions
        assert adj1.shape == adj2.shape, "Adjacency matrices must have same shape"
        
        # Weighted combination
        combined = alpha * adj1 + (1 - alpha) * adj2
        
        return combined.coalesce()
    
    @staticmethod
    def laplacian_matrix(adj: torch.sparse.Tensor, 
                        normalized: bool = True) -> torch.sparse.Tensor:
        """Compute graph Laplacian
        
        Args:
            adj: Adjacency matrix
            normalized: Whether to use normalized Laplacian
            
        Returns:
            Laplacian matrix
        """
        n_nodes = adj.shape[0]
        
        # Compute degree
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        
        if normalized:
            # Normalized Laplacian: I - D^(-1/2) * A * D^(-1/2)
            d_inv_sqrt = torch.pow(torch.clamp(degree, min=1e-6), -0.5)
            
            # D^(-1/2) * A * D^(-1/2)
            indices = adj._indices()
            values = adj._values()
            row, col = indices
            norm_values = values * d_inv_sqrt[row] * d_inv_sqrt[col]
            
            norm_adj = torch.sparse_coo_tensor(indices, norm_values, adj.shape)
            
            # I - normalized_adj
            identity = torch.sparse_coo_tensor(
                torch.arange(n_nodes).unsqueeze(0).repeat(2, 1),
                torch.ones(n_nodes),
                (n_nodes, n_nodes)
            )
            
            laplacian = identity - norm_adj
        else:
            # Unnormalized Laplacian: D - A
            degree_matrix = torch.sparse_coo_tensor(
                torch.arange(n_nodes).unsqueeze(0).repeat(2, 1),
                degree,
                (n_nodes, n_nodes)
            )
            
            laplacian = degree_matrix - adj
            
        return laplacian.coalesce()