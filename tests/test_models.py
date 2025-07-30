"""Tests for model implementations"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.dyhucog import DyHuCoG
from src.models.lightgcn import LightGCN
from src.models.ngcf import NGCF
from src.models.cooperative_game import CooperativeGameDAE, ShapleyValueNetwork
from src.utils.graph_builder import GraphBuilder


@pytest.fixture
def device():
    """Get test device"""
    return torch.device('cpu')


@pytest.fixture
def dummy_dataset():
    """Create dummy dataset for testing"""
    class DummyDataset:
        def __init__(self):
            self.n_users = 100
            self.n_items = 50
            self.n_genres = 5
            self.n_interactions = 500
            
            # Create dummy interaction matrix
            self.train_mat = torch.zeros(self.n_users + 1, self.n_items + 1)
            for _ in range(self.n_interactions):
                u = np.random.randint(1, self.n_users + 1)
                i = np.random.randint(1, self.n_items + 1)
                self.train_mat[u, i] = 1
                
            # Create dummy item genres
            self.item_genres = {}
            for i in range(1, self.n_items + 1):
                n_genres = np.random.randint(1, 4)
                self.item_genres[i] = np.random.choice(
                    range(self.n_genres), n_genres, replace=False
                ).tolist()
                
    return DummyDataset()


@pytest.fixture
def model_config():
    """Get model configuration"""
    return {
        'latent_dim': 32,
        'n_layers': 2,
        'dropout': 0.1,
        'dae_hidden': 64,
        'shapley_hidden': 64,
        'n_shapley_samples': 5,
        'use_attention': True,
        'use_genres': True
    }


class TestDyHuCoG:
    """Test DyHuCoG model"""
    
    def test_initialization(self, dummy_dataset, model_config, device):
        """Test model initialization"""
        model = DyHuCoG(
            n_users=dummy_dataset.n_users,
            n_items=dummy_dataset.n_items,
            n_genres=dummy_dataset.n_genres,
            config=model_config
        ).to(device)
        
        assert model.n_users == dummy_dataset.n_users
        assert model.n_items == dummy_dataset.n_items
        assert model.n_genres == dummy_dataset.n_genres
        assert model.latent_dim == model_config['latent_dim']
        
    def test_forward_pass(self, dummy_dataset, model_config, device):
        """Test forward propagation"""
        model = DyHuCoG(
            n_users=dummy_dataset.n_users,
            n_items=dummy_dataset.n_items,
            n_genres=dummy_dataset.n_genres,
            config=model_config
        ).to(device)
        
        # Build graph
        edge_index = torch.tensor([[0, 1], [1, 0]], device=device).t()
        edge_weight = torch.tensor([1.0, 1.0], device=device)
        model.build_hypergraph(edge_index, edge_weight, dummy_dataset.item_genres)
        
        # Forward pass
        embeddings = model.forward()
        
        assert embeddings.shape[0] == model.n_nodes
        assert embeddings.shape[1] == model_config['latent_dim']
        
    def test_predict(self, dummy_dataset, model_config, device):
        """Test prediction"""
        model = DyHuCoG(
            n_users=dummy_dataset.n_users,
            n_items=dummy_dataset.n_items,
            n_genres=dummy_dataset.n_genres,
            config=model_config
        ).to(device)
        
        # Build graph
        edge_index = torch.tensor([[0, 1], [1, 0]], device=device).t()
        edge_weight = torch.tensor([1.0, 1.0], device=device)
        model.build_hypergraph(edge_index, edge_weight, dummy_dataset.item_genres)
        
        # Test prediction
        users = torch.tensor([1, 2, 3], device=device)
        items = torch.tensor([1, 2, 3], device=device)
        
        scores = model.predict(users, items)
        
        assert scores.shape[0] == 3
        assert not torch.isnan(scores).any()
        
    def test_shapley_computation(self, dummy_dataset, model_config, device):
        """Test Shapley value computation"""
        model = DyHuCoG(
            n_users=dummy_dataset.n_users,
            n_items=dummy_dataset.n_items,
            n_genres=dummy_dataset.n_genres,
            config=model_config
        ).to(device)
        
        # Compute Shapley weights
        edge_weights = model.compute_shapley_weights(dummy_dataset.train_mat.to(device))
        
        assert isinstance(edge_weights, dict)
        # Check that weights are positive
        for weight in edge_weights.values():
            assert weight > 0


class TestLightGCN:
    """Test LightGCN model"""
    
    def test_initialization(self, dummy_dataset, model_config, device):
        """Test model initialization"""
        model = LightGCN(
            n_users=dummy_dataset.n_users,
            n_items=dummy_dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers']
        ).to(device)
        
        assert model.n_users == dummy_dataset.n_users
        assert model.n_items == dummy_dataset.n_items
        
    def test_forward_pass(self, dummy_dataset, model_config, device):
        """Test forward propagation"""
        model = LightGCN(
            n_users=dummy_dataset.n_users,
            n_items=dummy_dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers']
        ).to(device)
        
        # Build adjacency matrix
        adj, _ = GraphBuilder.build_user_item_graph(dummy_dataset)
        adj = GraphBuilder.normalize_adj(adj).to(device)
        
        # Forward pass
        user_emb, item_emb = model.forward(adj)
        
        assert user_emb.shape == (dummy_dataset.n_users, model_config['latent_dim'])
        assert item_emb.shape == (dummy_dataset.n_items, model_config['latent_dim'])
        
    def test_predict(self, dummy_dataset, model_config, device):
        """Test prediction"""
        model = LightGCN(
            n_users=dummy_dataset.n_users,
            n_items=dummy_dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers']
        ).to(device)
        
        # Build adjacency matrix
        adj, _ = GraphBuilder.build_user_item_graph(dummy_dataset)
        adj = GraphBuilder.normalize_adj(adj).to(device)
        
        # Test prediction
        users = torch.tensor([1, 2, 3], device=device)
        items = torch.tensor([1, 2, 3], device=device)
        
        scores = model.predict(users, items, adj)
        
        assert scores.shape[0] == 3
        assert not torch.isnan(scores).any()


class TestNGCF:
    """Test NGCF model"""
    
    def test_initialization(self, dummy_dataset, model_config, device):
        """Test model initialization"""
        model = NGCF(
            n_users=dummy_dataset.n_users,
            n_items=dummy_dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout']
        ).to(device)
        
        assert model.n_users == dummy_dataset.n_users
        assert model.n_items == dummy_dataset.n_items
        assert len(model.propagation_layers) == model_config['n_layers']
        
    def test_forward_pass(self, dummy_dataset, model_config, device):
        """Test forward propagation"""
        model = NGCF(
            n_users=dummy_dataset.n_users,
            n_items=dummy_dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout']
        ).to(device)
        
        # Build adjacency matrix
        adj, _ = GraphBuilder.build_user_item_graph(dummy_dataset)
        adj = GraphBuilder.normalize_adj(adj).to(device)
        
        # Forward pass
        user_emb, item_emb = model.forward(adj)
        
        # NGCF concatenates embeddings from all layers
        expected_dim = model_config['latent_dim'] * (model_config['n_layers'] + 1)
        assert user_emb.shape == (dummy_dataset.n_users, expected_dim)
        assert item_emb.shape == (dummy_dataset.n_items, expected_dim)


class TestCooperativeGame:
    """Test cooperative game components"""
    
    def test_dae(self, dummy_dataset, device):
        """Test Denoising AutoEncoder"""
        dae = CooperativeGameDAE(
            n_items=dummy_dataset.n_items,
            hidden_dim=64,
            dropout=0.1
        ).to(device)
        
        # Create dummy input
        batch_size = 10
        x = torch.rand(batch_size, dummy_dataset.n_items, device=device)
        
        # Forward pass
        output = dae(x)
        
        assert output.shape == (batch_size, dummy_dataset.n_items)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        
    def test_shapley_network(self, dummy_dataset, device):
        """Test Shapley value network"""
        shapley_net = ShapleyValueNetwork(
            n_items=dummy_dataset.n_items,
            hidden_dim=64,
            n_samples=5
        ).to(device)
        
        # Create dummy input
        batch_size = 10
        x = torch.rand(batch_size, dummy_dataset.n_items, device=device)
        
        # Forward pass
        shapley_values = shapley_net(x)
        
        assert shapley_values.shape == (batch_size, dummy_dataset.n_items)
        
    def test_shapley_computation(self, dummy_dataset, device):
        """Test exact Shapley value computation"""
        shapley_net = ShapleyValueNetwork(
            n_items=dummy_dataset.n_items,
            hidden_dim=64,
            n_samples=5
        ).to(device)
        
        dae = CooperativeGameDAE(
            n_items=dummy_dataset.n_items,
            hidden_dim=64
        ).to(device)
        
        # Create dummy input
        x = torch.zeros(1, dummy_dataset.n_items, device=device)
        x[0, :5] = 1  # User has 5 items
        
        # Compute Shapley values
        shapley_values = shapley_net.compute_exact_shapley_sample(
            x, dae.get_coalition_value, n_samples=10
        )
        
        assert shapley_values.shape == (1, dummy_dataset.n_items)
        # Non-zero items should have non-zero Shapley values
        assert torch.sum(shapley_values[0, :5]) > 0


class TestGraphBuilder:
    """Test graph building utilities"""
    
    def test_user_item_graph(self, dummy_dataset, device):
        """Test user-item graph construction"""
        adj, n_nodes = GraphBuilder.build_user_item_graph(dummy_dataset)
        
        assert n_nodes == dummy_dataset.n_users + dummy_dataset.n_items
        assert adj.shape == (n_nodes, n_nodes)
        
    def test_hypergraph(self, dummy_dataset, device):
        """Test hypergraph construction"""
        adj, n_nodes = GraphBuilder.build_hypergraph(dummy_dataset)
        
        expected_nodes = (dummy_dataset.n_users + 
                         dummy_dataset.n_items + 
                         dummy_dataset.n_genres)
        assert n_nodes == expected_nodes
        assert adj.shape == (n_nodes, n_nodes)
        
    def test_normalization(self, device):
        """Test adjacency matrix normalization"""
        # Create simple adjacency matrix
        indices = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device)
        values = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
        adj = torch.sparse_coo_tensor(indices, values, (3, 3))
        
        # Normalize
        norm_adj = GraphBuilder.normalize_adj(adj)
        
        assert norm_adj.shape == adj.shape
        # Check that values are normalized
        assert torch.all(norm_adj._values() <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])