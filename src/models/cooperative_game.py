"""
Cooperative Game components for DyHuCoG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import numpy as np


class CooperativeGameDAE(nn.Module):
    """Denoising AutoEncoder for cooperative game value function
    
    Args:
        n_items: Number of items
        hidden_dim: Hidden dimension size
        dropout: Dropout rate
    """
    
    def __init__(self, n_items: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_items),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, noise_factor: float = 0.2) -> torch.Tensor:
        """Forward pass with optional noise injection
        
        Args:
            x: Input item vectors [batch_size, n_items]
            noise_factor: Noise level for denoising
            
        Returns:
            Reconstructed item vectors
        """
        # Add noise during training
        if self.training and noise_factor > 0:
            noise = torch.randn_like(x) * noise_factor
            x_noisy = x + noise
            x_noisy = torch.clamp(x_noisy, 0, 1)
        else:
            x_noisy = x
        
        # Encode and decode
        h = self.encoder(x_noisy)
        out = self.decoder(h)
        
        return out
    
    def get_coalition_value(self, coalition_vector: torch.Tensor) -> torch.Tensor:
        """Get value for a coalition (subset of items)
        
        Args:
            coalition_vector: Binary vector indicating coalition membership
            
        Returns:
            Coalition value
        """
        with torch.no_grad():
            reconstructed = self.forward(coalition_vector, noise_factor=0)
            # Value is the sum of reconstructed probabilities for items in coalition
            value = (reconstructed * coalition_vector).sum(dim=-1)
        return value
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation from encoder
        
        Args:
            x: Input item vectors
            
        Returns:
            Latent representations
        """
        return self.encoder(x)


class ShapleyValueNetwork(nn.Module):
    """FastSHAP-style network for Shapley value approximation
    
    Args:
        n_items: Number of items
        hidden_dim: Hidden dimension size
        n_samples: Number of samples for Shapley value estimation
    """
    
    def __init__(self, n_items: int, hidden_dim: int, n_samples: int = 10):
        super().__init__()
        
        self.n_items = n_items
        self.n_samples = n_samples
        
        # Deep network for Shapley value estimation
        self.network = nn.Sequential(
            nn.Linear(n_items, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_items)
        )
        
        # Initialize output layer with small weights
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate Shapley values for all items
        
        Args:
            x: Item interaction vectors [batch_size, n_items]
            
        Returns:
            Shapley values [batch_size, n_items]
        """
        return self.network(x)
    
    def compute_exact_shapley_sample(self, x: torch.Tensor, 
                                   value_function: Callable,
                                   n_samples: Optional[int] = None) -> torch.Tensor:
        """Monte Carlo approximation of Shapley values for training
        
        Args:
            x: Item interaction vectors [batch_size, n_items]
            value_function: Coalition value function
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Approximate Shapley values [batch_size, n_items]
        """
        if n_samples is None:
            n_samples = self.n_samples
            
        batch_size = x.shape[0]
        n_items = x.shape[1]
        device = x.device
        shapley_values = torch.zeros_like(x)
        
        for b in range(batch_size):
            user_items = x[b]
            item_indices = torch.where(user_items > 0)[0]
            n_user_items = len(item_indices)
            
            if n_user_items == 0:
                continue
                
            item_shapley = torch.zeros(n_user_items, device=device)
            
            # Monte Carlo sampling
            for _ in range(n_samples):
                # Random permutation
                perm = torch.randperm(n_user_items, device=device)
                
                # Compute marginal contributions
                coalition = torch.zeros(n_items, device=device)
                prev_value = value_function(coalition.unsqueeze(0)).item()
                
                for idx in perm:
                    item_idx = item_indices[idx]
                    coalition[item_idx] = 1
                    curr_value = value_function(coalition.unsqueeze(0)).item()
                    
                    marginal = curr_value - prev_value
                    item_shapley[idx] += marginal / n_samples
                    
                    prev_value = curr_value
            
            # Assign Shapley values
            for i, idx in enumerate(item_indices):
                shapley_values[b, idx] = item_shapley[i]
                
        return shapley_values
    
    def compute_shapley_loss(self, pred_shapley: torch.Tensor,
                           target_shapley: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """Compute loss for Shapley value prediction
        
        Args:
            pred_shapley: Predicted Shapley values
            target_shapley: Target Shapley values
            mask: Mask for valid items
            
        Returns:
            Loss value
        """
        # Only compute loss on observed items
        if mask.sum() > 0:
            loss = F.mse_loss(pred_shapley[mask], target_shapley[mask])
        else:
            loss = torch.tensor(0.0, device=pred_shapley.device)
            
        return loss


class CooperativeGameTrainer:
    """Trainer for cooperative game components
    
    Args:
        dae: Cooperative game DAE model
        shapley_net: Shapley value network
        config: Training configuration
    """
    
    def __init__(self, dae: CooperativeGameDAE, 
                 shapley_net: ShapleyValueNetwork,
                 config: dict):
        self.dae = dae
        self.shapley_net = shapley_net
        self.config = config
        
        # Optimizers
        self.dae_optimizer = torch.optim.Adam(
            dae.parameters(), 
            lr=config.get('lr', 0.001)
        )
        
        self.shapley_optimizer = torch.optim.Adam(
            shapley_net.parameters(),
            lr=config.get('lr', 0.001)
        )
        
    def train_dae_step(self, user_items: torch.Tensor) -> float:
        """Single training step for DAE
        
        Args:
            user_items: User-item interaction matrix [batch_size, n_items + 1]
            
        Returns:
            Loss value
        """
        # Remove the first column (index 0) which is padding for 1-indexed data
        user_items = user_items[:, 1:]  # Now shape is [batch_size, n_items]
        
        # Forward pass
        reconstructed = self.dae(user_items)
        
        # Compute reconstruction loss on ALL items, not just observed ones
        # This encourages the model to predict 0 for unobserved items
        loss = F.binary_cross_entropy(reconstructed, user_items)
        
        # Alternatively, you can use a weighted loss to give more importance to observed items
        # pos_weight = (user_items == 0).sum() / (user_items == 1).sum()
        # loss = F.binary_cross_entropy_with_logits(reconstructed, user_items, pos_weight=pos_weight)
        
        # Backward pass
        self.dae_optimizer.zero_grad()
        loss.backward()
        self.dae_optimizer.step()
        
        return loss.item()
    def train_shapley_step(self, user_items: torch.Tensor) -> float:
        """Single training step for Shapley network
        
        Args:
            user_items: User-item interaction matrix [batch_size, n_items + 1]
            
        Returns:
            Loss value
        """
        # Remove the first column (index 0) which is padding for 1-indexed data
        user_items = user_items[:, 1:]  # Now shape is [batch_size, n_items]
        
        # Predict Shapley values
        pred_shapley = self.shapley_net(user_items)
        
        # Compute target Shapley values
        with torch.no_grad():
            target_shapley = self.shapley_net.compute_exact_shapley_sample(
                user_items, 
                self.dae.get_coalition_value,
                n_samples=self.config.get('n_shapley_samples', 10)
            )
        
        # Compute loss
        mask = user_items > 0
        loss = self.shapley_net.compute_shapley_loss(
            pred_shapley, target_shapley, mask
        )
        
        # Backward pass
        self.shapley_optimizer.zero_grad()
        loss.backward()
        self.shapley_optimizer.step()
        
        return loss.item()