"""
Cooperative Game components for DyHuCoG - Optimized Version
"""

import time
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
    
    def compute_exact_shapley_sample_optimized(self, x: torch.Tensor, 
                                              value_function: Callable,
                                              n_samples: Optional[int] = None) -> torch.Tensor:
        """Optimized Monte Carlo approximation of Shapley values
        
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
        
        # Process in smaller sub-batches for memory efficiency
        sub_batch_size = min(16, batch_size)  # Process max 16 users at once
        
        for batch_start in range(0, batch_size, sub_batch_size):
            batch_end = min(batch_start + sub_batch_size, batch_size)
            batch_indices = list(range(batch_start, batch_end))
            
            # Collect all coalitions to evaluate for this sub-batch
            all_coalitions = []
            coalition_metadata = []  # Store (batch_idx, sample_idx, item_idx, step)
            
            for b_idx, b in enumerate(batch_indices):
                user_items = x[b]
                item_indices = torch.where(user_items > 0)[0]
                n_user_items = len(item_indices)
                
                if n_user_items == 0:
                    continue
                
                # Limit items for very active users to speed up computation
                if n_user_items > 50:
                    # Sample most important items based on some heuristic
                    # For now, just take first 50
                    item_indices = item_indices[:50]
                    n_user_items = 50
                
                # Generate coalitions for Monte Carlo sampling
                for sample_idx in range(n_samples):
                    perm = torch.randperm(n_user_items, device=device)
                    coalition = torch.zeros(n_items, device=device)
                    
                    # Empty coalition
                    all_coalitions.append(coalition.clone())
                    coalition_metadata.append((b_idx, sample_idx, -1, 0))
                    
                    # Build coalitions incrementally
                    for step, perm_idx in enumerate(perm):
                        item_idx = item_indices[perm_idx]
                        coalition[item_idx] = 1
                        all_coalitions.append(coalition.clone())
                        coalition_metadata.append((b_idx, sample_idx, perm_idx.item(), step + 1))
            
            if not all_coalitions:
                continue
                
            # Batch evaluate all coalitions
            coalition_batch = torch.stack(all_coalitions)
            with torch.no_grad():
                coalition_values = value_function(coalition_batch).squeeze()
            
            # Process results to compute marginal contributions
            values_dict = {}
            for i, (b_idx, sample_idx, item_idx, step) in enumerate(coalition_metadata):
                key = (b_idx, sample_idx, step)
                values_dict[key] = coalition_values[i].item()
            
            # Compute Shapley values from marginal contributions
            for b_idx, b in enumerate(batch_indices):
                user_items = x[b]
                item_indices = torch.where(user_items > 0)[0]
                
                if len(item_indices) == 0:
                    continue
                    
                # Limit items if necessary
                if len(item_indices) > 50:
                    item_indices = item_indices[:50]
                
                item_shapley = torch.zeros(len(item_indices), device=device)
                
                for sample_idx in range(n_samples):
                    # Get the permutation order
                    perm = torch.randperm(len(item_indices), device=device)
                    
                    for perm_position, perm_idx in enumerate(perm):
                        # Marginal contribution = v(S ∪ {i}) - v(S)
                        prev_value = values_dict.get((b_idx, sample_idx, perm_position), 0.0)
                        curr_value = values_dict.get((b_idx, sample_idx, perm_position + 1), 0.0)
                        
                        marginal = curr_value - prev_value
                        item_shapley[perm_idx] += marginal / n_samples
                
                # Assign Shapley values back
                for i, idx in enumerate(item_indices):
                    shapley_values[b, idx] = item_shapley[i]
                    
        return shapley_values
    
    def compute_exact_shapley_sample(self, x: torch.Tensor, 
                                   value_function: Callable,
                                   n_samples: Optional[int] = None) -> torch.Tensor:
        """Keep original method for compatibility but use optimized version"""
        return self.compute_exact_shapley_sample_optimized(x, value_function, n_samples)
    
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
        
        self._shapley_epoch = 0
        
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
        
        # Compute reconstruction loss on all items
        loss = F.binary_cross_entropy(reconstructed, user_items)
        
        # Backward pass
        self.dae_optimizer.zero_grad()
        loss.backward()
        self.dae_optimizer.step()
        
        return loss.item()
    
    def train_shapley_step(self, user_items: torch.Tensor) -> float:
        """Optimized training step for Shapley network"""
        
        # Remove the first column (index 0) which is padding for 1-indexed data
        user_items = user_items[:, 1:]  # Now shape is [batch_size, n_items]
        
        # For efficiency, limit batch size for Shapley computation
        max_shapley_batch = 32
        if user_items.shape[0] > max_shapley_batch:
            # Randomly sample a smaller batch
            indices = torch.randperm(user_items.shape[0])[:max_shapley_batch]
            user_items = user_items[indices]
        
        # Predict Shapley values
        pred_shapley = self.shapley_net(user_items)
        
        # Choose between exact computation and self-supervised loss
        use_exact_shapley = self._shapley_epoch < 3
        
        if use_exact_shapley:
            # Compute target Shapley values (only for first few epochs)
            with torch.no_grad():
                # Use fewer samples for faster training
                n_samples = min(3, self.config.get('n_shapley_samples', 10))
                target_shapley = self.shapley_net.compute_exact_shapley_sample_optimized(
                    user_items, 
                    self.dae.get_coalition_value,
                    n_samples=n_samples
                )
            
            # Compute supervised loss
            mask = user_items > 0
            loss = self.shapley_net.compute_shapley_loss(
                pred_shapley, target_shapley, mask
            )
        else:
            # Self-supervised loss: ensure non-zero items have positive Shapley values
            # and that Shapley values sum to a meaningful total
            mask = user_items > 0
            
            # Loss 1: Non-zero items should have positive Shapley values
            loss1 = F.relu(-pred_shapley[mask]).mean()
            
            # Loss 2: Shapley values should sum to something meaningful
            # (e.g., close to the number of items)
            shapley_sums = (pred_shapley * mask.float()).sum(dim=1)
            item_counts = mask.float().sum(dim=1)
            loss2 = F.mse_loss(shapley_sums, item_counts * 0.5)  # Target: half of item count
            
            # Loss 3: Regularization - prevent too large values
            loss3 = (pred_shapley[mask] ** 2).mean() * 0.01
            
            loss = loss1 + loss2 + loss3
        
        # Backward pass
        self.shapley_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.shapley_net.parameters(), max_norm=1.0)
        
        self.shapley_optimizer.step()
        
        return loss.item()
    
    def increment_epoch(self):
        """Call this at the end of each epoch"""
        self._shapley_epoch += 1