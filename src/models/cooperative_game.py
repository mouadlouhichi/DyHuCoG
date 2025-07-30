import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

class CooperativeGameDAE(nn.Module):
    """Denoising AutoEncoder for cooperative game value function"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ShapleyValueNetwork(nn.Module):
    """Light network to predict Shapley values from user features"""
    def __init__(self, input_dim: int, hidden_dim: int, n_items: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.head = nn.Linear(hidden_dim, n_items)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        return self.head(h)

    def compute_exact_shapley_sample(
        self,
        x: torch.Tensor,
        value_function: Callable,
        n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Vectorized Monte Carlo Shapley approximation:
        - x: [B, M] interaction matrix
        - value_function: f(batch_x) -> [batch_size] values
        - n_samples: number of random masks per user
        Returns: [B, M] approximate Shapley values
        """
        if n_samples is None:
            raise ValueError("n_samples must be provided for Shapley sampling")

        B, M = x.shape
        k = n_samples
        device = x.device

        # 1) Generate random subset masks: [B, k, M]
        masks = torch.zeros(B, k, M, device=device)
        for b in range(B):
            idxs = torch.where(x[b] > 0)[0]
            if idxs.numel() == 0:
                continue
            n_i = idxs.size(0)
            # sample random sizes and subsets
            sizes = torch.randint(0, n_i + 1, (k,), device=device)
            for s in range(k):
                perm = idxs[torch.randperm(n_i, device=device)[:sizes[s]]]
                masks[b, s, perm] = 1.0

        # 2) Compute value for masked and full features
        x_exp = x.unsqueeze(1).expand(-1, k, -1).reshape(B * k, M)
        mask_flat = masks.reshape(B * k, M)
        vals_masked = value_function(x_exp * mask_flat).view(B, k)

        full_vals = value_function(x).view(B, 1)

        # 3) Compute marginal contributions per sample and item
        #    delta = full_vals - val_masked for each sample
        deltas = (full_vals - vals_masked) / k          # [B, k]
        # broadcast to items and weight by mask
        contrib = deltas.unsqueeze(-1) * masks         # [B, k, M]

        # 4) Average contributions across samples
        shapley_vals = contrib.mean(dim=1)             # [B, M]

        return shapley_vals

class CooperativeGameTrainer:
    """Trainer for cooperative game components"""
    def __init__(self, dae: CooperativeGameDAE, shapley_net: ShapleyValueNetwork, config: dict):
        self.dae = dae
        self.shapley_net = shapley_net
        self.config = config
        self.dae_opt = torch.optim.Adam(dae.parameters(), lr=config['lr'])
        self.shap_opt = torch.optim.Adam(shapley_net.parameters(), lr=config['lr'])

    def pretrain_shapley(self, user_items: torch.Tensor):
        """Run one epoch of Shapley pretraining"""
        n_samples = self.config.get('n_shapley_samples', 5)
        batch = user_items[:, 1:]  # remove padding

        # 1) exact shapley via vectorized sampling
        with torch.no_grad():
            target = self.shapley_net.compute_exact_shapley_sample(
                batch, self.dae.forward, n_samples=n_samples
            )

        # 2) predict from shared features
        self.shap_opt.zero_grad()
        shared = self.shapley_net.shared(batch)
        pred = self.shapley_net.head(shared)
        loss = F.mse_loss(pred, target)
        loss.backward()
        self.shap_opt.step()
        return loss.item()
