"""Preference-aware Monte Carlo Shapley approximation."""
import torch, numba, numpy as np
from torch import nn

@numba.njit(cache=True, fastmath=True)
def _mc_shapley(vals, S, out):
    n = vals.shape[0]
    for s in range(S):
        perm = np.random.permutation(n)
        prev = 0.0
        for p in perm:
            cur = vals[p]
            out[p] += (cur - prev) / S
            prev = cur

class PreferenceShapley(nn.Module):
    def __init__(self, samples=128, alpha=0.7):
        super().__init__()
        self.S = samples
        self.alpha = alpha

    def forward(self, neigh_emb, tgt_emb):
        # neigh_emb: |C| x d ; tgt_emb: d
        sim = torch.cosine_similarity(neigh_emb, tgt_emb.unsqueeze(0), dim=-1)
        pref = sim.pow(self.alpha)
        vals = (neigh_emb @ tgt_emb) * pref  # |C|
        vals_np = vals.detach().cpu().numpy()
        out = np.zeros_like(vals_np)
        _mc_shapley(vals_np, self.S, out)
        w = torch.from_numpy(out).to(neigh_emb.device)
        return torch.softmax(w, dim=0)
