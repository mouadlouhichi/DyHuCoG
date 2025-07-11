import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from shapley import PreferenceShapley

class LightGCNLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        deg_inv = deg.pow(-0.5)
        norm = deg_inv[row] * deg_inv[col]
        return self.propagate(edge_index, x=x, norm=norm.unsqueeze(-1))

    def message(self, x_j, norm):
        return norm * x_j

class DyHuCoG(nn.Module):
    def __init__(self, num_users, num_items, cfg):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.cfg = cfg
        latent_dim = cfg['latent_dim']
        self.embed = nn.Embedding(num_users + num_items, latent_dim)
        self.layers = nn.ModuleList([LightGCNLayer() for _ in range(cfg['layers'])])
        self.shap = PreferenceShapley(cfg['shapley_samples'], cfg['alpha_pref'])

    def propagate_embeddings(self, g):
        x = self.embed.weight
        for layer in self.layers:
            x = layer(x, g.edge_index)
        return x

    def forward(self, users, items, g):
        x = self.propagate_embeddings(g)
        u_emb = x[users]
        i_emb = x[items + self.num_users]

        # simple coalition = items interacted by user
        mask = g.edge_index[0] == users.unsqueeze(1)
        neigh_idx = g.edge_index[1][mask]  # (batch, variable) flattened
        # guard against empty coalition
        if neigh_idx.numel() == 0:
            score = (u_emb * i_emb).sum(-1)
            return torch.sigmoid(score)
        neigh_emb = x[neigh_idx]
        w = self.shap(neigh_emb, u_emb.mean(0))
        neigh_aggr = (w.unsqueeze(-1) * neigh_emb).sum(0)
        score = (u_emb * i_emb).sum(-1) + (u_emb * neigh_aggr).sum(-1)
        return torch.sigmoid(score)
