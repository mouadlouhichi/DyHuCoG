# Simplified DyHuCoG Demo - Analysis Version
# Reduced epochs for faster analysis

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import math
import matplotlib.pyplot as plt

# Set seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simplified hyperparameters for demo
latent_dim = 32
n_layers = 2
lr = 0.001
decay = 1e-4
batch_size = 512
epochs = 20  # Reduced for demo
ks = [5, 10]
dae_hidden = 64
fs_hidden = 64
fs_epochs = 10  # Reduced for demo
dae_epochs = 10  # Reduced for demo

# Path to MovieLens-100k
data_path = 'ml-100k/'

class RecommenderDataset:
    def __init__(self, path, is_baseline=False):
        self.path = path
        self.is_baseline = is_baseline
        self.load_ratings()
        self.load_items()
        self.build_interactions()
        self.identify_cold_users()
        self.build_graph()

    def load_ratings(self):
        self.ratings = pd.read_csv(self.path + 'u.data', sep='\t', 
                                 names=['user', 'item', 'rating', 'timestamp'], header=None)
        self.n_users = self.ratings['user'].max()
        self.n_items = self.ratings['item'].max()
        print(f"Dataset: {self.n_users} users, {self.n_items} items, {len(self.ratings)} ratings")
        
        # Implicit feedback, rating >3 as positive
        self.ratings = self.ratings[self.ratings['rating'] > 3]
        print(f"After filtering (rating>3): {len(self.ratings)} positive interactions")
        
        # Train test split
        self.train_ratings, self.test_ratings = train_test_split(
            self.ratings, test_size=0.2, random_state=seed)

    def load_items(self):
        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 
                     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                     'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.genre_cols = genre_cols
        self.n_genres = len(genre_cols) if not self.is_baseline else 0
        self.items = pd.read_csv(self.path + 'u.item', sep='|', 
                               names=['item', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols, 
                               encoding='latin-1', header=None)
        self.item_genres = [[] for _ in range(self.n_items + 1)]
        for row in self.items.itertuples():
            i = row[1]
            g = []
            for j, col in enumerate(genre_cols):
                if row[j + 6] == 1:
                    g.append(j)
            self.item_genres[i] = g

    def build_interactions(self):
        # Train mat for DAE, user x item, binary
        self.train_mat = np.zeros((self.n_users + 1, self.n_items + 1))
        for row in self.train_ratings.itertuples():
            self.train_mat[row.user, row.item] = 1
        self.train_mat = torch.FloatTensor(self.train_mat).to(device)
        # For test, user to list of items
        self.test_set = {u: [] for u in range(1, self.n_users + 1)}
        for row in self.test_ratings.itertuples():
            self.test_set[row.user].append(row.item)

    def identify_cold_users(self):
        self.cold_users = []
        self.warm_users = []
        for u in range(1, self.n_users + 1):
            if len(self.test_set[u]) == 0: continue
            train_count = (self.train_mat[u] > 0).sum().item()
            if train_count < 5:
                self.cold_users.append(u)
            else:
                self.warm_users.append(u)
        print(f"Cold users: {len(self.cold_users)}, Warm users: {len(self.warm_users)}")

    def build_graph(self):
        self.total_nodes = self.n_users + self.n_items + self.n_genres
        # Edge list for user-item from train
        ui_edges = []
        for row in self.train_ratings.itertuples():
            u = row.user
            i = self.n_users + row.item
            ui_edges.append([u, i])
            ui_edges.append([i, u])
        # Item-genre edges if not baseline
        ig_edges = []
        if not self.is_baseline:
            for i in range(1, self.n_items + 1):
                item_node = self.n_users + i
                for g in self.item_genres[i]:
                    genre_node = self.n_users + self.n_items + g + 1
                    ig_edges.append([item_node, genre_node])
                    ig_edges.append([genre_node, item_node])
        all_edges = ui_edges + ig_edges
        print(f"Graph: {self.total_nodes} nodes, {len(all_edges)} edges")
        indices = torch.LongTensor(np.array(all_edges).T).to(device)
        values = torch.FloatTensor(np.ones(len(all_edges))).to(device)
        self.adj = torch.sparse.FloatTensor(indices, values, 
                                          (self.total_nodes + 1, self.total_nodes + 1)).to(device)
        self.update_norm_adj()

    def update_norm_adj(self):
        degree = torch.sparse.sum(self.adj, dim=1).to_dense()
        degree[degree == 0] = 1
        d_inv_sqrt = degree ** -0.5
        d_indices = torch.stack((torch.arange(self.total_nodes + 1, device=device), 
                               torch.arange(self.total_nodes + 1, device=device)))
        d_values = d_inv_sqrt
        d_mat = torch.sparse.FloatTensor(d_indices, d_values, 
                                       (self.total_nodes + 1, self.total_nodes + 1)).to(device)
        self.norm_adj = torch.sparse.mm(d_mat, torch.sparse.mm(self.adj, d_mat))

class DAE(nn.Module):
    def __init__(self, n_items):
        super().__init__()
        self.encoder = nn.Linear(n_items + 1, dae_hidden)
        self.decoder = nn.Linear(dae_hidden, n_items + 1)

    def forward(self, x):
        h = F.relu(self.encoder(x))
        out = torch.sigmoid(self.decoder(h))
        return out

class FastSHAP(nn.Module):
    def __init__(self, n_items):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_items + 1, fs_hidden),
            nn.ReLU(),
            nn.Linear(fs_hidden, fs_hidden),
            nn.ReLU(),
            nn.Linear(fs_hidden, n_items + 1)
        )

    def forward(self, x):
        return self.mlp(x)

def train_dae(dataset):
    print("Training DAE...")
    dae = DAE(dataset.n_items).to(device)
    opt = torch.optim.Adam(dae.parameters(), lr=lr)
    for epoch in range(dae_epochs):
        total_loss = 0
        num_batches = math.ceil(dataset.n_users / batch_size)
        for b in range(num_batches):
            start = b * batch_size + 1
            end = min(start + batch_size - 1, dataset.n_users)
            x = dataset.train_mat[start:end + 1]
            out = dae(x)
            observed = x > 0
            loss = F.binary_cross_entropy(out, x, reduction='none') * observed.float()
            loss = loss.sum() / observed.sum() if observed.sum() > 0 else 0
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_batches
        print(f'DAE Epoch {epoch}, Loss {avg_loss:.4f}')
    return dae

def train_fs(dataset, dae):
    print("Training FastSHAP...")
    fs = FastSHAP(dataset.n_items).to(device)
    opt = torch.optim.Adam(fs.parameters(), lr=lr)
    for epoch in range(fs_epochs):
        total_loss = 0
        num_users = 0
        for u in range(1, dataset.n_users + 1):
            r = dataset.train_mat[u]
            N = r.sum().item()
            if N < 2: continue
            num_users += 1
            v_empty = (dae(torch.zeros_like(r)) * r).sum() / N
            v_full = (dae(r) * r).sum() / N
            num_s = 10  # Reduced for speed
            loss_u = 0
            for _ in range(num_s):
                k = np.random.randint(1, int(N))
                observed_indices = torch.where(r > 0)[0]
                perm = torch.randperm(int(N))
                s_indices = observed_indices[perm[:k]]
                mask = torch.zeros_like(r)
                mask[s_indices] = 1
                r_s = r * mask
                v_s = (dae(r_s) * r).sum() / N
                sum_phi_s = (fs(r) * mask).sum()
                l = (v_s - v_empty - sum_phi_s) ** 2
                loss_u += l
            loss_u /= num_s
            opt.zero_grad()
            loss_u.backward()
            opt.step()
            total_loss += loss_u.item()
        avg_loss = total_loss / num_users if num_users > 0 else 0
        print(f'FS Epoch {epoch}, Loss {avg_loss:.4f}')
    return fs

def compute_weights(dataset, fs):
    print("Computing Shapley weights...")
    weights = {}
    with torch.no_grad():
        for u in range(1, dataset.n_users + 1):
            r = dataset.train_mat[u]
            phi = fs(r) * r
            observed_indices = torch.where(r > 0)[0]
            for j in observed_indices:
                i = j.item()
                w = phi[j].item()
                weights[(u, i)] = max(w, 0) + 1
    print(f"Computed weights for {len(weights)} user-item pairs")
    return weights

def update_graph(dataset, weights):
    print("Updating graph with Shapley weights...")
    ui_edges = []
    values_ui = []
    for row in dataset.train_ratings.itertuples():
        u = row.user
        i = dataset.n_users + row.item
        w = weights.get((u, row.item), 1)
        ui_edges.append([u, i])
        ui_edges.append([i, u])
        values_ui.append(w)
        values_ui.append(w)
    ig_edges = []
    values_ig = []
    if not dataset.is_baseline:
        for i in range(1, dataset.n_items + 1):
            item_node = dataset.n_users + i
            for g in dataset.item_genres[i]:
                genre_node = dataset.n_users + dataset.n_items + g + 1
                ig_edges.append([item_node, genre_node])
                ig_edges.append([genre_node, item_node])
                values_ig.append(1.0)
                values_ig.append(1.0)
    all_edges = ui_edges + ig_edges
    all_values = values_ui + values_ig
    indices = torch.LongTensor(np.array(all_edges).T).to(device)
    values = torch.FloatTensor(all_values).to(device)
    dataset.adj = torch.sparse.FloatTensor(indices, values, 
                                         (dataset.total_nodes + 1, dataset.total_nodes + 1)).to(device)
    dataset.update_norm_adj()

class DyHuCoG(nn.Module):
    def __init__(self, dataset, latent_dim, n_layers):
        super().__init__()
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(dataset.total_nodes + 1, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self):
        emb = self.embedding.weight
        all_embs = [emb]
        for l in range(self.n_layers):
            emb = torch.sparse.mm(self.dataset.norm_adj, emb)
            all_embs.append(emb)
        final_emb = torch.mean(torch.stack(all_embs), dim=0)
        return final_emb

    def get_rating(self, users, items):
        final_emb = self.forward()
        user_emb = final_emb[users]
        item_emb = final_emb[self.dataset.n_users + items]
        rating = (user_emb * item_emb).sum(dim=1)
        return rating

def bpr_loss(model, users, pos_items, neg_items):
    final_emb = model.forward()
    users_emb = final_emb[users]
    pos_emb = final_emb[model.dataset.n_users + pos_items]
    neg_emb = final_emb[model.dataset.n_users + neg_items]
    pos_score = (users_emb * pos_emb).sum(dim=1)
    neg_score = (users_emb * neg_emb).sum(dim=1)
    loss = - F.logsigmoid(pos_score - neg_score).mean()
    reg = (model.embedding.weight[users].norm(2).pow(2) +
           model.embedding.weight[model.dataset.n_users + pos_items].norm(2).pow(2) +
           model.embedding.weight[model.dataset.n_users + neg_items].norm(2).pow(2)) / len(users) * decay
    return loss + reg

class BPRData(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.features = self.dataset.train_ratings[['user', 'item']].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        u = self.features[idx][0]
        pos = self.features[idx][1]
        neg = np.random.randint(1, self.dataset.n_items + 1)
        while self.dataset.train_mat[u, neg] > 0:
            neg = np.random.randint(1, self.dataset.n_items + 1)
        return u, pos, neg

def compute_ndcg(rec_items, test_items, k):
    dcg = 0
    for i, item in enumerate(rec_items[:k]):
        if item in test_items:
            dcg += 1 / math.log2(i + 2)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(test_items), k)))
    return dcg / idcg if idcg > 0 else 0

def evaluate(model, dataset, users_list, ks, prefix=''):
    model.eval()
    metrics = {k: {'prec': 0, 'recall': 0, 'ndcg': 0, 'hr': 0} for k in ks}
    num_users = 0
    with torch.no_grad():
        for u in users_list:
            test_items = dataset.test_set[u]
            if len(test_items) == 0: continue
            num_users += 1
            items = torch.tensor(range(1, dataset.n_items + 1), device=device)
            ratings = model.get_rating(torch.full_like(items, u, dtype=torch.long, device=device), items)
            trained = torch.nonzero(dataset.train_mat[u] > 0).squeeze()
            if trained.numel() > 0:
                ratings[trained - 1] = -float('inf')
            _, indices = torch.topk(ratings, max(ks))
            rec_items = (indices.cpu().numpy() + 1).tolist()
            for k in ks:
                rec_k = set(rec_items[:k])
                hits = len(rec_k & set(test_items))
                metrics[k]['prec'] += hits / k
                metrics[k]['recall'] += hits / len(test_items)
                metrics[k]['ndcg'] += compute_ndcg(rec_items, test_items, k)
                metrics[k]['hr'] += 1 if hits > 0 else 0
    for k in ks:
        for m in metrics[k]:
            metrics[k][m] /= num_users
    print(f'{prefix} Evaluation:')
    print('K\tPrec\tRecall\tNDCG\tHR')
    for k in ks:
        print(f'{k}\t{metrics[k]["prec"]:.4f}\t{metrics[k]["recall"]:.4f}\t{metrics[k]["ndcg"]:.4f}\t{metrics[k]["hr"]:.4f}')
    return metrics

def train(model, dataset, model_name):
    print(f"Training {model_name}...")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = BPRData(dataset)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            u, pos, neg = [x.to(device) for x in batch]
            l = bpr_loss(model, u, pos, neg)
            opt.zero_grad()
            l.backward()
            opt.step()
            total_loss += l.item()
        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0:
            print(f'{model_name} Epoch {epoch}, Loss {avg_loss:.4f}')
    final_metrics = evaluate(model, dataset, range(1, dataset.n_users + 1), ks, prefix=f'{model_name}')
    return final_metrics

if __name__ == '__main__':
    print("=== DyHuCoG vs LightGCN Comparison ===")
    
    # Baseline: LightGCN
    print("\n1. Running LightGCN Baseline")
    dataset_baseline = RecommenderDataset(data_path, is_baseline=True)
    model_baseline = DyHuCoG(dataset_baseline, latent_dim, n_layers).to(device)
    results_baseline = train(model_baseline, dataset_baseline, 'LightGCN')

    # DyHuCoG
    print("\n2. Running DyHuCoG")
    dataset = RecommenderDataset(data_path, is_baseline=False)
    dae = train_dae(dataset)
    fs = train_fs(dataset, dae)
    weights = compute_weights(dataset, fs)
    update_graph(dataset, weights)
    model = DyHuCoG(dataset, latent_dim, n_layers).to(device)
    results_dyhucog = train(model, dataset, 'DyHuCoG')

    # Compare results
    print("\n=== FINAL COMPARISON ===")
    print("Model\t\tK\tPrec\tRecall\tNDCG\tHR")
    print("-" * 60)
    for k in ks:
        lb = results_baseline[k]
        dy = results_dyhucog[k]
        print(f"LightGCN\t{k}\t{lb['prec']:.4f}\t{lb['recall']:.4f}\t{lb['ndcg']:.4f}\t{lb['hr']:.4f}")
        print(f"DyHuCoG\t\t{k}\t{dy['prec']:.4f}\t{dy['recall']:.4f}\t{dy['ndcg']:.4f}\t{dy['hr']:.4f}")
        print(f"Improvement\t{k}\t{(dy['prec']/lb['prec']-1)*100:+.1f}%\t{(dy['recall']/lb['recall']-1)*100:+.1f}%\t{(dy['ndcg']/lb['ndcg']-1)*100:+.1f}%\t{(dy['hr']/lb['hr']-1)*100:+.1f}%")
        print("-" * 60)