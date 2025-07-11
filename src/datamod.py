"""Data module: loads interaction data and provides PyTorch loaders."""
import torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class InteractionDataset(Dataset):
    def __init__(self, csv_path, implicit=True, min_inter=5):
        df = pd.read_csv(csv_path)
        # simple ID normalisation
        user_map = {u:i for i,u in enumerate(df.user_id.unique())}
        item_map = {it:i for i,it in enumerate(df.item_id.unique())}
        df['user_id'] = df.user_id.map(user_map)
        df['item_id'] = df.item_id.map(item_map)

        # filter sparse
        user_cnt = df.user_id.value_counts()
        item_cnt = df.item_id.value_counts()
        keep_u = user_cnt[user_cnt >= min_inter].index
        keep_i = item_cnt[item_cnt >= min_inter].index
        df = df[df.user_id.isin(keep_u) & df.item_id.isin(keep_i)]

        self.u = torch.tensor(df.user_id.values, dtype=torch.long)
        self.i = torch.tensor(df.item_id.values, dtype=torch.long)
        self.r = torch.ones_like(self.u) if implicit else torch.tensor(df.rating.values, dtype=torch.float32)
        self.t = torch.tensor(df.timestamp.values, dtype=torch.long)

        self.n_users = int(self.u.max()) + 1
        self.n_items = int(self.i.max()) + 1

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx], self.t[idx]

def get_loaders(csv_path, batch_size, implicit=True, min_inter=5):
    ds = InteractionDataset(csv_path, implicit, min_inter)
    idx = np.arange(len(ds)); np.random.shuffle(idx)
    cut = int(0.9 * len(idx))
    train_idx, test_idx = idx[:cut], idx[cut:]
    tr_ds = torch.utils.data.Subset(ds, train_idx)
    te_ds = torch.utils.data.Subset(ds, test_idx)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    te_loader = DataLoader(te_ds, batch_size=batch_size*2, shuffle=False, num_workers=2)
    return tr_loader, te_loader, ds
