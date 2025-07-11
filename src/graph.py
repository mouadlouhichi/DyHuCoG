import torch
import torch_geometric as tg

def build_graph(ds):
    # users 0..U-1, items shift by U
    offset_items = ds.i + ds.n_users
    # build undirected bipartite
    e_src = torch.cat([ds.u, offset_items])
    e_dst = torch.cat([offset_items, ds.u])
    edge_index = torch.stack([e_src, e_dst])
    g = tg.data.Data(edge_index=edge_index, num_nodes=ds.n_users + ds.n_items)
    return g
