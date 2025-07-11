import torch, numpy as np
from sklearn.metrics import ndcg_score

def ndcg_at_k(y_true, y_score, k=10):
    return ndcg_score([y_true], [y_score], k=k)

def hit_rate_at_k(y_true, y_score, k=10):
    # 1 if at least one positive item in top-k
    topk_idx = np.argsort(y_score)[::-1][:k]
    return int(any([y_true[i] > 0 for i in topk_idx]))

def intra_list_diversity(item_embs):
    if len(item_embs) < 2:
        return 0.0
    sims = np.matmul(item_embs, item_embs.T)
    tril = sims[np.tril_indices_from(sims, k=-1)]
    return 1 - np.mean(tril)

def catalog_coverage(recommended_items, total_items):
    return len(set(recommended_items)) / float(total_items)
