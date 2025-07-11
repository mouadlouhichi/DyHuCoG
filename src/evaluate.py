import torch, numpy as np, pandas as pd, time
from metrics import ndcg_at_k, hit_rate_at_k, catalog_coverage
from tqdm import tqdm

def evaluate(model, g, loader, cfg, epoch, out_csv):
    model.eval()
    ndcgs, hits = [], []
    rec_items = []
    start = time.time()
    with torch.no_grad():
        for u,i,r,t in tqdm(loader, desc='Eval'):
            u = u.to(cfg['device'])
            i = i.to(cfg['device'])
            scores = model(u,i,g).cpu().numpy()
            labels = r.numpy()
            for lbl, scr in zip(labels, scores):
                ndcgs.append(ndcg_at_k([lbl], [scr], cfg['top_k']))
                hits.append(hit_rate_at_k([lbl], [scr], cfg['top_k']))
        runtime = time.time() - start
    metrics = {
        'epoch': epoch,
        'nDCG@{}'.format(cfg['top_k']): float(np.mean(ndcgs)),
        'HR@{}'.format(cfg['top_k']): float(np.mean(hits)),
        'eval_time_sec': runtime
    }
    print(metrics)
    # append to CSV
    df = pd.DataFrame([metrics])
    if not out_csv.exists():
        df.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, mode='a', header=False, index=False)
