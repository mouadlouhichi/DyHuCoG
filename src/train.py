import torch, numpy as np, pandas as pd
from pathlib import Path
from utils import load_cfg, seed_everything, logger, ensure_dir
from datamod import get_loaders
from graph import build_graph
from model import DyHuCoG
from evaluate import evaluate
from tqdm import tqdm

def train(cfg_path: str):
    cfg = load_cfg(cfg_path)
    seed_everything(cfg['seed'])
    tr_loader, te_loader, ds = get_loaders(cfg['data_path'], cfg['batch_size'], cfg['implicit'])
    g = build_graph(ds).to(cfg['device'])
    model = DyHuCoG(ds.n_users, ds.n_items, cfg).to(cfg['device'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    ensure_dir('logs')
    metrics_csv = Path('logs/metrics.csv')
    if metrics_csv.exists():
        metrics_csv.unlink()

    for epoch in range(cfg['epochs']):
        model.train()
        losses = []
        for u,i,r,t in tqdm(tr_loader, desc=f'Epoch {epoch}'):
            u,i,r = u.to(cfg['device']), i.to(cfg['device']), r.to(cfg['device'])
            pred = model(u,i,g)
            loss = -(r*torch.log(pred + 1e-8) + (1-r)*torch.log(1-pred + 1e-8)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        logger.info(f'Epoch {epoch} | Loss {np.mean(losses):.4f}')
        if (epoch+1) % cfg['eval_every'] == 0:
            evaluate(model, g, te_loader, cfg, epoch, metrics_csv)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config.yaml')
    args = parser.parse_args()
    train(args.cfg)
