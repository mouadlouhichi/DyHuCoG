# DyHuCoG – Dynamic Hybrid Cooperative Graph Recommender

Ready‑to‑run reference implementation used in *“Dynamic Hybrid Recommender via Graph‑based Cooperative Games”* (2025).

## Quick start

```bash
git clone <this‑repo>
cd dyhucog
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_ml20m.sh
```

Training logs and metrics will appear in `logs/`.  
To visualise learning curves:

```bash
python -m src.visualize logs/metrics.csv
```

## Datasets

The repo expects a pre‑processed `ml-20m` CSV with columns  
`user_id,item_id,rating,timestamp`.  
Replace `config.yaml:data_path` with your own path to run other datasets.

## Metrics

* nDCG@K, HR@K computed on a 10% hold‑out split  
* Evaluation runtime logged for scalability reporting  
* Diversity & coverage metrics hooks in `src/metrics.py`

## Licence

MIT.
