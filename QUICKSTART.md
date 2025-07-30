# DyHuCoG Quick Start Guide

This guide will help you get DyHuCoG up and running quickly.

## 🚀 5-Minute Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/DyHuCoG.git
cd DyHuCoG

# Create environment
conda create -n dyhucog python=3.8
conda activate dyhucog

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download MovieLens-100K (smallest dataset for quick testing)
bash scripts/download_data.sh ml-100k
```

### 3. Train DyHuCoG

```bash
# Quick training with reduced epochs for testing
python scripts/train.py --dataset ml-100k --model dyhucog --epochs 20
```

### 4. Evaluate

```bash
# Evaluate the trained model
python scripts/evaluate.py --checkpoint checkpoints/*/best_model.pth
```

### 5. Generate Explanations

```bash
# Explain recommendations for a user
python scripts/explain.py --checkpoint checkpoints/*/best_model.pth --user_id 123
```

## 📊 Expected Results

After ~5 minutes of training on GPU (or ~15 minutes on CPU), you should see:
- NDCG@10: ~0.09-0.10
- HR@10: ~0.05-0.06
- Training visualizations in `results/`

## 🎯 Next Steps

### Full Training

For paper-quality results, train for 100 epochs:

```bash
python scripts/train.py --dataset ml-100k --model dyhucog
```

### Compare with Baselines

```bash
# Train baselines
python scripts/train.py --dataset ml-100k --model lightgcn
python scripts/train.py --dataset ml-100k --model ngcf

# Run comprehensive comparison
python experiments/run_baselines.py --dataset ml-100k --n_runs 5
```

### Hyperparameter Tuning

Edit `config/config.yaml` or use command-line overrides:

```bash
python scripts/train.py --dataset ml-100k --model dyhucog \
  --lr 0.0005 --batch_size 512 --latent_dim 128
```

### Larger Datasets

```bash
# Download MovieLens-1M
bash scripts/download_data.sh ml-1m

# Train on larger dataset
python scripts/train.py --dataset ml-1m --model dyhucog
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 128`
- Reduce embedding dimension: `--latent_dim 32`

### Slow Training
- Use GPU: Ensure CUDA is installed
- Reduce evaluation frequency in config
- Use smaller dataset for testing

### Import Errors
- Ensure you're in the conda environment: `conda activate dyhucog`
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

## 📖 Documentation

- Full documentation: [docs/](docs/)
- API reference: [docs/api.md](docs/api.md)
- Paper reproduction: [docs/reproduction.md](docs/reproduction.md)

## 💬 Getting Help

- Check [FAQ](docs/FAQ.md)
- Open an [issue](https://github.com/yourusername/DyHuCoG/issues)
- Contact: your.email@example.com