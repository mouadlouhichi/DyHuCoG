# DyHuCoG: Dynamic Hybrid Recommender via Graph-based Cooperative Games

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.12%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Paper](https://img.shields.io/badge/paper-CIKM%202024-red)](https://arxiv.org/abs/your_paper_link)

Official PyTorch implementation of **"Dynamic Hybrid Recommender via Graph-based Cooperative Games (DyHuCoG)"**.

<p align="center">
  <img src="results/figures/dyhucog_architecture.png" width="800">
</p>

## 🎯 Highlights

- **🎮 Cooperative Game Theory**: Models user-item interactions as dynamic cooperative games
- **📊 Shapley Value Weighting**: Preference-aware edge weighting using approximate Shapley values
- **🔗 Hypergraph Structure**: User-item-context connections in a unified framework
- **🔍 SHAP Integration**: Full explainability support for recommendations
- **❄️ Cold-Start Robustness**: Superior performance on cold-start scenarios

## 📊 Performance

| Model | Precision@10 | Recall@10 | NDCG@10 | HR@10 | Coverage | Diversity |
|-------|-------------|-----------|---------|--------|----------|-----------|
| LightGCN | 0.0823 | 0.0412 | 0.0923 | 0.0534 | 0.3120 | 0.4230 |
| NGCF | 0.0845 | 0.0423 | 0.0945 | 0.0556 | 0.3250 | 0.4450 |
| **DyHuCoG** | **0.0912** | **0.0456** | **0.1023** | **0.0623** | **0.3870** | **0.5120** |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DyHuCoG.git
cd DyHuCoG

# Create conda environment
conda create -n dyhucog python=3.8
conda activate dyhucog

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Download MovieLens-100K
bash scripts/download_data.sh ml-100k

# Download MovieLens-1M (optional)
bash scripts/download_data.sh ml-1m
```

### Training

```bash
# Train DyHuCoG on MovieLens-100K
python scripts/train.py --dataset ml-100k --model dyhucog

# Train with custom configuration
python scripts/train.py --config config/config.yaml

# Train baselines
python scripts/train.py --dataset ml-100k --model lightgcn
python scripts/train.py --dataset ml-100k --model ngcf
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/dyhucog_best.pth

# Run comprehensive evaluation
python experiments/run_baselines.py --dataset ml-100k --n_runs 5
```

### Explainability Analysis

```bash
# Generate SHAP explanations
python scripts/explain.py --checkpoint checkpoints/dyhucog_best.pth --user_id 123

# Run full explainability analysis
python src/explainability/shap_analyzer.py --checkpoint checkpoints/dyhucog_best.pth
```

## 📁 Project Structure

```
DyHuCoG/
├── config/          # Configuration files
├── src/             # Source code
│   ├── models/      # Model implementations
│   ├── data/        # Data loading and preprocessing
│   ├── utils/       # Training and evaluation utilities
│   └── explainability/  # SHAP and visualization tools
├── scripts/         # Executable scripts
├── experiments/     # Experiment runners
└── results/         # Output directory
```

## 🔧 Configuration

Edit `config/config.yaml` to modify hyperparameters:

```yaml
model:
  name: dyhucog
  latent_dim: 64
  n_layers: 3
  dropout: 0.1

training:
  epochs: 100
  batch_size: 256
  lr: 0.001
  weight_decay: 1e-4

cooperative_game:
  dae_hidden: 128
  dae_epochs: 30
  shapley_hidden: 128
  shapley_epochs: 30
```

## 📈 Reproduce Results

### Full Experiments

```bash
# Run all experiments from the paper
python experiments/run_baselines.py --dataset ml-100k --n_runs 5
python experiments/run_ablation.py --dataset ml-100k
python experiments/run_statistical_tests.py --results_dir results/
```

### Visualization

```bash
# Generate all figures
python src/explainability/visualizer.py --results_dir results/

# Or use the notebook
jupyter notebook notebooks/results_visualization.ipynb
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py::TestDyHuCoG
```

## 📊 Pre-trained Models

Download pre-trained models from [Google Drive](https://drive.google.com/your_link) and place them in `checkpoints/`.

| Dataset | Model | Checkpoint | NDCG@10 |
|---------|-------|------------|---------|
| ML-100K | DyHuCoG | [Download](link) | 0.1023 |
| ML-1M | DyHuCoG | [Download](link) | 0.1156 |

## 📚 Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{dyhucog2024,
  title={Dynamic Hybrid Recommender via Graph-based Cooperative Games},
  author={Your Name and Collaborators},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  year={2024},
  pages={xxxx--xxxx}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the authors of [LightGCN](https://github.com/kuandeng/LightGCN) and [NGCF](https://github.com/huangtinglin/NGCF) for their implementations
- MovieLens dataset from [GroupLens](https://grouplens.org/)

## 📧 Contact

For questions, please open an issue or contact: your.email@example.com