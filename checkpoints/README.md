# Checkpoints Directory

This directory stores model checkpoints from training experiments.

## Directory Structure

```
checkpoints/
├── dyhucog_ml100k_20240120_143022/   # Example experiment
│   ├── best_model.pth                 # Best model based on validation
│   ├── checkpoint_epoch_50.pth        # Periodic checkpoint
│   └── checkpoint_epoch_100.pth       # Periodic checkpoint
├── baselines/                         # Baseline model checkpoints
│   ├── lightgcn_run1/
│   ├── ngcf_run1/
│   └── ...
└── ablation/                          # Ablation study checkpoints
    ├── ablation_NoShapley/
    ├── ablation_NoGenre/
    └── ...
```

## Checkpoint Format

Each checkpoint file contains:
```python
{
    'epoch': int,                      # Training epoch
    'model_state_dict': OrderedDict,   # Model parameters
    'optimizer_state_dict': dict,      # Optimizer state
    'scheduler_state_dict': dict,      # LR scheduler state
    'best_val_metric': float,          # Best validation NDCG@10
    'best_metrics': dict,              # All metrics at best epoch
    'training_history': dict,          # Complete training history
    'config': dict,                    # Model configuration
    'n_users': int,                    # Dataset dimensions
    'n_items': int,
    'n_genres': int,
    'edge_weights': dict               # For DyHuCoG: Shapley weights
}
```

## Loading Checkpoints

### Load for evaluation:
```python
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Resume training:
```python
python scripts/train.py --checkpoint checkpoints/best_model.pth
```

### Use in evaluation:
```python
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

## Pre-trained Models

Download pre-trained models:

| Dataset | Model | Link | NDCG@10 |
|---------|-------|------|---------|
| ML-100K | DyHuCoG | [Download](#) | 0.1023 |
| ML-100K | LightGCN | [Download](#) | 0.0923 |
| ML-100K | NGCF | [Download](#) | 0.0945 |
| ML-1M | DyHuCoG | [Download](#) | 0.1156 |

Place downloaded models in this directory.

## Best Practices

1. **Naming Convention**: 
   - Format: `{model}_{dataset}_{timestamp}/`
   - Example: `dyhucog_ml100k_20240120_143022/`

2. **Storage Management**:
   - Keep only best models to save space
   - Periodic checkpoints can be deleted after training
   - Use `--save_best` flag to save only best model

3. **Reproducibility**:
   - Each checkpoint includes the full configuration
   - Random seeds are saved for exact reproduction

## Disk Space

Approximate checkpoint sizes:
- DyHuCoG: ~50MB (includes edge weights)
- LightGCN: ~20MB
- NGCF: ~30MB

Consider using compression for long-term storage:
```bash
tar -czf checkpoint.tar.gz checkpoint_dir/
```