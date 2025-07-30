#!/usr/bin/env python
"""
Fast training script for DyHuCoG - optimized for quick results
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import main, parse_args, load_config

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Override with fast settings
    if args.config is None:
        args.config = 'config/colab_config.yaml'
    
    # Load and modify config for fast training
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Fast settings
    config['model']['dae_epochs'] = 3
    config['model']['shapley_epochs'] = 3
    config['model']['n_shapley_samples'] = 2
    config['training']['batch_size'] = 64
    config['training']['epochs'] = 10
    config['training']['eval_every'] = 5
    
    # Save modified config
    fast_config_path = Path('config/fast_config.yaml')
    with open(fast_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Update args
    args.config = str(fast_config_path)
    
    # Run training
    print("Running DyHuCoG with optimized settings for fast training...")
    print(f"DAE epochs: {config['model']['dae_epochs']}")
    print(f"Shapley epochs: {config['model']['shapley_epochs']}")
    print(f"Shapley samples: {config['model']['n_shapley_samples']}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    # Call main training function
    main()