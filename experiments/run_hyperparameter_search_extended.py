#!/usr/bin/env python
"""
Extended hyperparameter search with fair comparison across all models
"""

import os
import sys
from pathlib import Path
import optuna
import torch
import numpy as np
import json
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from experiments.run_comprehensive_evaluation import ComprehensiveEvaluator
from src.utils.logger import setup_logger


def objective_factory(model_name: str, dataset_name: str, 
                     evaluator: ComprehensiveEvaluator, device: torch.device):
    """Create objective function for Optuna"""
    
    def objective(trial):
        # Common hyperparameters
        config = {
            'latent_dim': trial.suggest_categorical('latent_dim', [32, 64, 128]),
            'n_layers': trial.suggest_int('n_layers', 1, 4),
            'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            'dropout': trial.suggest_uniform('dropout', 0.0, 0.3)
        }
        
        # Model-specific hyperparameters
        if model_name == 'dyhucog':
            config['dae_hidden'] = trial.suggest_categorical('dae_hidden', [64, 128, 256])
            config['shapley_hidden'] = trial.suggest_categorical('shapley_hidden', [64, 128, 256])
            config['n_shapley_samples'] = trial.suggest_int('n_shapley_samples', 5, 20)
        elif model_name in ['sgl', 'simgcl']:
            config['ssl_temp'] = trial.suggest_uniform('ssl_temp', 0.1, 0.5)
            config['ssl_reg'] = trial.suggest_uniform('ssl_reg', 0.05, 0.2)
        elif model_name == 'hccf':
            config['n_hyperedges'] = trial.suggest_int('n_hyperedges', 500, 2000)
            
        # Update evaluator config
        eval_config = evaluator.config.copy()
        eval_config['model'].update(config)
        eval_config['training']['learning_rate'] = config['lr']
        eval_config['training']['batch_size'] = config['batch_size']
        eval_config['training']['epochs'] = 50  # Reduced for search
        
        evaluator.config = eval_config
        
        # Load dataset
        dataset = evaluator.load_dataset(dataset_name)
        
        # Run single evaluation
        results = evaluator.evaluate_single_run(
            model_name, dataset_name, dataset, device, 0
        )
        
        # Return validation NDCG@10
        return results['metrics'][10]['ndcg']
    
    return objective


def run_hyperparameter_search(models: list, datasets: list, 
                            n_trials: int = 50, output_dir: Path = None):
    """Run hyperparameter search for all models"""
    
    # Setup
    logger = setup_logger('hyperparam_search')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base config
    with open('config/config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    with open('config/datasets_config.yaml', 'r') as f:
        datasets_config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        base_config, datasets_config, output_dir, logger
    )
    
    # Results storage
    best_params = {}
    
    for dataset in datasets:
        best_params[dataset] = {}
        
        for model in models:
            logger.info(f"Searching hyperparameters for {model} on {dataset}")
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Optimize
            objective = objective_factory(model, dataset, evaluator, device)
            study.optimize(objective, n_trials=n_trials)
            
            # Store results
            best_params[dataset][model] = {
                'params': study.best_params,
                'value': study.best_value,
                'n_trials': len(study.trials)
            }
            
            logger.info(f"Best params for {model}: {study.best_params}")
            logger.info(f"Best value: {study.best_value:.4f}")
    
    # Save results
    with open(output_dir / 'best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    return best_params


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', 
                       default=['lightgcn', 'ngcf', 'sgl', 'dyhucog'])
    parser.add_argument('--datasets', nargs='+', 
                       default=['ml-100k', 'ml-1m'])
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='results/hyperparam_search')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_hyperparameter_search(
        args.models, args.datasets, args.n_trials, output_dir
    )