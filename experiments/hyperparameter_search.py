#!/usr/bin/env python
"""
Hyperparameter search for DyHuCoG model
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import itertools
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import RecommenderDataset
from src.models.dyhucog import DyHuCoG
from src.utils.trainer import Trainer
from src.utils.logger import setup_logger, ExperimentLogger
from src.utils.graph_builder import GraphBuilder
from config.model_config import get_model_config


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset name')
    parser.add_argument('--search_type', type=str, default='grid',
                       choices=['grid', 'random', 'bayesian'],
                       help='Search type')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of trials for random/bayesian search')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Base configuration file')
    parser.add_argument('--output_dir', type=str, default='results/hyperparam_search',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Use early stopping based on validation performance')
    
    return parser.parse_args()


# Define hyperparameter search space
SEARCH_SPACE = {
    'latent_dim': [32, 64, 128],
    'n_layers': [2, 3, 4],
    'dropout': [0.0, 0.1, 0.2],
    'lr': [0.0001, 0.0005, 0.001, 0.005],
    'batch_size': [128, 256, 512],
    'dae_hidden': [64, 128, 256],
    'shapley_hidden': [64, 128, 256],
    'n_shapley_samples': [5, 10, 20]
}


def sample_hyperparameters(search_space: Dict[str, List[Any]], 
                          search_type: str = 'random') -> Dict[str, Any]:
    """Sample hyperparameters from search space
    
    Args:
        search_space: Dictionary defining search space
        search_type: Type of search ('random' or 'grid')
        
    Returns:
        Sampled hyperparameters
    """
    if search_type == 'random':
        params = {}
        for param, values in search_space.items():
            params[param] = np.random.choice(values)
        return params
    else:
        # For grid search, this will be handled differently
        raise NotImplementedError("Use generate_grid_search for grid search")


def generate_grid_search(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations for grid search
    
    Args:
        search_space: Dictionary defining search space
        
    Returns:
        List of parameter combinations
    """
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    combinations = []
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        combinations.append(params)
        
    return combinations


def evaluate_hyperparameters(params: Dict[str, Any], dataset: RecommenderDataset,
                            base_config: dict, device: torch.device, 
                            trial_id: int, logger) -> Dict[str, float]:
    """Evaluate a set of hyperparameters
    
    Args:
        params: Hyperparameter values
        dataset: Dataset object
        base_config: Base configuration
        device: Torch device
        trial_id: Trial identifier
        logger: Logger instance
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"\nTrial {trial_id}: {params}")
    
    # Update configuration with hyperparameters
    config = base_config.copy()
    config['model']['latent_dim'] = params['latent_dim']
    config['model']['n_layers'] = params['n_layers']
    config['model']['dropout'] = params['dropout']
    config['model']['dae_hidden'] = params['dae_hidden']
    config['model']['shapley_hidden'] = params['shapley_hidden']
    config['model']['n_shapley_samples'] = params['n_shapley_samples']
    config['training']['learning_rate'] = params['lr']
    config['training']['batch_size'] = params['batch_size']
    
    # Reduce epochs for hyperparameter search
    config['training']['epochs'] = 50
    config['training']['early_stopping_patience'] = 10
    
    # Create model
    model = DyHuCoG(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_genres=dataset.n_genres,
        config=config['model']
    ).to(device)
    
    # Create output directories
    checkpoint_dir = Path(config['experiment']['checkpoint_dir']) / f"hyperparam_trial_{trial_id}"
    log_dir = Path(config['experiment']['log_dir']) / f"hyperparam_trial_{trial_id}"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Train model
    try:
        history = trainer.train()
        
        # Get validation performance
        best_metrics = trainer.best_metrics
        val_ndcg = best_metrics['val_metrics'][10]['ndcg']
        val_hr = best_metrics['val_metrics'][10]['hit_rate']
        
        # Also get test performance for reference
        test_ndcg = best_metrics['test_metrics'][10]['ndcg']
        test_hr = best_metrics['test_metrics'][10]['hit_rate']
        
        results = {
            'val_ndcg': val_ndcg,
            'val_hr': val_hr,
            'test_ndcg': test_ndcg,
            'test_hr': test_hr,
            'training_time': sum(history['epoch_times']),
            'best_epoch': best_metrics['epoch']
        }
        
    except Exception as e:
        logger.error(f"Trial {trial_id} failed: {e}")
        results = {
            'val_ndcg': 0.0,
            'val_hr': 0.0,
            'test_ndcg': 0.0,
            'test_hr': 0.0,
            'training_time': 0.0,
            'best_epoch': 0
        }
    
    return results


def bayesian_optimization(search_space: Dict[str, List[Any]], 
                         dataset: RecommenderDataset,
                         base_config: dict, device: torch.device,
                         n_trials: int, logger):
    """Perform Bayesian optimization
    
    Note: This is a placeholder. For actual implementation,
    use libraries like Optuna or scikit-optimize
    """
    try:
        import optuna
        
        def objective(trial):
            # Sample hyperparameters
            params = {
                'latent_dim': trial.suggest_categorical('latent_dim', search_space['latent_dim']),
                'n_layers': trial.suggest_categorical('n_layers', search_space['n_layers']),
                'dropout': trial.suggest_categorical('dropout', search_space['dropout']),
                'lr': trial.suggest_categorical('lr', search_space['lr']),
                'batch_size': trial.suggest_categorical('batch_size', search_space['batch_size']),
                'dae_hidden': trial.suggest_categorical('dae_hidden', search_space['dae_hidden']),
                'shapley_hidden': trial.suggest_categorical('shapley_hidden', search_space['shapley_hidden']),
                'n_shapley_samples': trial.suggest_categorical('n_shapley_samples', search_space['n_shapley_samples'])
            }
            
            # Evaluate
            results = evaluate_hyperparameters(
                params, dataset, base_config, device, 
                trial.number, logger
            )
            
            return results['val_ndcg']
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Return results
        return [{
            'params': trial.params,
            'results': {
                'val_ndcg': trial.value,
                'val_hr': 0.0,  # Would need to store these separately
                'test_ndcg': 0.0,
                'test_hr': 0.0,
                'training_time': 0.0,
                'best_epoch': 0
            },
            'trial_id': trial.number
        } for trial in study.trials]
        
    except ImportError:
        logger.warning("Optuna not installed. Falling back to random search.")
        return None


def main():
    args = parse_args()
    
    # Set up experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=f"hyperparam_search_{args.dataset}",
        log_dir=Path(args.output_dir)
    )
    
    # Load base configuration
    base_config = get_model_config(args.config).to_dict()
    base_config['dataset']['name'] = args.dataset
    base_config['model']['name'] = 'dyhucog'
    
    exp_logger.log_config(base_config)
    
    # Get device
    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available() 
        else args.device if args.device != 'auto' 
        else 'cpu'
    )
    
    exp_logger.logger.info(f"Using device: {device}")
    
    # Load dataset
    exp_logger.logger.info(f"Loading dataset: {args.dataset}")
    dataset = RecommenderDataset(
        name=args.dataset,
        path='data/',
        test_size=base_config['dataset']['test_size'],
        val_size=base_config['dataset']['val_size']
    )
    
    # Perform hyperparameter search
    all_results = []
    
    if args.search_type == 'grid':
        # Grid search
        exp_logger.logger.info("Performing grid search...")
        param_combinations = generate_grid_search(SEARCH_SPACE)
        exp_logger.logger.info(f"Total combinations: {len(param_combinations)}")
        
        # Limit grid search if too many combinations
        if len(param_combinations) > 100:
            exp_logger.logger.warning(f"Too many combinations ({len(param_combinations)}). "
                                    f"Consider using random search instead.")
            param_combinations = param_combinations[:100]
        
        for trial_id, params in enumerate(param_combinations):
            # Set seed
            torch.manual_seed(base_config['experiment']['seed'] + trial_id)
            np.random.seed(base_config['experiment']['seed'] + trial_id)
            
            # Evaluate
            results = evaluate_hyperparameters(
                params, dataset, base_config, device, trial_id, exp_logger.logger
            )
            
            all_results.append({
                'trial_id': trial_id,
                'params': params,
                'results': results
            })
            
            # Log results
            exp_logger.log_metrics(results, step=trial_id, prefix='trial')
            
    elif args.search_type == 'random':
        # Random search
        exp_logger.logger.info(f"Performing random search with {args.n_trials} trials...")
        
        for trial_id in range(args.n_trials):
            # Sample hyperparameters
            params = sample_hyperparameters(SEARCH_SPACE, 'random')
            
            # Set seed
            torch.manual_seed(base_config['experiment']['seed'] + trial_id)
            np.random.seed(base_config['experiment']['seed'] + trial_id)
            
            # Evaluate
            results = evaluate_hyperparameters(
                params, dataset, base_config, device, trial_id, exp_logger.logger
            )
            
            all_results.append({
                'trial_id': trial_id,
                'params': params,
                'results': results
            })
            
            # Log results
            exp_logger.log_metrics(results, step=trial_id, prefix='trial')
            
    elif args.search_type == 'bayesian':
        # Bayesian optimization
        exp_logger.logger.info(f"Performing Bayesian optimization with {args.n_trials} trials...")
        
        bayesian_results = bayesian_optimization(
            SEARCH_SPACE, dataset, base_config, device, 
            args.n_trials, exp_logger.logger
        )
        
        if bayesian_results is not None:
            all_results = bayesian_results
        else:
            # Fallback to random search
            args.search_type = 'random'
            main()  # Recursive call with random search
            return
    
    # Find best hyperparameters
    best_trial = max(all_results, key=lambda x: x['results']['val_ndcg'])
    
    # Print results
    exp_logger.logger.info("\n" + "="*70)
    exp_logger.logger.info("HYPERPARAMETER SEARCH RESULTS")
    exp_logger.logger.info("="*70)
    
    # Top 10 configurations
    sorted_results = sorted(all_results, 
                          key=lambda x: x['results']['val_ndcg'], 
                          reverse=True)[:10]
    
    exp_logger.logger.info("\nTop 10 configurations:")
    for i, result in enumerate(sorted_results):
        exp_logger.logger.info(f"\n{i+1}. Trial {result['trial_id']}:")
        exp_logger.logger.info(f"   Val NDCG: {result['results']['val_ndcg']:.4f}")
        exp_logger.logger.info(f"   Test NDCG: {result['results']['test_ndcg']:.4f}")
        exp_logger.logger.info(f"   Parameters: {result['params']}")
    
    # Best configuration
    exp_logger.logger.info("\n" + "="*70)
    exp_logger.logger.info("BEST CONFIGURATION")
    exp_logger.logger.info("="*70)
    exp_logger.logger.info(f"Trial ID: {best_trial['trial_id']}")
    exp_logger.logger.info(f"Validation NDCG@10: {best_trial['results']['val_ndcg']:.4f}")
    exp_logger.logger.info(f"Test NDCG@10: {best_trial['results']['test_ndcg']:.4f}")
    exp_logger.logger.info("\nBest hyperparameters:")
    for param, value in best_trial['params'].items():
        exp_logger.logger.info(f"  {param}: {value}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save best configuration
    best_config = base_config.copy()
    for param, value in best_trial['params'].items():
        if param in ['latent_dim', 'n_layers', 'dropout', 'dae_hidden', 
                    'shapley_hidden', 'n_shapley_samples']:
            best_config['model'][param] = value
        elif param == 'lr':
            best_config['training']['learning_rate'] = value
        elif param == 'batch_size':
            best_config['training']['batch_size'] = value
    
    with open(output_dir / 'best_config.yaml', 'w') as f:
        import yaml
        yaml.dump(best_config, f, default_flow_style=False)
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'trial_id': r['trial_id'],
            'val_ndcg': r['results']['val_ndcg'],
            'test_ndcg': r['results']['test_ndcg'],
            **r['params']
        }
        for r in all_results
    ])
    
    results_df.to_csv(output_dir / 'search_results.csv', index=False)
    
    # Create visualizations
    create_hyperparameter_plots(results_df, output_dir)
    
    exp_logger.log_best_results(best_trial)
    exp_logger.save_results()
    
    exp_logger.logger.info(f"\nAll results saved to {output_dir}")
    exp_logger.logger.info("Hyperparameter search completed!")


def create_hyperparameter_plots(results_df: pd.DataFrame, output_dir: Path):
    """Create visualizations for hyperparameter search results
    
    Args:
        results_df: DataFrame with search results
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Parallel coordinates plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize data for plotting
    norm_df = results_df.copy()
    for col in SEARCH_SPACE.keys():
        if col in norm_df.columns:
            norm_df[col] = (norm_df[col] - norm_df[col].min()) / (norm_df[col].max() - norm_df[col].min())
    
    # Sort by validation NDCG
    norm_df = norm_df.sort_values('val_ndcg', ascending=False)
    
    # Plot top 20 configurations
    top_20 = norm_df.head(20)
    
    from pandas.plotting import parallel_coordinates
    parallel_coordinates(top_20, 'val_ndcg', colormap='viridis', alpha=0.7)
    
    plt.title('Hyperparameter Configurations (Top 20)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'parallel_coordinates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Hyperparameter importance
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, param in enumerate(SEARCH_SPACE.keys()):
        if i < len(axes) and param in results_df.columns:
            ax = axes[i]
            
            # Box plot by parameter value
            unique_values = sorted(results_df[param].unique())
            data_by_value = [results_df[results_df[param] == v]['val_ndcg'] for v in unique_values]
            
            ax.boxplot(data_by_value, labels=[str(v) for v in unique_values])
            ax.set_xlabel(param)
            ax.set_ylabel('Val NDCG@10')
            ax.set_title(f'Effect of {param}')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Hyperparameter Effects on Validation Performance')
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select numeric columns
    numeric_cols = ['val_ndcg', 'test_ndcg'] + list(SEARCH_SPACE.keys())
    numeric_cols = [col for col in numeric_cols if col in results_df.columns]
    
    correlation_matrix = results_df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    
    plt.title('Hyperparameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()