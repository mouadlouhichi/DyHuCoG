#!/usr/bin/env python
"""
Run baseline experiments comparing DyHuCoG with LightGCN and NGCF
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import RecommenderDataset
from src.models.dyhucog import DyHuCoG
from src.models.lightgcn import LightGCN
from src.models.ngcf import NGCF
from src.utils.trainer import Trainer
from src.utils.logger import setup_logger, ExperimentLogger
from src.utils.graph_builder import GraphBuilder
from config.model_config import get_model_config


def parse_args():
    parser = argparse.ArgumentParser(description='Run baseline experiments')
    
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset name')
    parser.add_argument('--models', nargs='+', 
                       default=['lightgcn', 'ngcf', 'dyhucog'],
                       help='Models to compare')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of runs for statistical significance')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file')
    parser.add_argument('--output_dir', type=str, default='results/baselines',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    
    return parser.parse_args()


def run_single_experiment(model_name: str, dataset: RecommenderDataset,
                         config: dict, device: torch.device, 
                         run_id: int, logger) -> dict:
    """Run a single experiment for a model
    
    Args:
        model_name: Name of the model
        dataset: Dataset object
        config: Configuration dictionary
        device: Torch device
        run_id: Run identifier
        logger: Logger instance
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\nStarting {model_name} - Run {run_id}")
    
    # Create model
    if model_name == 'dyhucog':
        model = DyHuCoG(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            n_genres=dataset.n_genres,
            config=config['model']
        )
    elif model_name == 'lightgcn':
        model = LightGCN(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            latent_dim=config['model']['latent_dim'],
            n_layers=config['model']['n_layers'],
            dropout=0.0
        )
    elif model_name == 'ngcf':
        model = NGCF(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            latent_dim=config['model']['latent_dim'],
            n_layers=config['model']['n_layers'],
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    model = model.to(device)
    
    # Build graph and prepare model
    if model_name == 'dyhucog':
        # For DyHuCoG, we need to compute edge weights and build hypergraph
        logger.info("Computing Shapley weights for DyHuCoG...")
        edge_weights = model.compute_shapley_weights(dataset.train_mat.to(device))
        
        # Build hypergraph with edge weights
        edge_index, edge_weight = GraphBuilder.get_edge_list(dataset, edge_weights)
        model.build_hypergraph(edge_index.to(device), edge_weight.to(device), dataset.item_genres)
        
        # For DyHuCoG, adj is None in trainer
        adj = None
    else:
        # For baseline models, build standard graph
        adj, n_nodes = GraphBuilder.build_user_item_graph(dataset)
        adj = GraphBuilder.normalize_adj(adj).to(device)
    
    # Create output directories
    checkpoint_dir = Path(config['experiment']['checkpoint_dir']) / f"{model_name}_run{run_id}"
    log_dir = Path(config['experiment']['log_dir']) / f"{model_name}_run{run_id}"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        adj=adj
    )
    
    # Train model
    history = trainer.train()
    
    # Get best results
    best_metrics = trainer.best_metrics
    
    # Evaluate on different user groups
    from src.utils.evaluator import Evaluator
    evaluator = Evaluator(
        model=model,
        dataset=dataset,
        device=device,
        adj=adj
    )
    
    cold_results = evaluator.evaluate_cold_start(config['evaluation']['k_values'])
    
    return {
        'model': model_name,
        'run_id': run_id,
        'history': history,
        'best_metrics': best_metrics,
        'cold_results': cold_results,
        'training_time': sum(history['epoch_times'])
    }


def aggregate_results(all_results: dict) -> dict:
    """Aggregate results across runs
    
    Args:
        all_results: Dictionary mapping model names to list of run results
        
    Returns:
        Aggregated results dictionary
    """
    aggregated = {}
    
    for model_name, runs in all_results.items():
        # Extract metrics from all runs
        val_ndcg = [run['best_metrics']['val_metrics'][10]['ndcg'] for run in runs]
        val_hr = [run['best_metrics']['val_metrics'][10]['hit_rate'] for run in runs]
        test_ndcg = [run['best_metrics']['test_metrics'][10]['ndcg'] for run in runs]
        test_hr = [run['best_metrics']['test_metrics'][10]['hit_rate'] for run in runs]
        
        # Cold-start metrics
        cold_ndcg = [run['cold_results']['cold']['metrics'][10]['ndcg'] for run in runs]
        cold_hr = [run['cold_results']['cold']['metrics'][10]['hit_rate'] for run in runs]
        
        # Training time
        train_times = [run['training_time'] for run in runs]
        
        aggregated[model_name] = {
            'val_ndcg': {
                'mean': np.mean(val_ndcg),
                'std': np.std(val_ndcg),
                'values': val_ndcg
            },
            'val_hr': {
                'mean': np.mean(val_hr),
                'std': np.std(val_hr),
                'values': val_hr
            },
            'test_ndcg': {
                'mean': np.mean(test_ndcg),
                'std': np.std(test_ndcg),
                'values': test_ndcg
            },
            'test_hr': {
                'mean': np.mean(test_hr),
                'std': np.std(test_hr),
                'values': test_hr
            },
            'cold_ndcg': {
                'mean': np.mean(cold_ndcg),
                'std': np.std(cold_ndcg),
                'values': cold_ndcg
            },
            'cold_hr': {
                'mean': np.mean(cold_hr),
                'std': np.std(cold_hr),
                'values': cold_hr
            },
            'training_time': {
                'mean': np.mean(train_times),
                'std': np.std(train_times),
                'total': sum(train_times)
            }
        }
        
    return aggregated


def create_results_table(aggregated_results: dict) -> pd.DataFrame:
    """Create results table for paper
    
    Args:
        aggregated_results: Aggregated results dictionary
        
    Returns:
        Pandas DataFrame with formatted results
    """
    rows = []
    
    for model_name, metrics in aggregated_results.items():
        row = {
            'Model': model_name.upper(),
            'Val NDCG@10': f"{metrics['val_ndcg']['mean']:.4f} ± {metrics['val_ndcg']['std']:.4f}",
            'Val HR@10': f"{metrics['val_hr']['mean']:.4f} ± {metrics['val_hr']['std']:.4f}",
            'Test NDCG@10': f"{metrics['test_ndcg']['mean']:.4f} ± {metrics['test_ndcg']['std']:.4f}",
            'Test HR@10': f"{metrics['test_hr']['mean']:.4f} ± {metrics['test_hr']['std']:.4f}",
            'Cold NDCG@10': f"{metrics['cold_ndcg']['mean']:.4f} ± {metrics['cold_ndcg']['std']:.4f}",
            'Cold HR@10': f"{metrics['cold_hr']['mean']:.4f} ± {metrics['cold_hr']['std']:.4f}",
            'Time (s)': f"{metrics['training_time']['mean']:.1f}"
        }
        rows.append(row)
        
    return pd.DataFrame(rows)


def save_latex_table(df: pd.DataFrame, output_path: Path):
    """Save results as LaTeX table
    
    Args:
        df: Results DataFrame
        output_path: Output file path
    """
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'c' * (len(df.columns) - 1),
        caption="Performance comparison on MovieLens dataset",
        label="tab:baseline_results"
    )
    
    with open(output_path, 'w') as f:
        f.write(latex)


def main():
    args = parse_args()
    
    # Set up experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=f"baselines_{args.dataset}",
        log_dir=Path(args.output_dir)
    )
    
    # Load configuration
    config = get_model_config(args.config).to_dict()
    config['dataset']['name'] = args.dataset
    
    # Fix num_workers for Colab
    config['training']['num_workers'] = 0
    
    # Override device if specified
    if args.device != 'auto':
        config['training']['device'] = args.device
        
    exp_logger.log_config(config)
    
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
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size']
    )
    
    # Log dataset statistics
    stats = dataset.get_statistics()
    exp_logger.log_metrics(stats, step=0, prefix="dataset")
    
    # Run experiments
    all_results = {model: [] for model in args.models}
    
    for model_name in args.models:
        exp_logger.logger.info(f"\n{'='*50}")
        exp_logger.logger.info(f"Running {model_name.upper()} experiments")
        exp_logger.logger.info(f"{'='*50}")
        
        # Update model-specific config
        model_config = config.copy()
        model_config['model']['name'] = model_name
        
        # Run multiple times
        for run_id in range(1, args.n_runs + 1):
            # Set seed for reproducibility
            seed = config['experiment']['seed'] + run_id
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Run experiment
            results = run_single_experiment(
                model_name=model_name,
                dataset=dataset,
                config=model_config,
                device=device,
                run_id=run_id,
                logger=exp_logger.logger
            )
            
            all_results[model_name].append(results)
            
            # Log run results
            exp_logger.log_metrics(
                {
                    'val_ndcg': results['best_metrics']['val_metrics'][10]['ndcg'],
                    'test_ndcg': results['best_metrics']['test_metrics'][10]['ndcg'],
                    'cold_ndcg': results['cold_results']['cold']['metrics'][10]['ndcg'],
                    'training_time': results['training_time']
                },
                step=run_id,
                prefix=model_name
            )
    
    # Aggregate results
    exp_logger.logger.info("\nAggregating results...")
    aggregated_results = aggregate_results(all_results)
    
    # Create results table
    results_df = create_results_table(aggregated_results)
    exp_logger.logger.info("\nFinal Results:")
    exp_logger.logger.info(results_df.to_string())
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    with open(output_dir / 'raw_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
        
    # Save aggregated results
    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated_results, f, indent=2)
        
    # Save tables
    results_df.to_csv(output_dir / 'results_table.csv', index=False)
    save_latex_table(results_df, output_dir / 'results_table.tex')
    
    # Log best results
    exp_logger.log_best_results(aggregated_results)
    exp_logger.save_results()
    
    exp_logger.logger.info(f"\nAll results saved to {output_dir}")
    exp_logger.logger.info("Baseline experiments completed!")


if __name__ == '__main__':
    main()