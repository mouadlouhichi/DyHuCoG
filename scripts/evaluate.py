#!/usr/bin/env python
"""
Evaluation script for trained models
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import RecommenderDataset
from src.models.dyhucog import DyHuCoG
from src.models.lightgcn import LightGCN
from src.models.ngcf import NGCF
from src.utils.evaluator import Evaluator
from src.utils.logger import setup_logger
from src.utils.graph_builder import GraphBuilder


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (override checkpoint config)')
    parser.add_argument('--split', type=str, default='test',
                       help='Split to evaluate: val, test')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Evaluation batch size')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20],
                       help='K values for metrics')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--user_groups', action='store_true',
                       help='Evaluate on user groups (cold/warm/hot)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    config = checkpoint.get('config', {})
    model_name = config['model']['name']
    
    # Reconstruct model
    if model_name == 'dyhucog':
        model = DyHuCoG(
            n_users=checkpoint['n_users'],
            n_items=checkpoint['n_items'],
            n_genres=checkpoint['n_genres'],
            config=config['model']
        )
    elif model_name == 'lightgcn':
        model = LightGCN(
            n_users=checkpoint['n_users'],
            n_items=checkpoint['n_items'],
            latent_dim=config['model']['latent_dim'],
            n_layers=config['model']['n_layers'],
            dropout=config['model']['dropout']
        )
    elif model_name == 'ngcf':
        model = NGCF(
            n_users=checkpoint['n_users'],
            n_items=checkpoint['n_items'],
            latent_dim=config['model']['latent_dim'],
            n_layers=config['model']['n_layers'],
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def evaluate_user_groups(evaluator: Evaluator, dataset: RecommenderDataset, 
                        k_values: list, logger):
    """Evaluate on different user groups"""
    results = {}
    
    # Define user groups
    user_groups = {
        'all': list(range(1, dataset.n_users + 1)),
        'cold': dataset.cold_users,
        'warm': dataset.warm_users,
        'hot': dataset.hot_users
    }
    
    # Evaluate each group
    for group_name, user_list in user_groups.items():
        logger.info(f"Evaluating {group_name} users ({len(user_list)} users)...")
        
        metrics, coverage, diversity = evaluator.evaluate(
            user_list=user_list,
            k_values=k_values
        )
        
        results[group_name] = {
            'metrics': metrics,
            'coverage': coverage,
            'diversity': diversity,
            'n_users': len(user_list)
        }
        
        # Log results
        logger.info(f"{group_name.capitalize()} Users Results:")
        for k in k_values:
            logger.info(f"  @{k}: NDCG={metrics[k]['ndcg']:.4f}, "
                       f"HR={metrics[k]['hit_rate']:.4f}, "
                       f"Precision={metrics[k]['precision']:.4f}, "
                       f"Recall={metrics[k]['recall']:.4f}")
        logger.info(f"  Coverage: {coverage:.4f}, Diversity: {diversity:.4f}")
    
    return results


def save_predictions(evaluator: Evaluator, dataset: RecommenderDataset,
                    output_path: Path, n_users: int = 100, top_k: int = 50):
    """Save top-k predictions for sampled users"""
    predictions = []
    
    # Sample users
    user_sample = np.random.choice(
        range(1, dataset.n_users + 1), 
        min(n_users, dataset.n_users), 
        replace=False
    )
    
    for user_id in user_sample:
        # Get predictions
        scores = evaluator.get_user_predictions(user_id)
        
        # Get top-k items
        top_items = torch.topk(scores, top_k).indices.cpu().numpy() + 1
        
        for rank, item_id in enumerate(top_items):
            predictions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rank': rank + 1,
                'score': scores[item_id - 1].item()
            })
    
    # Save to CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    
    return df


def main():
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        name='evaluate',
        log_file=output_dir / f'evaluate_{timestamp}.log'
    )
    
    logger.info(f"Evaluation script started")
    logger.info(f"Arguments: {args}")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Override dataset if specified
    if args.dataset:
        config['dataset']['name'] = args.dataset
    
    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    dataset = RecommenderDataset(
        name=config['dataset']['name'],
        path=config['dataset']['path'],
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size']
    )
    
    # Build graph if needed
    adj = None
    if config['model']['name'] in ['lightgcn', 'ngcf']:
        logger.info("Building graph structure...")
        adj, _ = GraphBuilder.build_user_item_graph(dataset)
        adj = GraphBuilder.normalize_adj(adj).to(device)
    elif config['model']['name'] == 'dyhucog':
        # For DyHuCoG, we need to rebuild the graph with edge weights
        logger.info("Rebuilding DyHuCoG hypergraph...")
        # This would require loading edge weights from checkpoint
        # For evaluation, we can use the model's internal adj if available
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        adj=adj
    )
    
    # Main evaluation
    logger.info(f"Evaluating on {args.split} split...")
    metrics, coverage, diversity = evaluator.evaluate(
        split=args.split,
        k_values=args.k_values
    )
    
    # Log results
    logger.info("Overall Results:")
    for k in args.k_values:
        logger.info(f"@{k}: NDCG={metrics[k]['ndcg']:.4f}, "
                   f"HR={metrics[k]['hit_rate']:.4f}, "
                   f"Precision={metrics[k]['precision']:.4f}, "
                   f"Recall={metrics[k]['recall']:.4f}")
    logger.info(f"Coverage: {coverage:.4f}, Diversity: {diversity:.4f}")
    
    # User group evaluation
    group_results = None
    if args.user_groups:
        logger.info("\nEvaluating user groups...")
        group_results = evaluate_user_groups(
            evaluator, dataset, args.k_values, logger
        )
    
    # Save predictions
    predictions_df = None
    if args.save_predictions:
        logger.info("\nSaving predictions...")
        predictions_path = output_dir / f'predictions_{timestamp}.csv'
        predictions_df = save_predictions(
            evaluator, dataset, predictions_path
        )
        logger.info(f"Predictions saved to {predictions_path}")
    
    # Save all results
    results = {
        'checkpoint': args.checkpoint,
        'dataset': config['dataset']['name'],
        'split': args.split,
        'timestamp': timestamp,
        'metrics': {k: metrics[k] for k in args.k_values},
        'coverage': coverage,
        'diversity': diversity,
        'user_group_results': group_results,
        'config': config
    }
    
    # Save as JSON
    results_path = output_dir / f'results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    
    # Create summary table
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Dataset: {config['dataset']['name']}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Best NDCG@10: {metrics[10]['ndcg']:.4f}")
    logger.info(f"Best HR@10: {metrics[10]['hit_rate']:.4f}")
    
    if group_results:
        logger.info("\nCold-Start Performance:")
        cold_metrics = group_results['cold']['metrics']
        logger.info(f"Cold NDCG@10: {cold_metrics[10]['ndcg']:.4f}")
        logger.info(f"Cold HR@10: {cold_metrics[10]['hit_rate']:.4f}")


if __name__ == '__main__':
    main()