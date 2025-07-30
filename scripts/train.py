#!/usr/bin/env python
"""
Training script for DyHuCoG and baseline models
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import RecommenderDataset
from src.data.dataloader import get_dataloader
from src.models.dyhucog import DyHuCoG
from src.models.lightgcn import LightGCN
from src.models.ngcf import NGCF
from src.utils.trainer import Trainer
from src.utils.logger import setup_logger
from src.utils.graph_builder import GraphBuilder


def parse_args():
    parser = argparse.ArgumentParser(description='Train DyHuCoG model')
    
    # Basic arguments
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset name: ml-100k, ml-1m, amazon-book')
    parser.add_argument('--model', type=str, default='dyhucog',
                       help='Model name: dyhucog, lightgcn, ngcf')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    # Override config arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: auto, cpu, cuda, cuda:0')
    
    # Other arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> dict:
    """Load configuration and override with command line arguments"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.model:
        config['model']['name'] = args.model
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.device is not None:
        config['training']['device'] = args.device
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
        
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Get torch device from string"""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def create_model(model_name: str, dataset: RecommenderDataset, 
                 config: dict, device: torch.device):
    """Create model based on name"""
    model_config = config['model']
    
    if model_name == 'dyhucog':
        model = DyHuCoG(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            n_genres=dataset.n_genres,
            config=model_config
        )
    elif model_name == 'lightgcn':
        model = LightGCN(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout']
        )
    elif model_name == 'ngcf':
        model = NGCF(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Set up experiment
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config['model']['name']}_{config['dataset']['name']}_{timestamp}"
    
    # Create directories
    checkpoint_dir = Path(config['experiment']['checkpoint_dir']) / experiment_name
    log_dir = Path(config['experiment']['log_dir']) / experiment_name
    results_dir = Path(config['experiment']['results_dir']) / experiment_name
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = setup_logger(
        name='train',
        log_file=log_dir / 'train.log',
        level=config['logging']['level']
    )
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Configuration: {config}")
    
    # Set seed
    seed = config['experiment']['seed']
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Get device
    device = get_device(config['training']['device'])
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    dataset = RecommenderDataset(
        name=config['dataset']['name'],
        path=config['dataset']['path'],
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size']
    )
    logger.info(f"Dataset loaded - Users: {dataset.n_users}, Items: {dataset.n_items}")
    
    # Build graph
    logger.info("Building graph structures...")
    if config['model']['name'] == 'dyhucog' and config['model'].get('use_genres', True):
        # Build hypergraph for DyHuCoG
        adj, n_nodes = GraphBuilder.build_hypergraph(dataset)
    else:
        # Build user-item bipartite graph
        adj, n_nodes = GraphBuilder.build_user_item_graph(dataset)
    
    adj = GraphBuilder.normalize_adj(adj).to(device)
    logger.info(f"Graph built - Nodes: {n_nodes}, Edges: {adj._nnz()}")
    
    # Create model
    logger.info(f"Creating model: {config['model']['name']}")
    model = create_model(config['model']['name'], dataset, config, device)
    
    # Special initialization for DyHuCoG
# In scripts/train.py, update the pre-training section:

    if config['model']['name'] == 'dyhucog':
        # Pre-train cooperative game components
        train_loader = get_dataloader(dataset, 'train', config)
        
        logger.info("Pre-training cooperative game components...")
        logger.info(f"  DAE epochs: {config['model']['dae_epochs']}")
        logger.info(f"  Shapley epochs: {config['model']['shapley_epochs']}")
        logger.info(f"  Batch size: {config['training']['batch_size']}")
        logger.info(f"  Number of batches: {len(train_loader)}")
        
        from src.models.cooperative_game import CooperativeGameTrainer
        
        coop_trainer = CooperativeGameTrainer(
            model.dae, model.shapley_net, config['model']
        )
        
        # Train DAE
        logger.info("\nStarting DAE pre-training...")
        dae_start_time = time.time()
        
        for epoch in range(config['model']['dae_epochs']):
            epoch_start_time = time.time()
            total_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                user_items = batch['user_items'].to(device)
                loss = coop_trainer.train_dae_step(user_items)
                total_loss += loss
                batch_count += 1
                
                # Log every 10 batches
                if batch_idx % 10 == 0:
                    logger.info(f"  DAE Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}: "
                            f"Loss = {loss:.4f}")
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / batch_count
            
            logger.info(f"DAE Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, "
                    f"Time = {epoch_time:.2f}s ({epoch_time/batch_count:.3f}s/batch)")
        
        dae_total_time = time.time() - dae_start_time
        logger.info(f"DAE pre-training completed in {dae_total_time:.2f}s")
        
        # Train Shapley network
        logger.info("\nStarting Shapley network pre-training...")
        shapley_start_time = time.time()
        
        for epoch in range(config['model']['shapley_epochs']):
            epoch_start_time = time.time()
            total_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                user_items = batch['user_items'].to(device)
                
                # Log first batch details
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"  Shapley batch shape: {user_items.shape}")
                    logger.info(f"  Non-zero elements per user: {(user_items > 0).sum(dim=1).float().mean():.2f}")
                
                batch_start_time = time.time()
                loss = coop_trainer.train_shapley_step(user_items)
                batch_time = time.time() - batch_start_time
                
                total_loss += loss
                batch_count += 1
                
                # Log every 10 batches with timing
                if batch_idx % 10 == 0:
                    logger.info(f"  Shapley Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}: "
                            f"Loss = {loss:.4f}, Time = {batch_time:.3f}s")
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / batch_count
            
            logger.info(f"Shapley Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, "
                    f"Time = {epoch_time:.2f}s ({epoch_time/batch_count:.3f}s/batch)")
        
        shapley_total_time = time.time() - shapley_start_time
        logger.info(f"Shapley pre-training completed in {shapley_total_time:.2f}s")
        
        # Update edge weights
        logger.info("\nComputing Shapley value edge weights...")
        weight_start_time = time.time()
        model.edge_weights = model.compute_shapley_weights(dataset.train_mat.to(device))
        weight_time = time.time() - weight_start_time
        logger.info(f"Edge weight computation completed in {weight_time:.2f}s")
        logger.info(f"Number of edge weights: {len(model.edge_weights)}")
        
        # Build weighted hypergraph
        logger.info("\nBuilding weighted hypergraph...")
        graph_start_time = time.time()
        edge_index, edge_weight = GraphBuilder.get_edge_list(dataset, model.edge_weights)
        model.build_hypergraph(edge_index.to(device), edge_weight.to(device), dataset.item_genres)
        graph_time = time.time() - graph_start_time
        logger.info(f"Hypergraph built in {graph_time:.2f}s")
    
        total_preprocessing_time = time.time() - dae_start_time
        logger.info(f"\nTotal DyHuCoG preprocessing time: {total_preprocessing_time:.2f}s")
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        adj=adj if config['model']['name'] != 'dyhucog' else None
    )
    
    # Resume from checkpoint if specified
    if args.checkpoint:
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train()
    
    # Save final results
    logger.info("Saving results...")
    results = {
        'config': config,
        'history': history,
        'best_metrics': trainer.best_metrics,
        'experiment_name': experiment_name
    }
    
    torch.save(results, results_dir / 'results.pth')
    
    # Save best model separately
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'n_users': dataset.n_users,
        'n_items': dataset.n_items,
        'n_genres': dataset.n_genres
    }, checkpoint_dir / 'best_model.pth')
    
    logger.info(f"Training completed! Results saved to {results_dir}")
    logger.info(f"Best validation NDCG@10: {trainer.best_metrics['ndcg@10']:.4f}")


if __name__ == '__main__':
    main()