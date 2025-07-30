#!/usr/bin/env python
"""
Training script for DyHuCoG and baseline models
"""

import os
import sys
import argparse
import time
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
from src.models.sgl import SGL
from src.models.simgcl import SimGCL
from src.models.hccf import HCCF
from src.utils.trainer import Trainer
from src.utils.logger import setup_logger
from src.utils.graph_builder import GraphBuilder

# Import extended datasets
try:
    from src.data.yelp_dataset import YelpDataset, GowallaDataset, AmazonElectronicsDataset
    EXTENDED_DATASETS = True
except ImportError:
    EXTENDED_DATASETS = False
    print("Warning: Extended datasets not available")


def parse_args():
    parser = argparse.ArgumentParser(description='Train DyHuCoG model')
    
    # Basic arguments
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset name: ml-100k, ml-1m, amazon-book, yelp2018, gowalla, amazon-electronics')
    parser.add_argument('--model', type=str, default='dyhucog',
                       help='Model name: dyhucog, lightgcn, ngcf, sgl, simgcl, hccf')
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
    
    # SSL arguments for new models
    parser.add_argument('--ssl_temp', type=float, default=None,
                       help='Temperature for SSL loss')
    parser.add_argument('--ssl_reg', type=float, default=None,
                       help='SSL regularization weight')
    
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
        
    # SSL parameters for new models
    if args.ssl_temp is not None:
        config['model']['ssl_temp'] = args.ssl_temp
    if args.ssl_reg is not None:
        config['model']['ssl_reg'] = args.ssl_reg
        
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


def load_dataset(dataset_name: str, config: dict):
    """Load dataset based on name"""
    dataset_config = config['dataset']
    
    # Standard datasets
    if dataset_name in ['ml-100k', 'ml-1m', 'amazon-book']:
        return RecommenderDataset(
            name=dataset_name,
            path=dataset_config['path'],
            test_size=dataset_config['test_size'],
            val_size=dataset_config['val_size']
        )
    
    # Extended datasets
    if EXTENDED_DATASETS:
        if dataset_name == 'yelp2018':
            return YelpDataset(
                name=dataset_name,
                path=dataset_config['path'],
                test_size=dataset_config['test_size'],
                val_size=dataset_config['val_size']
            )
        elif dataset_name == 'gowalla':
            return GowallaDataset(
                path=dataset_config['path'],
                test_size=dataset_config['test_size'],
                val_size=dataset_config['val_size']
            )
        elif dataset_name == 'amazon-electronics':
            return AmazonElectronicsDataset(
                path=dataset_config['path'],
                test_size=dataset_config['test_size'],
                val_size=dataset_config['val_size']
            )
    
    raise ValueError(f"Unknown or unavailable dataset: {dataset_name}")


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
    elif model_name == 'sgl':
        model = SGL(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout'],
            ssl_temp=model_config.get('ssl_temp', 0.2),
            ssl_reg=model_config.get('ssl_reg', 0.1),
            aug_type=model_config.get('aug_type', 'ed')
        )
    elif model_name == 'simgcl':
        model = SimGCL(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers'],
            noise_eps=model_config.get('noise_eps', 0.1),
            ssl_temp=model_config.get('ssl_temp', 0.2),
            ssl_reg=model_config.get('ssl_reg', 0.1)
        )
    elif model_name == 'hccf':
        model = HCCF(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            latent_dim=model_config['latent_dim'],
            n_layers=model_config['n_layers'],
            n_hyperedges=model_config.get('n_hyperedges', 1000),
            ssl_temp=model_config.get('ssl_temp', 0.2),
            ssl_reg=model_config.get('ssl_reg', 0.1)
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
    dataset = load_dataset(config['dataset']['name'], config)
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
    
    # Special initialization for models
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
                
                # Log every 50 batches instead of 10
                if batch_idx % 50 == 0:
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
        logger.info("Note: Using optimized Shapley computation with batching and reduced samples")
        shapley_start_time = time.time()
        
        for epoch in range(config['model']['shapley_epochs']):
            epoch_start_time = time.time()
            total_loss = 0
            batch_count = 0
            
            # Increment epoch in trainer for loss strategy
            coop_trainer.increment_epoch()
            
            for batch_idx, batch in enumerate(train_loader):
                user_items = batch['user_items'].to(device)
                
                # Log first batch details
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"  Shapley batch shape: {user_items.shape}")
                    logger.info(f"  Non-zero elements per user: {(user_items > 0).sum(dim=1).float().mean():.2f}")
                    logger.info(f"  Using {'exact' if epoch < 3 else 'self-supervised'} Shapley training")
                
                batch_start_time = time.time()
                loss = coop_trainer.train_shapley_step(user_items)
                batch_time = time.time() - batch_start_time
                
                total_loss += loss
                batch_count += 1
                
                # Log every 50 batches with timing
                if batch_idx % 50 == 0:
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
        
        # Compute in batches to save memory
        edge_weights = {}
        batch_size = 100
        
        for user_start in range(1, dataset.n_users + 1, batch_size):
            user_end = min(user_start + batch_size, dataset.n_users + 1)
            user_batch = range(user_start, user_end)
            
            # Get user items for batch
            user_items_batch = []
            user_ids = []
            
            for user_id in user_batch:
                user_items = dataset.train_mat[user_id]
                if user_items.sum() > 0:
                    user_items_batch.append(user_items[1:])  # Remove index 0
                    user_ids.append(user_id)
            
            if user_items_batch:
                # Compute Shapley values for batch
                user_items_tensor = torch.stack(user_items_batch).to(device)
                with torch.no_grad():
                    shapley_values_batch = model.shapley_net(user_items_tensor)
                
                # Extract edge weights
                for i, user_id in enumerate(user_ids):
                    user_items = dataset.train_mat[user_id]
                    item_indices = torch.where(user_items > 0)[0]
                    shapley_vals = shapley_values_batch[i]
                    
                    for item_idx in item_indices:
                        item_id = item_idx.item()
                        if item_id > 0:  # Skip index 0
                            shapley_val = shapley_vals[item_id - 1].item()
                            weight = max(shapley_val, 0.1) + 1.0
                            edge_weights[(user_id, item_id)] = weight
            
            if user_start % 500 == 1:
                logger.info(f"  Processed users {user_start} to {user_end-1}")
        
        model.edge_weights = edge_weights
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
        
    elif config['model']['name'] == 'hccf':
        # Build hypergraph for HCCF
        logger.info("Building hypergraph for HCCF...")
        model.build_hypergraph(dataset.train_mat)
    
    # Create trainer with SSL support
    logger.info("Creating trainer...")
    
    # Modify trainer for SSL models
    if config['model']['name'] in ['sgl', 'simgcl', 'hccf']:
        logger.info(f"Enabling SSL training for {config['model']['name']}")
        config['training']['use_ssl'] = True
    
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
        'n_genres': dataset.n_genres,
        'edge_weights': model.edge_weights if hasattr(model, 'edge_weights') else None
    }, checkpoint_dir / 'best_model.pth')
    
    logger.info(f"Training completed! Results saved to {results_dir}")
    logger.info(f"Best validation NDCG@10: {trainer.best_metrics.get('ndcg@10', 'N/A'):.4f}")


if __name__ == '__main__':
    main()