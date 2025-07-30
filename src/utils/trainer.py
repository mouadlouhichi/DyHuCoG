"""
Training utilities for DyHuCoG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
import time

from ..data.dataloader import get_dataloader
from .evaluator import Evaluator
from .metrics import calculate_metrics
from .logger import setup_logger


class Trainer:
    """Trainer class for DyHuCoG and baseline models
    
    Args:
        model: Model to train
        dataset: Dataset object
        config: Training configuration
        device: Torch device
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for tensorboard logs
        adj: Adjacency matrix (optional, for baseline models)
    """
    
    def __init__(self, model: nn.Module, dataset, config: Dict,
                 device: torch.device, checkpoint_dir: Path,
                 log_dir: Path, adj: Optional[torch.Tensor] = None):
        
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.adj = adj
        
        # Training settings
        self.epochs = config['training']['epochs']
        self.eval_every = config['training']['eval_every']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        self.save_best = config['training']['save_best']
        
        # Create data loaders
        self.train_loader = get_dataloader(dataset, 'train', config)
        self.val_loader = get_dataloader(dataset, 'val', config)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Evaluator
        self.evaluator = Evaluator(
            model, dataset, device, 
            batch_size=config['evaluation']['eval_batch_size'],
            adj=adj
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir) if config['experiment'].get('tensorboard', True) else None
        
        # Logger
        self.logger = setup_logger('trainer', log_dir / 'trainer.log')
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = -float('inf')
        self.best_metrics = {}
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_metrics': [],
            'test_metrics': [],
            'epoch_times': []
        }
        
    def train(self) -> Dict:
        """Main training loop
        
        Returns:
            Training history dictionary
        """
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.epochs):
            epoch_start_time = time.time()
            
            # Training epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluation
            if (epoch + 1) % self.eval_every == 0:
                val_metrics = self.evaluate('val', epoch)
                test_metrics = self.evaluate('test', epoch)
                
                # Update best metrics
                current_val_ndcg = val_metrics[10]['ndcg']
                if current_val_ndcg > self.best_val_metric:
                    self.best_val_metric = current_val_ndcg
                    self.best_metrics = {
                        'epoch': epoch,
                        'val_metrics': val_metrics,
                        'test_metrics': test_metrics
                    }
                    self.patience_counter = 0
                    
                    if self.save_best:
                        self.save_checkpoint('best_model.pth', epoch)
                else:
                    self.patience_counter += 1
                
                # Learning rate scheduling
                self.scheduler.step(current_val_ndcg)
                
                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch)
            
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch_times'].append(epoch_time)
            
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation NDCG@10: {self.best_val_metric:.4f} at epoch {self.best_metrics.get('epoch', -1)}")
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
            
        return self.training_history
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            users = batch['users'].to(self.device)
            pos_items = batch['pos_items'].to(self.device)
            neg_items = batch['neg_items'].to(self.device)
            
            # Forward pass
            if self.adj is not None:
                # For baseline models (LightGCN, NGCF)
                pos_scores = self.model.predict(users, pos_items, self.adj)
                neg_scores = self.model.predict(users, neg_items, self.adj)
            else:
                # For DyHuCoG
                pos_scores = self.model.predict(users, pos_items)
                neg_scores = self.model.predict(users, neg_items)
            
            # BPR loss
            loss = self.bpr_loss(pos_scores, neg_scores)
            
            # Add regularization
            reg_loss = self.regularization_loss()
            total_loss_batch = loss + reg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Update progress bar
            total_loss += total_loss_batch.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Log to tensorboard
            if self.writer and batch_idx % 100 == 0:
                global_step = epoch * n_batches + batch_idx
                self.writer.add_scalar('Loss/train_batch', total_loss_batch.item(), global_step)
        
        avg_epoch_loss = total_loss / n_batches
        self.training_history['train_loss'].append(avg_epoch_loss)
        
        if self.writer:
            self.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
            
        return avg_epoch_loss
    
    def bpr_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """Bayesian Personalized Ranking loss
        
        Args:
            pos_scores: Scores for positive items
            neg_scores: Scores for negative items
            
        Returns:
            BPR loss value
        """
        return -F.logsigmoid(pos_scores - neg_scores).mean()
    
    def regularization_loss(self) -> torch.Tensor:
        """L2 regularization loss
        
        Returns:
            Regularization loss value
        """
        reg_loss = 0
        for param in self.model.parameters():
            reg_loss += param.norm(2).pow(2)
        
        return self.config['training']['weight_decay'] * reg_loss
    
    def evaluate(self, split: str, epoch: int) -> Dict:
        """Evaluate model on validation or test set
        
        Args:
            split: 'val' or 'test'
            epoch: Current epoch number
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating on {split} set...")
        
        # Get user list for evaluation
        if split == 'val':
            # Use a subset of warm users for validation
            user_list = self.dataset.warm_users[:min(1000, len(self.dataset.warm_users))]
        else:
            # Use all users for test
            user_list = list(range(1, self.dataset.n_users + 1))
        
        # Evaluate
        metrics, coverage, diversity = self.evaluator.evaluate(
            user_list=user_list,
            k_values=self.config['evaluation']['k_values'],
            split=split
        )
        
        # Log results
        self.logger.info(f"{split.capitalize()} Results - Epoch {epoch+1}:")
        for k in self.config['evaluation']['k_values']:
            self.logger.info(f"  @{k}: NDCG={metrics[k]['ndcg']:.4f}, "
                           f"HR={metrics[k]['hit_rate']:.4f}, "
                           f"Precision={metrics[k]['precision']:.4f}, "
                           f"Recall={metrics[k]['recall']:.4f}")
        self.logger.info(f"  Coverage: {coverage:.4f}, Diversity: {diversity:.4f}")
        
        # Log to tensorboard
        if self.writer:
            for k in self.config['evaluation']['k_values']:
                for metric_name, value in metrics[k].items():
                    self.writer.add_scalar(f'{split}/{metric_name}@{k}', value, epoch)
            self.writer.add_scalar(f'{split}/coverage', coverage, epoch)
            self.writer.add_scalar(f'{split}/diversity', diversity, epoch)
        
        # Store in history
        if split == 'val':
            self.training_history['val_metrics'].append((metrics, coverage, diversity))
        else:
            self.training_history['test_metrics'].append((metrics, coverage, diversity))
            
        return metrics
    
    def save_checkpoint(self, filename: str, epoch: int):
        """Save model checkpoint
        
        Args:
            filename: Checkpoint filename
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'config': self.config,
            'n_users': self.dataset.n_users,
            'n_items': self.dataset.n_items,
            'n_genres': getattr(self.dataset, 'n_genres', 0)
        }
        
        # Add model-specific information
        if hasattr(self.model, 'edge_weights'):
            checkpoint['edge_weights'] = self.model.edge_weights
            
        torch.save(checkpoint, self.checkpoint_dir / filename)
        self.logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_metric = checkpoint['best_val_metric']
        self.best_metrics = checkpoint['best_metrics']
        self.training_history = checkpoint['training_history']
        
        # Load model-specific information
        if 'edge_weights' in checkpoint and hasattr(self.model, 'edge_weights'):
            self.model.edge_weights = checkpoint['edge_weights']
            
        self.logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")