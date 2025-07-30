"""Data loaders for training and evaluation"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple


class BPRDataset(Dataset):
    """Dataset for Bayesian Personalized Ranking training
    
    Args:
        dataset: RecommenderDataset object
        n_neg_samples: Number of negative samples per positive sample
    """
    
    def __init__(self, dataset, n_neg_samples: int = 1):
        self.dataset = dataset
        self.n_neg_samples = n_neg_samples
        
        # Extract positive samples
        self.users = []
        self.pos_items = []
        
        for _, row in dataset.train_ratings.iterrows():
            self.users.append(row['user'])
            self.pos_items.append(row['item'])
            
        self.n_samples = len(self.users)
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample
        
        Returns:
            Dictionary with users, positive items, negative items
        """
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        
        # Sample negative items
        neg_items = []
        for _ in range(self.n_neg_samples):
            neg_item = np.random.randint(1, self.dataset.n_items + 1)
            while self.dataset.train_mat[user, neg_item] > 0:
                neg_item = np.random.randint(1, self.dataset.n_items + 1)
            neg_items.append(neg_item)
        
        return {
            'users': torch.tensor(user, dtype=torch.long),
            'pos_items': torch.tensor(pos_item, dtype=torch.long),
            'neg_items': torch.tensor(neg_items[0] if self.n_neg_samples == 1 else neg_items,
                                    dtype=torch.long),
            'user_items': self.dataset.train_mat[user]  # For cooperative game training
        }


class UniformSampleDataset(Dataset):
    """Uniform sampling dataset for evaluation
    
    Args:
        dataset: RecommenderDataset object
        phase: 'train', 'val', or 'test'
    """
    
    def __init__(self, dataset, phase: str = 'test'):
        self.dataset = dataset
        self.phase = phase
        
        # Get all users with interactions in this phase
        if phase == 'val':
            self.users = list(dataset.val_dict.keys())
        elif phase == 'test':
            self.users = list(dataset.test_dict.keys())
        else:
            raise ValueError(f"Invalid phase: {phase}")
            
    def __len__(self) -> int:
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get user data for evaluation"""
        user = self.users[idx]
        
        return {
            'user': torch.tensor(user, dtype=torch.long),
            'items': torch.arange(1, self.dataset.n_items + 1, dtype=torch.long)
        }


class UserBatchDataset(Dataset):
    """Dataset that returns user batches for efficient evaluation
    
    Args:
        dataset: RecommenderDataset object
        user_list: List of users to evaluate
    """
    
    def __init__(self, dataset, user_list):
        self.dataset = dataset
        self.user_list = user_list
        
    def __len__(self) -> int:
        return len(self.user_list)
    
    def __getitem__(self, idx: int) -> int:
        return self.user_list[idx]


def get_dataloader(dataset, split: str, config: Dict, **kwargs) -> DataLoader:
    """Get data loader for specific split
    
    Args:
        dataset: RecommenderDataset object
        split: 'train', 'val', or 'test'
        config: Configuration dictionary
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader object
    """
    if split == 'train':
        # BPR training
        train_dataset = BPRDataset(
            dataset,
            n_neg_samples=config.get('n_neg_samples', 1)
        )
        
        return DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
            **kwargs
        )
        
    elif split in ['val', 'test']:
        # Evaluation
        eval_dataset = UniformSampleDataset(dataset, phase=split)
        
        return DataLoader(
            eval_dataset,
            batch_size=config['evaluation'].get('eval_batch_size', 512),
            shuffle=False,
            num_workers=config['training'].get('num_workers', 4),
            pin_memory=True,
            **kwargs
        )
        
    else:
        raise ValueError(f"Unknown split: {split}")


def collate_fn_bpr(batch):
    """Custom collate function for BPR training
    
    Args:
        batch: List of samples
        
    Returns:
        Dictionary of batched tensors
    """
    users = torch.stack([sample['users'] for sample in batch])
    pos_items = torch.stack([sample['pos_items'] for sample in batch])
    neg_items = torch.stack([sample['neg_items'] for sample in batch])
    
    # Handle user_items if present
    if 'user_items' in batch[0]:
        user_items = torch.stack([sample['user_items'] for sample in batch])
        return {
            'users': users,
            'pos_items': pos_items,
            'neg_items': neg_items,
            'user_items': user_items
        }
    
    return {
        'users': users,
        'pos_items': pos_items,
        'neg_items': neg_items
    }