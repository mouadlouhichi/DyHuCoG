"""Dataset classes for recommendation systems"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RecommenderDataset:
    """Base class for recommender system datasets
    
    Args:
        name: Dataset name
        path: Path to dataset
        test_size: Test set ratio
        val_size: Validation set ratio
    """
    
    def __init__(self, name: str, path: str, test_size: float = 0.2, 
                 val_size: float = 0.1):
        self.name = name
        self.path = Path(path) / name
        self.test_size = test_size
        self.val_size = val_size
        
        # Initialize dataset attributes
        self.n_users = 0
        self.n_items = 0
        self.n_interactions = 0
        
        # Data storage
        self.ratings = None
        self.train_ratings = None
        self.val_ratings = None
        self.test_ratings = None
        
        # Interaction matrices
        self.train_mat = None
        self.val_dict = defaultdict(list)
        self.test_dict = defaultdict(list)
        
        # User groups
        self.cold_users = []
        self.warm_users = []
        self.hot_users = []
        
        # Load specific dataset
        if name == 'ml-100k':
            self._load_ml100k()
        elif name == 'ml-1m':
            self._load_ml1m()
        elif name == 'amazon-book':
            self._load_amazon_book()
        else:
            raise ValueError(f"Unknown dataset: {name}")
            
        # Process data
        self._split_data()
        self._build_interaction_matrices()
        self._identify_user_groups()
        
    def _load_ml100k(self):
        """Load MovieLens-100K dataset"""
        # Load ratings
        self.ratings = pd.read_csv(
            self.path / 'u.data',
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp'],
            engine='python'
        )
        
        self.n_users = self.ratings['user'].max()
        self.n_items = self.ratings['item'].max()
        
        # Convert to implicit feedback (rating > 3)
        self.ratings = self.ratings[self.ratings['rating'] > 3]
        self.n_interactions = len(self.ratings)
        
        # Load item features (genres)
        self.genre_cols = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Childrens',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        self.items = pd.read_csv(
            self.path / 'u.item',
            sep='|',
            names=['item', 'title', 'release_date', 'video_release_date', 
                   'imdb_url'] + self.genre_cols,
            encoding='latin-1',
            engine='python'
        )
        
        self.n_genres = len(self.genre_cols)
        
        # Create item-genre mapping
        self.item_genres = {}
        for _, row in self.items.iterrows():
            item_id = row['item']
            genres = [i for i, g in enumerate(self.genre_cols) if row[g] == 1]
            self.item_genres[item_id] = genres
            
        logger.info(f"Loaded ML-100K: {self.n_users} users, {self.n_items} items, "
                   f"{self.n_interactions} interactions")
    
    def _load_ml1m(self):
        """Load MovieLens-1M dataset"""
        # Load ratings
        self.ratings = pd.read_csv(
            self.path / 'ratings.dat',
            sep='::',
            names=['user', 'item', 'rating', 'timestamp'],
            engine='python'
        )
        
        self.n_users = self.ratings['user'].max()
        self.n_items = self.ratings['item'].max()
        
        # Convert to implicit feedback
        self.ratings = self.ratings[self.ratings['rating'] > 3]
        self.n_interactions = len(self.ratings)
        
        # Load movies
        self.items = pd.read_csv(
            self.path / 'movies.dat',
            sep='::',
            names=['item', 'title', 'genres'],
            encoding='latin-1',
            engine='python'
        )
        
        # Process genres
        all_genres = set()
        for genres in self.items['genres']:
            all_genres.update(genres.split('|'))
        
        self.genre_cols = sorted(list(all_genres))
        self.n_genres = len(self.genre_cols)
        
        # Create item-genre mapping
        self.item_genres = {}
        genre_to_idx = {g: i for i, g in enumerate(self.genre_cols)}
        
        for _, row in self.items.iterrows():
            item_id = row['item']
            genres = row['genres'].split('|')
            genre_indices = [genre_to_idx[g] for g in genres if g in genre_to_idx]
            self.item_genres[item_id] = genre_indices
            
        logger.info(f"Loaded ML-1M: {self.n_users} users, {self.n_items} items, "
                   f"{self.n_interactions} interactions")
    
    def _load_amazon_book(self):
        """Load Amazon Book dataset"""
        # This is a placeholder - implement based on your data format
        raise NotImplementedError("Amazon Book dataset loading not implemented")
    
    def _split_data(self):
        """Temporal split for train/val/test sets"""
        # Sort by timestamp for temporal split
        self.ratings = self.ratings.sort_values('timestamp')
        
        # Calculate split points
        n_interactions = len(self.ratings)
        train_size = int(n_interactions * (1 - self.test_size - self.val_size))
        val_size = int(n_interactions * (1 - self.test_size))
        
        # Split data
        self.train_ratings = self.ratings.iloc[:train_size]
        self.val_ratings = self.ratings.iloc[train_size:val_size]
        self.test_ratings = self.ratings.iloc[val_size:]
        
        logger.info(f"Data split - Train: {len(self.train_ratings)}, "
                   f"Val: {len(self.val_ratings)}, Test: {len(self.test_ratings)}")
    
    def _build_interaction_matrices(self):
        """Build sparse interaction matrices"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Train matrix
        self.train_mat = torch.zeros((self.n_users + 1, self.n_items + 1), device=device)
        for _, row in self.train_ratings.iterrows():
            self.train_mat[row['user'], row['item']] = 1
            
        # Validation and test dictionaries
        for _, row in self.val_ratings.iterrows():
            self.val_dict[row['user']].append(row['item'])
            
        for _, row in self.test_ratings.iterrows():
            self.test_dict[row['user']].append(row['item'])
    
    def _identify_user_groups(self):
        """Identify cold, warm, and hot users based on interaction count"""
        user_counts = self.train_ratings['user'].value_counts()
        
        for user in range(1, self.n_users + 1):
            count = user_counts.get(user, 0)
            if count < 5:
                self.cold_users.append(user)
            elif count < 20:
                self.warm_users.append(user)
            else:
                self.hot_users.append(user)
                
        logger.info(f"User groups - Cold: {len(self.cold_users)}, "
                   f"Warm: {len(self.warm_users)}, Hot: {len(self.hot_users)}")
    
    def get_user_history(self, user_id: int, split: str = 'train') -> List[int]:
        """Get interaction history for a user
        
        Args:
            user_id: User ID
            split: Data split ('train', 'val', 'test')
            
        Returns:
            List of item IDs
        """
        if split == 'train':
            return torch.where(self.train_mat[user_id] > 0)[0].tolist()
        elif split == 'val':
            return self.val_dict.get(user_id, [])
        elif split == 'test':
            return self.test_dict.get(user_id, [])
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def get_item_users(self, item_id: int) -> List[int]:
        """Get users who interacted with an item
        
        Args:
            item_id: Item ID
            
        Returns:
            List of user IDs
        """
        return torch.where(self.train_mat[:, item_id] > 0)[0].tolist()
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_interactions': self.n_interactions,
            'n_train': len(self.train_ratings),
            'n_val': len(self.val_ratings),
            'n_test': len(self.test_ratings),
            'density': self.n_interactions / (self.n_users * self.n_items),
            'avg_interactions_per_user': self.n_interactions / self.n_users,
            'avg_interactions_per_item': self.n_interactions / self.n_items,
            'n_cold_users': len(self.cold_users),
            'n_warm_users': len(self.warm_users),
            'n_hot_users': len(self.hot_users)
        }
        
        if hasattr(self, 'n_genres'):
            stats['n_genres'] = self.n_genres
            
        return stats


class MovieLensDataset(RecommenderDataset):
    """MovieLens dataset wrapper"""
    
    def __init__(self, version: str = '100k', **kwargs):
        """Initialize MovieLens dataset
        
        Args:
            version: Dataset version ('100k', '1m', '10m', '20m')
            **kwargs: Additional arguments for parent class
        """
        if version == '100k':
            name = 'ml-100k'
        elif version == '1m':
            name = 'ml-1m'
        else:
            raise ValueError(f"Unsupported MovieLens version: {version}")
            
        super().__init__(name=name, **kwargs)


class AmazonDataset(RecommenderDataset):
    """Amazon dataset wrapper"""
    
    def __init__(self, category: str = 'book', **kwargs):
        """Initialize Amazon dataset
        
        Args:
            category: Product category
            **kwargs: Additional arguments for parent class
        """
        name = f'amazon-{category}'
        super().__init__(name=name, **kwargs)