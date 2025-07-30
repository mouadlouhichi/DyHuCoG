"""Yelp dataset loader"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import torch

from .dataset import RecommenderDataset


class YelpDataset(RecommenderDataset):
    """Yelp dataset for recommendation
    
    Supports:
    - Yelp 2018 
    - Yelp 2020
    """
    
    def __init__(self, name: str = 'yelp2018', path: str = 'data/',
                 test_size: float = 0.2, val_size: float = 0.1,
                 min_interactions: int = 10):
        self.min_interactions = min_interactions
        
        # Initialize parent class
        super().__init__(name=name, path=path, test_size=test_size, val_size=val_size)
        
    def _load_yelp2018(self):
        """Load Yelp 2018 dataset"""
        data_path = self.path
        
        # Load interactions
        train_file = data_path / 'train.txt'
        test_file = data_path / 'test.txt'
        
        # Read training data
        train_data = []
        with open(train_file, 'r') as f:
            for idx, line in enumerate(f):
                items = line.strip().split()
                user = idx + 1
                for item in items:
                    train_data.append({
                        'user': user,
                        'item': int(item) + 1,
                        'rating': 1.0,
                        'timestamp': 0
                    })
        
        # Read test data
        test_data = []
        with open(test_file, 'r') as f:
            for idx, line in enumerate(f):
                items = line.strip().split()
                user = idx + 1
                for item in items:
                    test_data.append({
                        'user': user,
                        'item': int(item) + 1,
                        'rating': 1.0,
                        'timestamp': 1
                    })
        
        # Combine and create DataFrame
        all_data = train_data + test_data
        self.ratings = pd.DataFrame(all_data)
        
        # Basic statistics
        self.n_users = self.ratings['user'].max()
        self.n_items = self.ratings['item'].max()
        self.n_interactions = len(self.ratings)
        
        # Load categories as genres
        self._load_categories()
        
    def _load_categories(self):
        """Load business categories as item features"""
        category_file = self.path / 'business_categories.json'
        
        if category_file.exists():
            with open(category_file, 'r') as f:
                categories_data = json.load(f)
            
            # Get unique categories
            all_categories = set()
            for cats in categories_data.values():
                all_categories.update(cats)
            
            self.genre_cols = sorted(list(all_categories))
            self.n_genres = len(self.genre_cols)
            
            # Create item-category mapping
            cat_to_idx = {cat: idx for idx, cat in enumerate(self.genre_cols)}
            
            self.item_genres = {}
            for item_str, cats in categories_data.items():
                item_id = int(item_str)
                self.item_genres[item_id] = [cat_to_idx[cat] for cat in cats if cat in cat_to_idx]
        else:
            # No category info
            self.n_genres = 0
            self.genre_cols = []
            self.item_genres = {}


class GowallaDataset(RecommenderDataset):
    """Gowalla dataset for recommendation"""
    
    def __init__(self, path: str = 'data/gowalla',
                 test_size: float = 0.2, val_size: float = 0.1):
        super().__init__(name='gowalla', path=path, 
                        test_size=test_size, val_size=val_size)
    
    def _load_gowalla(self):
        """Load Gowalla dataset"""
        # Load check-ins
        train_file = self.path / 'train.txt'
        test_file = self.path / 'test.txt'
        
        # Similar loading as Yelp
        train_data = []
        with open(train_file, 'r') as f:
            for idx, line in enumerate(f):
                items = line.strip().split()
                user = idx + 1
                for item in items:
                    train_data.append({
                        'user': user,
                        'item': int(item) + 1,
                        'rating': 1.0,
                        'timestamp': 0
                    })
        
        self.ratings = pd.DataFrame(train_data)
        
        self.n_users = self.ratings['user'].max()
        self.n_items = self.ratings['item'].max()
        self.n_interactions = len(self.ratings)
        
        # Gowalla doesn't have genres
        self.n_genres = 0
        self.genre_cols = []
        self.item_genres = {}


class AmazonElectronicsDataset(RecommenderDataset):
    """Amazon Electronics dataset"""
    
    def __init__(self, path: str = 'data/amazon-electronics',
                 test_size: float = 0.2, val_size: float = 0.1):
        super().__init__(name='amazon-electronics', path=path,
                        test_size=test_size, val_size=val_size)
    
    def _load_amazon_electronics(self):
        """Load Amazon Electronics dataset"""
        # Load ratings
        ratings_file = self.path / 'ratings_Electronics.csv'
        
        self.ratings = pd.read_csv(ratings_file, 
                                  names=['user', 'item', 'rating', 'timestamp'])
        
        # Convert to implicit
        self.ratings = self.ratings[self.ratings['rating'] >= 4.0]
        self.ratings['rating'] = 1.0
        
        # Reindex
        from sklearn.preprocessing import LabelEncoder
        
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        self.ratings['user'] = user_encoder.fit_transform(self.ratings['user']) + 1
        self.ratings['item'] = item_encoder.fit_transform(self.ratings['item']) + 1
        
        self.n_users = self.ratings['user'].max()
        self.n_items = self.ratings['item'].max()
        self.n_interactions = len(self.ratings)
        
        # Load item metadata if available
        self._load_item_metadata()
        
    def _load_item_metadata(self):
        """Load item categories"""
        meta_file = self.path / 'meta_Electronics.json'
        
        if meta_file.exists():
            categories = {}
            with open(meta_file, 'r') as f:
                for line in f:
                    item_data = json.loads(line)
                    if 'category' in item_data and 'asin' in item_data:
                        categories[item_data['asin']] = item_data['category']
            
            # Process categories
            all_cats = set()
            for cat_list in categories.values():
                all_cats.update(cat_list)
            
            self.genre_cols = sorted(list(all_cats))[:20]  # Top 20 categories
            self.n_genres = len(self.genre_cols)
            
            # Map to indices
            cat_to_idx = {cat: idx for idx, cat in enumerate(self.genre_cols)}
            
            self.item_genres = {}
            # Would need ASIN to item_id mapping here
        else:
            self.n_genres = 0
            self.genre_cols = []
            self.item_genres = {}