"""Tests for data loading and preprocessing"""

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import RecommenderDataset
from src.data.dataloader import BPRDataset, get_dataloader
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def create_dummy_data():
    """Create dummy dataset files for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy MovieLens-like data
        data_dir = Path(tmpdir) / 'ml-test'
        data_dir.mkdir()
        
        # Create ratings data
        n_users = 50
        n_items = 30
        n_ratings = 500
        
        ratings_data = []
        for _ in range(n_ratings):
            user = np.random.randint(1, n_users + 1)
            item = np.random.randint(1, n_items + 1)
            rating = np.random.randint(1, 6)
            timestamp = np.random.randint(1000000, 2000000)
            ratings_data.append([user, item, rating, timestamp])
            
        ratings_df = pd.DataFrame(ratings_data, 
                                 columns=['user', 'item', 'rating', 'timestamp'])
        ratings_df.to_csv(data_dir / 'u.data', sep='\t', index=False, header=False)
        
        # Create item data with genres
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
        items_data = []
        
        for item_id in range(1, n_items + 1):
            title = f"Movie {item_id}"
            # Random genres (binary encoding)
            genre_vector = np.random.randint(0, 2, len(genres))
            row = [item_id, title, '', '', ''] + genre_vector.tolist()
            items_data.append(row)
            
        items_df = pd.DataFrame(items_data)
        items_df.to_csv(data_dir / 'u.item', sep='|', index=False, header=False)
        
        yield str(data_dir)


class TestDataset:
    """Test dataset loading"""
    
    def test_dataset_loading(self, create_dummy_data):
        """Test loading dummy dataset"""
        # This would require modifying RecommenderDataset to handle test data
        # For now, we'll test the structure
        data_path = Path(create_dummy_data)
        
        assert (data_path / 'u.data').exists()
        assert (data_path / 'u.item').exists()
        
        # Load ratings
        ratings = pd.read_csv(
            data_path / 'u.data',
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp']
        )
        
        assert len(ratings) > 0
        assert 'user' in ratings.columns
        assert 'item' in ratings.columns
        
    def test_data_split(self):
        """Test train/val/test split logic"""
        # Create dummy ratings
        n_ratings = 1000
        ratings = pd.DataFrame({
            'user': np.random.randint(1, 101, n_ratings),
            'item': np.random.randint(1, 51, n_ratings),
            'rating': np.random.randint(1, 6, n_ratings),
            'timestamp': np.arange(n_ratings)
        })
        
        # Test temporal split
        test_size = 0.2
        val_size = 0.1
        
        train_size = int(n_ratings * (1 - test_size - val_size))
        val_end = int(n_ratings * (1 - test_size))
        
        train_ratings = ratings.iloc[:train_size]
        val_ratings = ratings.iloc[train_size:val_end]
        test_ratings = ratings.iloc[val_end:]
        
        assert len(train_ratings) + len(val_ratings) + len(test_ratings) == n_ratings
        assert len(test_ratings) / n_ratings == pytest.approx(test_size, rel=0.01)
        assert len(val_ratings) / n_ratings == pytest.approx(val_size, rel=0.01)
        
    def test_user_groups(self):
        """Test user group identification"""
        # Create interaction counts
        user_counts = pd.Series({
            1: 2,    # Cold user
            2: 10,   # Warm user
            3: 30,   # Hot user
            4: 4,    # Cold user
            5: 15    # Warm user
        })
        
        cold_users = []
        warm_users = []
        hot_users = []
        
        for user, count in user_counts.items():
            if count < 5:
                cold_users.append(user)
            elif count < 20:
                warm_users.append(user)
            else:
                hot_users.append(user)
                
        assert cold_users == [1, 4]
        assert warm_users == [2, 5]
        assert hot_users == [3]


class TestDataLoader:
    """Test data loading utilities"""
    
    def test_bpr_dataset(self):
        """Test BPR dataset"""
        # Create dummy dataset
        class DummyDataset:
            def __init__(self):
                self.n_users = 100
                self.n_items = 50
                self.train_mat = torch.zeros(101, 51)  # 1-indexed
                
                # Add some interactions
                self.train_ratings = pd.DataFrame({
                    'user': [1, 1, 2, 2, 3],
                    'item': [1, 2, 2, 3, 1]
                })
                
                for _, row in self.train_ratings.iterrows():
                    self.train_mat[row['user'], row['item']] = 1
                    
        dataset = DummyDataset()
        bpr_dataset = BPRDataset(dataset)
        
        assert len(bpr_dataset) == len(dataset.train_ratings)
        
        # Test sampling
        sample = bpr_dataset[0]
        
        assert 'users' in sample
        assert 'pos_items' in sample
        assert 'neg_items' in sample
        
        # Check that negative item is indeed negative
        user = sample['users'].item()
        neg_item = sample['neg_items'].item()
        assert dataset.train_mat[user, neg_item] == 0
        
    def test_dataloader_creation(self):
        """Test dataloader creation"""
        # Create dummy dataset
        class DummyDataset:
            def __init__(self):
                self.n_users = 100
                self.n_items = 50
                self.train_mat = torch.zeros(101, 51)
                self.train_ratings = pd.DataFrame({
                    'user': list(range(1, 101)),
                    'item': list(range(1, 51)) + list(range(1, 51))
                })
                self.val_dict = {i: [i] for i in range(1, 21)}
                self.test_dict = {i: [i] for i in range(1, 21)}
                
        dataset = DummyDataset()
        config = {
            'training': {
                'batch_size': 32,
                'num_workers': 0
            },
            'evaluation': {
                'eval_batch_size': 64
            }
        }
        
        # Test train dataloader
        train_loader = get_dataloader(dataset, 'train', config)
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        
        # Get a batch
        batch = next(iter(train_loader))
        assert batch['users'].shape[0] <= config['training']['batch_size']


class TestPreprocessor:
    """Test data preprocessing"""
    
    def test_filtering(self):
        """Test interaction filtering"""
        # Create dummy ratings
        ratings = pd.DataFrame({
            'user': [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4],
            'item': [1, 2, 3, 1, 2, 1, 1, 2, 3, 4, 5],
            'rating': [5] * 11
        })
        
        preprocessor = DataPreprocessor(min_interactions=3)
        
        # Count interactions before filtering
        initial_users = ratings['user'].nunique()
        initial_items = ratings['item'].nunique()
        
        # Apply filtering
        filtered = preprocessor._filter_by_interactions(ratings)
        
        # Check that users/items with < 3 interactions are removed
        assert filtered['user'].nunique() < initial_users
        assert 2 not in filtered['user'].values  # User 2 had only 2 interactions
        assert 3 not in filtered['user'].values  # User 3 had only 1 interaction
        
    def test_implicit_conversion(self):
        """Test conversion to implicit feedback"""
        ratings = pd.DataFrame({
            'user': [1, 1, 2, 2],
            'item': [1, 2, 1, 2],
            'rating': [5, 3, 2, 4]
        })
        
        preprocessor = DataPreprocessor(implicit_threshold=3.0)
        implicit = preprocessor._convert_to_implicit(ratings)
        
        # Only ratings > 3 should remain
        assert len(implicit) == 2
        assert all(implicit['rating'] == 1.0)
        
    def test_reindexing(self):
        """Test ID reindexing"""
        # Create ratings with non-consecutive IDs
        ratings = pd.DataFrame({
            'user': [10, 10, 20, 20],
            'item': [100, 200, 100, 300]
        })
        
        preprocessor = DataPreprocessor()
        reindexed = preprocessor._reindex_ids(ratings)
        
        # Check that IDs are now consecutive starting from 1
        assert reindexed['user'].min() == 1
        assert reindexed['user'].max() == 2
        assert reindexed['item'].min() == 1
        assert reindexed['item'].max() == 3
        
    def test_temporal_split(self):
        """Test temporal data splitting"""
        # Create ratings with timestamps
        n_ratings = 1000
        ratings = pd.DataFrame({
            'user': np.random.randint(1, 101, n_ratings),
            'item': np.random.randint(1, 51, n_ratings),
            'rating': np.random.randint(1, 6, n_ratings),
            'timestamp': np.arange(n_ratings)
        })
        
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.split_temporal(ratings, test_ratio=0.2, val_ratio=0.1)
        
        # Check split sizes
        assert len(train) + len(val) + len(test) == n_ratings
        assert len(test) / n_ratings == pytest.approx(0.2, rel=0.01)
        assert len(val) / n_ratings == pytest.approx(0.1, rel=0.01)
        
        # Check temporal ordering
        assert train['timestamp'].max() <= val['timestamp'].min()
        assert val['timestamp'].max() <= test['timestamp'].min()


if __name__ == "__main__":
    pytest.main([__file__])