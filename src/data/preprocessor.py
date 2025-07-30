"""Data preprocessing utilities"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing for recommendation datasets
    
    Args:
        min_interactions: Minimum interactions for users/items
        implicit_threshold: Rating threshold for implicit feedback
    """
    
    def __init__(self, min_interactions: int = 5, implicit_threshold: float = 3.0):
        self.min_interactions = min_interactions
        self.implicit_threshold = implicit_threshold
        
        # Encoders for ID mapping
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
    def preprocess_ratings(self, df: pd.DataFrame, 
                          implicit: bool = True) -> pd.DataFrame:
        """Preprocess ratings dataframe
        
        Args:
            df: Raw ratings dataframe
            implicit: Whether to convert to implicit feedback
            
        Returns:
            Preprocessed dataframe
        """
        logger.info(f"Preprocessing {len(df)} ratings...")
        
        # Filter by minimum interactions
        df = self._filter_by_interactions(df)
        
        # Convert to implicit feedback if needed
        if implicit:
            df = self._convert_to_implicit(df)
            
        # Reindex users and items
        df = self._reindex_ids(df)
        
        logger.info(f"Preprocessed data: {len(df)} ratings, "
                   f"{df['user'].nunique()} users, {df['item'].nunique()} items")
        
        return df
    
    def _filter_by_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter users and items by minimum interactions
        
        Args:
            df: Ratings dataframe
            
        Returns:
            Filtered dataframe
        """
        # Iterative filtering
        while True:
            initial_size = len(df)
            
            # Filter users
            user_counts = df['user'].value_counts()
            valid_users = user_counts[user_counts >= self.min_interactions].index
            df = df[df['user'].isin(valid_users)]
            
            # Filter items
            item_counts = df['item'].value_counts()
            valid_items = item_counts[item_counts >= self.min_interactions].index
            df = df[df['item'].isin(valid_items)]
            
            # Check convergence
            if len(df) == initial_size:
                break
                
        logger.info(f"Filtered to {len(df)} ratings after k-core filtering")
        return df
    
    def _convert_to_implicit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert explicit ratings to implicit feedback
        
        Args:
            df: Ratings dataframe with 'rating' column
            
        Returns:
            Dataframe with implicit feedback
        """
        if 'rating' in df.columns:
            # Keep only positive feedback
            df = df[df['rating'] > self.implicit_threshold].copy()
            # Remove rating column for implicit feedback
            df['rating'] = 1.0
            
        return df
    
    def _reindex_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex user and item IDs to be consecutive
        
        Args:
            df: Ratings dataframe
            
        Returns:
            Dataframe with reindexed IDs
        """
        # Fit encoders
        self.user_encoder.fit(df['user'].unique())
        self.item_encoder.fit(df['item'].unique())
        
        # Transform IDs (add 1 to make them 1-indexed)
        df['user'] = self.user_encoder.transform(df['user']) + 1
        df['item'] = self.item_encoder.transform(df['item']) + 1
        
        return df
    
    def encode_features(self, features: pd.DataFrame, 
                       feature_cols: List[str]) -> np.ndarray:
        """Encode categorical features
        
        Args:
            features: Feature dataframe
            feature_cols: List of feature columns
            
        Returns:
            Encoded feature matrix
        """
        encoded_features = []
        
        for col in feature_cols:
            if features[col].dtype == 'object':
                # One-hot encode categorical features
                one_hot = pd.get_dummies(features[col], prefix=col)
                encoded_features.append(one_hot.values)
            else:
                # Keep numerical features as is
                encoded_features.append(features[col].values.reshape(-1, 1))
                
        return np.hstack(encoded_features)
    
    def split_temporal(self, df: pd.DataFrame, 
                      test_ratio: float = 0.2,
                      val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal split for train/val/test
        
        Args:
            df: Ratings dataframe with 'timestamp' column
            test_ratio: Test set ratio
            val_ratio: Validation set ratio
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate split points
        n_total = len(df)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_test - n_val
        
        # Split
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]
        
        return train_df, val_df, test_df
    
    def split_random(self, df: pd.DataFrame,
                    test_ratio: float = 0.2,
                    val_ratio: float = 0.1,
                    seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Random split for train/val/test
        
        Args:
            df: Ratings dataframe
            test_ratio: Test set ratio
            val_ratio: Validation set ratio
            seed: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Shuffle
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Calculate split points
        n_total = len(df)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_test - n_val
        
        # Split
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]
        
        return train_df, val_df, test_df
    
    def create_negative_samples(self, df: pd.DataFrame,
                               n_users: int, n_items: int,
                               n_neg_per_pos: int = 4) -> pd.DataFrame:
        """Create negative samples for training
        
        Args:
            df: Positive interactions dataframe
            n_users: Total number of users
            n_items: Total number of items
            n_neg_per_pos: Number of negative samples per positive
            
        Returns:
            Dataframe with negative samples added
        """
        # Build interaction set for fast lookup
        interaction_set = set(zip(df['user'], df['item']))
        
        negative_samples = []
        
        for _, row in df.iterrows():
            user = row['user']
            
            # Sample negative items
            neg_items = []
            while len(neg_items) < n_neg_per_pos:
                item = np.random.randint(1, n_items + 1)
                if (user, item) not in interaction_set:
                    neg_items.append(item)
                    
            # Add negative samples
            for item in neg_items:
                negative_samples.append({
                    'user': user,
                    'item': item,
                    'rating': 0,
                    'timestamp': row.get('timestamp', 0)
                })
                
        # Combine positive and negative samples
        neg_df = pd.DataFrame(negative_samples)
        combined_df = pd.concat([df, neg_df], ignore_index=True)
        
        return combined_df.sort_values(['user', 'timestamp']).reset_index(drop=True)