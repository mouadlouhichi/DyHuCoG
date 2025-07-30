#!/usr/bin/env python
"""
Data preprocessing script for recommendation datasets
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess recommendation dataset')
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name or path to raw data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--min_interactions', type=int, default=5,
                       help='Minimum interactions for users/items')
    parser.add_argument('--implicit_threshold', type=float, default=3.0,
                       help='Rating threshold for implicit feedback')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--split_method', type=str, default='temporal',
                       choices=['temporal', 'random'],
                       help='Data split method')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def preprocess_movielens(data_path: Path, args: argparse.Namespace, logger):
    """Preprocess MovieLens dataset"""
    logger.info("Preprocessing MovieLens dataset...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        min_interactions=args.min_interactions,
        implicit_threshold=args.implicit_threshold
    )
    
    # Load ratings
    if 'ml-100k' in str(data_path):
        ratings = pd.read_csv(
            data_path / 'u.data',
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp'],
            engine='python'
        )
    elif 'ml-1m' in str(data_path):
        ratings = pd.read_csv(
            data_path / 'ratings.dat',
            sep='::',
            names=['user', 'item', 'rating', 'timestamp'],
            engine='python'
        )
    else:
        raise ValueError(f"Unknown MovieLens version: {data_path}")
        
    logger.info(f"Loaded {len(ratings)} ratings")
    
    # Preprocess ratings
    ratings = preprocessor.preprocess_ratings(ratings, implicit=True)
    
    # Split data
    if args.split_method == 'temporal':
        train_df, val_df, test_df = preprocessor.split_temporal(
            ratings, args.test_ratio, args.val_ratio
        )
    else:
        train_df, val_df, test_df = preprocessor.split_random(
            ratings, args.test_ratio, args.val_ratio, args.seed
        )
        
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save processed data
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    # Save statistics
    stats = {
        'n_users': ratings['user'].max(),
        'n_items': ratings['item'].max(),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'density': len(ratings) / (ratings['user'].max() * ratings['item'].max()),
        'min_interactions': args.min_interactions,
        'implicit_threshold': args.implicit_threshold
    }
    
    pd.Series(stats).to_csv(output_dir / 'stats.csv')
    
    logger.info(f"Preprocessed data saved to {output_dir}")
    
    return output_dir


def preprocess_amazon(data_path: Path, args: argparse.Namespace, logger):
    """Preprocess Amazon dataset"""
    logger.info("Preprocessing Amazon dataset...")
    
    # Load ratings (assuming standard format)
    ratings = pd.read_csv(
        data_path / 'ratings.csv',
        names=['user', 'item', 'rating', 'timestamp']
    )
    
    # Similar preprocessing as MovieLens
    preprocessor = DataPreprocessor(
        min_interactions=args.min_interactions,
        implicit_threshold=args.implicit_threshold
    )
    
    ratings = preprocessor.preprocess_ratings(ratings, implicit=True)
    
    # Split and save
    if args.split_method == 'temporal':
        train_df, val_df, test_df = preprocessor.split_temporal(
            ratings, args.test_ratio, args.val_ratio
        )
    else:
        train_df, val_df, test_df = preprocessor.split_random(
            ratings, args.test_ratio, args.val_ratio, args.seed
        )
        
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    logger.info(f"Preprocessed data saved to {output_dir}")
    
    return output_dir


def generate_negative_samples(data_path: Path, n_neg_per_pos: int = 4):
    """Generate negative samples for training"""
    logger = setup_logger('negative_sampling')
    logger.info(f"Generating negative samples with ratio 1:{n_neg_per_pos}")
    
    # Load processed data
    train_df = pd.read_csv(data_path / 'train.csv')
    stats = pd.read_csv(data_path / 'stats.csv', index_col=0).squeeze("columns")
    
    n_users = int(stats['n_users'])
    n_items = int(stats['n_items'])
    
    # Build interaction set
    interaction_set = set(zip(train_df['user'], train_df['item']))
    
    negative_samples = []
    
    for _, row in train_df.iterrows():
        user = row['user']
        
        # Sample negative items
        neg_items = []
        while len(neg_items) < n_neg_per_pos:
            item = np.random.randint(1, n_items + 1)
            if (user, item) not in interaction_set:
                neg_items.append(item)
                
        for item in neg_items:
            negative_samples.append({
                'user': user,
                'item': item,
                'rating': 0,
                'timestamp': row.get('timestamp', 0)
            })
            
    # Save negative samples
    neg_df = pd.DataFrame(negative_samples)
    neg_df.to_csv(data_path / 'negative_samples.csv', index=False)
    
    logger.info(f"Generated {len(neg_df)} negative samples")


def main():
    args = parse_args()
    
    # Set up logger
    logger = setup_logger(
        'preprocess',
        log_file=Path('logs') / 'preprocess.log'
    )
    
    logger.info(f"Preprocessing arguments: {args}")
    
    # Determine dataset path
    if args.dataset in ['ml-100k', 'ml-1m']:
        data_path = Path('data') / args.dataset
    elif args.dataset.startswith('amazon-'):
        data_path = Path('data') / args.dataset
    else:
        data_path = Path(args.dataset)
        
    if not data_path.exists():
        logger.error(f"Dataset path not found: {data_path}")
        logger.info("Please download the dataset first using scripts/download_data.sh")
        return
        
    # Preprocess based on dataset type
    if 'ml-' in args.dataset:
        output_dir = preprocess_movielens(data_path, args, logger)
    elif 'amazon' in args.dataset:
        output_dir = preprocess_amazon(data_path, args, logger)
    else:
        logger.error(f"Unknown dataset type: {args.dataset}")
        return
        
    # Generate negative samples
    generate_negative_samples(output_dir)
    
    logger.info("Preprocessing completed!")


if __name__ == '__main__':
    main()