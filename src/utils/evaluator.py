"""Model evaluation utilities"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import logging

from .metrics import calculate_metrics, coverage, diversity

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for recommendation models
    
    Args:
        model: Recommendation model
        dataset: Dataset object
        device: Torch device
        batch_size: Batch size for evaluation
        adj: Adjacency matrix (for graph-based models)
    """
    
    def __init__(self, model, dataset, device: torch.device,
                 batch_size: int = 512, adj: Optional[torch.Tensor] = None):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.adj = adj
        
    def evaluate(self, user_list: Optional[List[int]] = None,
                 k_values: List[int] = [5, 10, 20],
                 split: str = 'test') -> Tuple[Dict, float, float]:
        """Evaluate model on given users
        
        Args:
            user_list: List of users to evaluate (None for all)
            k_values: List of k values for metrics
            split: Data split to evaluate on
            
        Returns:
            Tuple of (metrics_dict, coverage, diversity)
        """
        self.model.eval()
        
        # Get evaluation data
        if split == 'val':
            eval_dict = self.dataset.val_dict
        else:
            eval_dict = self.dataset.test_dict
            
        # Get users to evaluate
        if user_list is None:
            user_list = list(eval_dict.keys())
        else:
            # Filter users that have test data
            user_list = [u for u in user_list if u in eval_dict]
            
        if len(user_list) == 0:
            logger.warning("No users to evaluate")
            return {k: defaultdict(float) for k in k_values}, 0.0, 0.0
            
        # Initialize metrics
        metrics = {k: defaultdict(float) for k in k_values}
        all_recommendations = []
        coverage_items = set()
        
        # Evaluate in batches
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(user_list), self.batch_size),
                                 desc="Evaluating"):
                end_idx = min(start_idx + self.batch_size, len(user_list))
                batch_users = user_list[start_idx:end_idx]
                
                # Get predictions for batch
                batch_predictions = self._get_batch_predictions(batch_users)
                
                # Calculate metrics for each user
                for i, user in enumerate(batch_users):
                    predictions = batch_predictions[i]
                    ground_truth = eval_dict.get(user, [])
                    
                    if not ground_truth:
                        continue
                        
                    # Filter out training items
                    train_items = torch.where(self.dataset.train_mat[user] > 0)[0]
                    if len(train_items) > 0:
                        predictions[train_items - 1] = -float('inf')
                        
                    # Calculate metrics
                    user_metrics = calculate_metrics(predictions, ground_truth, k_values)
                    
                    # Aggregate metrics
                    for k in k_values:
                        for metric, value in user_metrics[k].items():
                            metrics[k][metric] += value
                            
                        # Get top-k items for coverage/diversity
                        _, top_k_indices = torch.topk(predictions, k)
                        top_k_items = (top_k_indices + 1).cpu().numpy()
                        all_recommendations.append(top_k_items.tolist())
                        coverage_items.update(top_k_items)
                        
        # Average metrics
        n_eval_users = len([u for u in user_list if eval_dict.get(u, [])])
        for k in k_values:
            for metric in metrics[k]:
                metrics[k][metric] /= n_eval_users if n_eval_users > 0 else 1
                
        # Calculate coverage and diversity
        coverage_score = coverage([list(coverage_items)], self.dataset.n_items)
        diversity_score = diversity(
            all_recommendations[:len(user_list)],  # Use first recommendation per user
            self.dataset.item_genres,
            self.dataset.n_genres
        )
        
        return metrics, coverage_score, diversity_score
    
    def _get_batch_predictions(self, users: List[int]) -> torch.Tensor:
        """Get predictions for a batch of users
        
        Args:
            users: List of user IDs
            
        Returns:
            Predictions tensor [batch_size, n_items]
        """
        users_tensor = torch.tensor(users, device=self.device)
        
        # Different methods for different models
        if hasattr(self.model, 'get_all_predictions'):
            # Models with efficient batch prediction
            if self.adj is not None:
                predictions = self.model.get_all_predictions(users_tensor, self.adj)
            else:
                predictions = self.model.get_all_predictions(users_tensor)
        else:
            # Fall back to item-by-item prediction
            predictions = []
            items = torch.arange(1, self.dataset.n_items + 1, device=self.device)
            
            for user in users:
                user_tensor = torch.full_like(items, user, device=self.device)
                
                if hasattr(self.model, 'predict'):
                    if self.adj is not None:
                        user_pred = self.model.predict(user_tensor, items, self.adj)
                    else:
                        user_pred = self.model.predict(user_tensor, items)
                else:
                    # Direct forward pass
                    user_pred = self.model(user_tensor, items)
                    
                predictions.append(user_pred)
                
            predictions = torch.stack(predictions)
            
        return predictions
    
    def get_user_predictions(self, user_id: int) -> torch.Tensor:
        """Get predictions for a single user
        
        Args:
            user_id: User ID
            
        Returns:
            Predictions for all items
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = self._get_batch_predictions([user_id])[0]
            
        return predictions
    
    def evaluate_cold_start(self, k_values: List[int] = [5, 10, 20]) -> Dict:
        """Evaluate on different user groups
        
        Args:
            k_values: List of k values for metrics
            
        Returns:
            Dictionary with metrics for each user group
        """
        results = {}
        
        user_groups = {
            'cold': self.dataset.cold_users,
            'warm': self.dataset.warm_users,
            'hot': self.dataset.hot_users
        }
        
        for group_name, user_list in user_groups.items():
            logger.info(f"Evaluating {group_name} users ({len(user_list)} users)")
            
            metrics, cov, div = self.evaluate(user_list, k_values)
            
            results[group_name] = {
                'metrics': metrics,
                'coverage': cov,
                'diversity': div,
                'n_users': len(user_list)
            }
            
        return results
    
    def sample_recommendations(self, n_users: int = 10, k: int = 10) -> Dict:
        """Sample recommendations for visualization
        
        Args:
            n_users: Number of users to sample
            k: Number of recommendations per user
            
        Returns:
            Dictionary with sampled recommendations
        """
        # Sample users from different groups
        sample_users = {
            'cold': np.random.choice(self.dataset.cold_users,
                                    min(n_users // 3, len(self.dataset.cold_users)),
                                    replace=False),
            'warm': np.random.choice(self.dataset.warm_users,
                                    min(n_users // 3, len(self.dataset.warm_users)),
                                    replace=False),
            'hot': np.random.choice(self.dataset.hot_users,
                                   min(n_users // 3, len(self.dataset.hot_users)),
                                   replace=False)
        }
        
        recommendations = {}
        
        for group_name, users in sample_users.items():
            group_recs = []
            
            for user_id in users:
                predictions = self.get_user_predictions(user_id)
                
                # Filter training items
                train_items = torch.where(self.dataset.train_mat[user_id] > 0)[0]
                if len(train_items) > 0:
                    predictions[train_items - 1] = -float('inf')
                    
                # Get top-k
                _, top_k_indices = torch.topk(predictions, k)
                top_k_items = (top_k_indices + 1).cpu().numpy()
                
                # Get item details
                rec_details = []
                for item_id in top_k_items:
                    item_info = {
                        'item_id': int(item_id),
                        'score': float(predictions[item_id - 1])
                    }
                    
                    # Add item metadata if available
                    if hasattr(self.dataset, 'items'):
                        item_data = self.dataset.items[self.dataset.items['item'] == item_id]
                        if not item_data.empty:
                            item_info['title'] = item_data.iloc[0].get('title', f'Item {item_id}')
                            
                    rec_details.append(item_info)
                    
                group_recs.append({
                    'user_id': int(user_id),
                    'recommendations': rec_details,
                    'ground_truth': self.dataset.test_dict.get(user_id, [])
                })
                
            recommendations[group_name] = group_recs
            
        return recommendations