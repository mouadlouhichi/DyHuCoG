"""Evaluation metrics for recommendation systems"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Set
import math


def calculate_metrics(predictions: torch.Tensor, ground_truth: List[int], 
                     k_values: List[int]) -> Dict[int, Dict[str, float]]:
    """Calculate all metrics for given k values
    
    Args:
        predictions: Predicted scores for all items
        ground_truth: List of ground truth item IDs
        k_values: List of k values for metrics
        
    Returns:
        Dictionary mapping k to metrics
    """
    if not ground_truth:
        return {k: {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0, 'hit_rate': 0.0} 
                for k in k_values}
    
    # Get top-k predictions
    _, top_k_indices = torch.topk(predictions, max(k_values))
    top_k_items = (top_k_indices + 1).cpu().numpy()  # Convert to 1-indexed
    
    metrics = {}
    for k in k_values:
        top_k = top_k_items[:k]
        
        metrics[k] = {
            'precision': precision_at_k(top_k, ground_truth),
            'recall': recall_at_k(top_k, ground_truth),
            'ndcg': ndcg_at_k(top_k, ground_truth),
            'hit_rate': hit_rate_at_k(top_k, ground_truth)
        }
        
    return metrics


def precision_at_k(predictions: np.ndarray, ground_truth: List[int]) -> float:
    """Calculate Precision@k
    
    Args:
        predictions: Top-k predicted items
        ground_truth: Ground truth items
        
    Returns:
        Precision@k value
    """
    if len(predictions) == 0:
        return 0.0
        
    hits = len(set(predictions) & set(ground_truth))
    return hits / len(predictions)


def recall_at_k(predictions: np.ndarray, ground_truth: List[int]) -> float:
    """Calculate Recall@k
    
    Args:
        predictions: Top-k predicted items
        ground_truth: Ground truth items
        
    Returns:
        Recall@k value
    """
    if len(ground_truth) == 0:
        return 0.0
        
    hits = len(set(predictions) & set(ground_truth))
    return hits / len(ground_truth)


def ndcg_at_k(predictions: np.ndarray, ground_truth: List[int]) -> float:
    """Calculate NDCG@k
    
    Args:
        predictions: Top-k predicted items
        ground_truth: Ground truth items
        
    Returns:
        NDCG@k value
    """
    if len(ground_truth) == 0:
        return 0.0
        
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(predictions):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because i starts from 0
            
    # Calculate IDCG
    idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(ground_truth), len(predictions)))])
    
    if idcg == 0:
        return 0.0
        
    return dcg / idcg


def hit_rate_at_k(predictions: np.ndarray, ground_truth: List[int]) -> float:
    """Calculate Hit Rate@k (1 if any hit, 0 otherwise)
    
    Args:
        predictions: Top-k predicted items
        ground_truth: Ground truth items
        
    Returns:
        Hit rate (0 or 1)
    """
    return 1.0 if len(set(predictions) & set(ground_truth)) > 0 else 0.0


def coverage(recommendations: List[List[int]], n_items: int) -> float:
    """Calculate catalog coverage
    
    Args:
        recommendations: List of recommendation lists
        n_items: Total number of items
        
    Returns:
        Coverage ratio
    """
    recommended_items = set()
    for rec_list in recommendations:
        recommended_items.update(rec_list)
        
    return len(recommended_items) / n_items


def diversity(recommendations: List[List[int]], item_features: Dict[int, List[int]],
              n_features: int) -> float:
    """Calculate diversity based on item features
    
    Args:
        recommendations: List of recommendation lists
        item_features: Dictionary mapping items to feature indices
        n_features: Total number of features
        
    Returns:
        Average diversity score
    """
    diversity_scores = []
    
    for rec_list in recommendations:
        if len(rec_list) == 0:
            continue
            
        # Get all features in this recommendation list
        features = set()
        for item in rec_list:
            if item in item_features:
                features.update(item_features[item])
                
        # Diversity is the ratio of unique features
        diversity_score = len(features) / n_features if n_features > 0 else 0
        diversity_scores.append(diversity_score)
        
    return np.mean(diversity_scores) if diversity_scores else 0.0


def mean_average_precision(recommendations: List[List[int]], 
                          ground_truth: Dict[int, List[int]]) -> float:
    """Calculate Mean Average Precision
    
    Args:
        recommendations: List of recommendation lists
        ground_truth: Dictionary of ground truth items per user
        
    Returns:
        MAP value
    """
    ap_scores = []
    
    for user_id, rec_list in enumerate(recommendations):
        if user_id not in ground_truth or len(ground_truth[user_id]) == 0:
            continue
            
        # Calculate Average Precision for this user
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(rec_list):
            if item in ground_truth[user_id]:
                hits += 1
                sum_precisions += hits / (i + 1)
                
        ap = sum_precisions / len(ground_truth[user_id])
        ap_scores.append(ap)
        
    return np.mean(ap_scores) if ap_scores else 0.0


def auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Calculate AUC (Area Under ROC Curve)
    
    Args:
        positive_scores: Scores for positive items
        negative_scores: Scores for negative items
        
    Returns:
        AUC value
    """
    # Count correct rankings
    correct = 0
    total = len(positive_scores) * len(negative_scores)
    
    for pos_score in positive_scores:
        for neg_score in negative_scores:
            if pos_score > neg_score:
                correct += 1
            elif pos_score == neg_score:
                correct += 0.5
                
    return correct / total if total > 0 else 0.5