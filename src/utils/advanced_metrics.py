"""Advanced evaluation metrics for diversity and novelty"""

import numpy as np
import torch
from typing import List, Dict, Set
from collections import Counter
import math


def intra_list_diversity(recommendations: List[int], 
                        item_features: Dict[int, List[int]]) -> float:
    """Calculate Intra-List Diversity (ILD)
    
    Average dissimilarity between all pairs of items in a recommendation list
    
    Args:
        recommendations: List of recommended items
        item_features: Item feature vectors
        
    Returns:
        ILD score
    """
    if len(recommendations) < 2:
        return 0.0
    
    total_distance = 0
    n_pairs = 0
    
    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            item_i = recommendations[i]
            item_j = recommendations[j]
            
            # Calculate Jaccard distance
            if item_i in item_features and item_j in item_features:
                features_i = set(item_features[item_i])
                features_j = set(item_features[item_j])
                
                if len(features_i | features_j) > 0:
                    jaccard_sim = len(features_i & features_j) / len(features_i | features_j)
                    distance = 1 - jaccard_sim
                else:
                    distance = 1.0
            else:
                distance = 1.0
                
            total_distance += distance
            n_pairs += 1
    
    return total_distance / n_pairs if n_pairs > 0 else 0.0


def aggregate_diversity(all_recommendations: List[List[int]]) -> float:
    """Calculate Aggregate Diversity
    
    Total number of distinct items recommended across all users
    
    Args:
        all_recommendations: Recommendations for all users
        
    Returns:
        Number of unique items recommended
    """
    unique_items = set()
    for rec_list in all_recommendations:
        unique_items.update(rec_list)
    
    return len(unique_items)


def novelty(recommendations: List[int], item_popularity: Dict[int, float]) -> float:
    """Calculate novelty based on item popularity
    
    Args:
        recommendations: List of recommended items
        item_popularity: Item popularity scores (e.g., interaction counts)
        
    Returns:
        Average novelty score
    """
    if not recommendations:
        return 0.0
    
    novelty_scores = []
    max_popularity = max(item_popularity.values()) if item_popularity else 1.0
    
    for item in recommendations:
        popularity = item_popularity.get(item, 0)
        # Novelty is inverse of popularity
        novelty_score = -np.log2((popularity + 1) / (max_popularity + 1))
        novelty_scores.append(novelty_score)
    
    return np.mean(novelty_scores)


def serendipity(recommendations: List[int], ground_truth: List[int],
                expected_items: List[int]) -> float:
    """Calculate serendipity
    
    Fraction of recommended items that are both relevant and unexpected
    
    Args:
        recommendations: Recommended items
        ground_truth: Relevant items
        expected_items: Expected/obvious recommendations
        
    Returns:
        Serendipity score
    """
    if not recommendations:
        return 0.0
    
    relevant_items = set(ground_truth)
    expected_set = set(expected_items)
    
    serendipitous_items = 0
    for item in recommendations:
        if item in relevant_items and item not in expected_set:
            serendipitous_items += 1
    
    return serendipitous_items / len(recommendations)


def long_tail_coverage(recommendations: List[List[int]], 
                      item_popularity: Dict[int, int],
                      tail_threshold: float = 0.8) -> float:
    """Calculate long-tail item coverage
    
    Args:
        recommendations: All recommendations
        item_popularity: Item interaction counts
        tail_threshold: Percentile threshold for long-tail
        
    Returns:
        Fraction of long-tail items covered
    """
    # Identify long-tail items
    sorted_items = sorted(item_popularity.items(), key=lambda x: x[1])
    n_tail = int(len(sorted_items) * tail_threshold)
    tail_items = set([item for item, _ in sorted_items[:n_tail]])
    
    # Count recommended tail items
    recommended_items = set()
    for rec_list in recommendations:
        recommended_items.update(rec_list)
    
    recommended_tail = recommended_items & tail_items
    
    return len(recommended_tail) / len(tail_items) if tail_items else 0.0


def gini_coefficient(recommendations: List[List[int]]) -> float:
    """Calculate Gini coefficient for recommendation distribution
    
    Measures inequality in item recommendation frequency
    Lower values indicate more equal distribution
    
    Args:
        recommendations: All recommendation lists
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    # Count item frequencies
    item_counts = Counter()
    for rec_list in recommendations:
        item_counts.update(rec_list)
    
    if not item_counts:
        return 0.0
    
    # Get frequency values
    frequencies = list(item_counts.values())
    n = len(frequencies)
    
    # Sort frequencies
    frequencies.sort()
    
    # Calculate Gini
    cumsum = 0
    for i, freq in enumerate(frequencies):
        cumsum += (n - i) * freq
    
    return 1 - 2 * cumsum / (n * sum(frequencies))


def personalization(recommendations: List[List[int]]) -> float:
    """Calculate personalization (1 - average overlap between users)
    
    Args:
        recommendations: Recommendations for all users
        
    Returns:
        Personalization score
    """
    n_users = len(recommendations)
    if n_users < 2:
        return 1.0
    
    total_overlap = 0
    n_pairs = 0
    
    for i in range(n_users):
        for j in range(i + 1, n_users):
            set_i = set(recommendations[i])
            set_j = set(recommendations[j])
            
            if len(set_i) > 0 and len(set_j) > 0:
                overlap = len(set_i & set_j) / min(len(set_i), len(set_j))
                total_overlap += overlap
                n_pairs += 1
    
    avg_overlap = total_overlap / n_pairs if n_pairs > 0 else 0
    return 1 - avg_overlap


def entropy_diversity(recommendations: List[int], n_items: int) -> float:
    """Calculate entropy-based diversity
    
    Args:
        recommendations: Single recommendation list
        n_items: Total number of items in catalog
        
    Returns:
        Normalized entropy
    """
    if not recommendations or n_items == 0:
        return 0.0
    
    # Count unique items
    unique_items = len(set(recommendations))
    
    # Maximum possible entropy
    max_entropy = np.log2(min(len(recommendations), n_items))
    
    # Actual entropy (assuming uniform distribution over recommended items)
    if unique_items > 0:
        actual_entropy = np.log2(unique_items)
    else:
        actual_entropy = 0.0
    
    return actual_entropy / max_entropy if max_entropy > 0 else 0.0


def calibration_error(recommendations: List[int], 
                     user_profile: Dict[int, float],
                     item_categories: Dict[int, List[int]]) -> float:
    """Calculate calibration error
    
    Measures how well recommendations match user's historical preferences
    
    Args:
        recommendations: Recommended items
        user_profile: User's category preferences (distribution)
        item_categories: Item to category mapping
        
    Returns:
        Calibration error (lower is better)
    """
    if not recommendations:
        return 0.0
    
    # Calculate recommendation category distribution
    rec_categories = Counter()
    for item in recommendations:
        if item in item_categories:
            for cat in item_categories[item]:
                rec_categories[cat] += 1
    
    # Normalize to distribution
    total = sum(rec_categories.values())
    if total == 0:
        return 1.0
    
    rec_distribution = {cat: count/total for cat, count in rec_categories.items()}
    
    # Calculate KL divergence
    kl_div = 0.0
    for cat, p_user in user_profile.items():
        p_rec = rec_distribution.get(cat, 1e-10)
        if p_user > 0:
            kl_div += p_user * np.log(p_user / p_rec)
    
    return kl_div