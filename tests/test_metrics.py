"""Tests for evaluation metrics"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k,
    calculate_metrics, coverage, diversity, mean_average_precision, auc
)


class TestBasicMetrics:
    """Test individual metric functions"""
    
    def test_precision_at_k(self):
        """Test Precision@k calculation"""
        # Test case 1: Perfect prediction
        predictions = np.array([1, 2, 3, 4, 5])
        ground_truth = [1, 2, 3, 4, 5]
        assert precision_at_k(predictions, ground_truth) == 1.0
        
        # Test case 2: Half correct
        predictions = np.array([1, 2, 6, 7, 8])
        ground_truth = [1, 2, 3, 4, 5]
        assert precision_at_k(predictions, ground_truth) == 0.4
        
        # Test case 3: No correct predictions
        predictions = np.array([6, 7, 8, 9, 10])
        ground_truth = [1, 2, 3, 4, 5]
        assert precision_at_k(predictions, ground_truth) == 0.0
        
        # Test case 4: Empty predictions
        predictions = np.array([])
        ground_truth = [1, 2, 3]
        assert precision_at_k(predictions, ground_truth) == 0.0
        
    def test_recall_at_k(self):
        """Test Recall@k calculation"""
        # Test case 1: All ground truth items retrieved
        predictions = np.array([1, 2, 3, 4, 5, 6, 7])
        ground_truth = [1, 2, 3]
        assert recall_at_k(predictions, ground_truth) == 1.0
        
        # Test case 2: Half of ground truth retrieved
        predictions = np.array([1, 2, 6, 7, 8])
        ground_truth = [1, 2, 3, 4]
        assert recall_at_k(predictions, ground_truth) == 0.5
        
        # Test case 3: Empty ground truth
        predictions = np.array([1, 2, 3])
        ground_truth = []
        assert recall_at_k(predictions, ground_truth) == 0.0
        
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation"""
        # Test case 1: Perfect ranking
        predictions = np.array([1, 2, 3])
        ground_truth = [1, 2, 3]
        assert ndcg_at_k(predictions, ground_truth) == 1.0
        
        # Test case 2: Imperfect ranking
        predictions = np.array([3, 1, 2])
        ground_truth = [1, 2, 3]
        ndcg = ndcg_at_k(predictions, ground_truth)
        assert 0 < ndcg < 1.0
        
        # Test case 3: Only first item correct
        predictions = np.array([1, 4, 5])
        ground_truth = [1, 2, 3]
        ndcg = ndcg_at_k(predictions, ground_truth)
        # DCG = 1/log2(2) = 1, IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
        expected_dcg = 1.0
        expected_idcg = 1.0 + 1/np.log2(3) + 1/np.log2(4)
        expected_ndcg = expected_dcg / expected_idcg
        assert ndcg == pytest.approx(expected_ndcg, rel=1e-6)
        
    def test_hit_rate_at_k(self):
        """Test Hit Rate@k calculation"""
        # Test case 1: Hit
        predictions = np.array([1, 2, 3])
        ground_truth = [3, 4, 5]
        assert hit_rate_at_k(predictions, ground_truth) == 1.0
        
        # Test case 2: No hit
        predictions = np.array([1, 2, 3])
        ground_truth = [4, 5, 6]
        assert hit_rate_at_k(predictions, ground_truth) == 0.0


class TestCalculateMetrics:
    """Test combined metrics calculation"""
    
    def test_calculate_metrics(self):
        """Test calculate_metrics function"""
        # Create dummy predictions
        predictions = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        ground_truth = [1, 3, 5]  # Items 1, 3, 5 are relevant (1-indexed)
        k_values = [3, 5, 10]
        
        metrics = calculate_metrics(predictions, ground_truth, k_values)
        
        # Check structure
        assert set(metrics.keys()) == set(k_values)
        for k in k_values:
            assert 'precision' in metrics[k]
            assert 'recall' in metrics[k]
            assert 'ndcg' in metrics[k]
            assert 'hit_rate' in metrics[k]
            
        # Check values make sense
        # Top-3 predictions are indices 0, 1, 2 (items 1, 2, 3)
        # Ground truth has items 1, 3, 5
        assert metrics[3]['precision'] == 2/3  # Items 1, 3 are in top-3
        assert metrics[3]['recall'] == 2/3     # 2 out of 3 ground truth items
        assert metrics[3]['hit_rate'] == 1.0   # At least one hit
        
        # Check that metrics increase with k
        assert metrics[3]['recall'] <= metrics[5]['recall']
        assert metrics[5]['recall'] <= metrics[10]['recall']
        
    def test_calculate_metrics_empty_ground_truth(self):
        """Test metrics calculation with empty ground truth"""
        predictions = torch.rand(10)
        ground_truth = []
        k_values = [5, 10]
        
        metrics = calculate_metrics(predictions, ground_truth, k_values)
        
        for k in k_values:
            assert metrics[k]['precision'] == 0.0
            assert metrics[k]['recall'] == 0.0
            assert metrics[k]['ndcg'] == 0.0
            assert metrics[k]['hit_rate'] == 0.0


class TestDiversityMetrics:
    """Test diversity and coverage metrics"""
    
    def test_coverage(self):
        """Test catalog coverage calculation"""
        # Test case 1: All items recommended
        recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        n_items = 10
        assert coverage(recommendations, n_items) == 1.0
        
        # Test case 2: Half of items recommended
        recommendations = [[1, 2], [1, 3], [2, 3], [1, 2, 3]]
        n_items = 6
        assert coverage(recommendations, n_items) == 0.5
        
        # Test case 3: No recommendations
        recommendations = [[], [], []]
        n_items = 10
        assert coverage(recommendations, n_items) == 0.0
        
    def test_diversity(self):
        """Test diversity calculation"""
        # Create item features (genres)
        item_features = {
            1: [0, 1],      # Action, Comedy
            2: [0],         # Action
            3: [1, 2],      # Comedy, Drama
            4: [2],         # Drama
            5: [0, 1, 2]    # All genres
        }
        n_features = 3
        
        # Test case 1: Diverse recommendations
        recommendations = [[1, 3, 5]]  # Covers all 3 genres
        div = diversity(recommendations, item_features, n_features)
        assert div == 1.0
        
        # Test case 2: Less diverse recommendations
        recommendations = [[1, 2]]  # Only covers genres 0 and 1
        div = diversity(recommendations, item_features, n_features)
        assert div == pytest.approx(2/3, rel=1e-6)
        
        # Test case 3: Multiple recommendation lists
        recommendations = [[1, 2], [3, 4], [5]]
        div = diversity(recommendations, item_features, n_features)
        # First list: genres 0,1 (2/3)
        # Second list: genres 1,2 (2/3)
        # Third list: genres 0,1,2 (3/3)
        expected = (2/3 + 2/3 + 3/3) / 3
        assert div == pytest.approx(expected, rel=1e-6)


class TestRankingMetrics:
    """Test ranking-specific metrics"""
    
    def test_mean_average_precision(self):
        """Test MAP calculation"""
        recommendations = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [1, 3, 5, 7, 9]
        ]
        
        ground_truth = {
            0: [1, 3],      # User 0: items 1 and 3 are relevant
            1: [6, 10],     # User 1: items 6 and 10 are relevant
            2: [2, 4, 6]    # User 2: items 2, 4, 6 are relevant
        }
        
        map_score = mean_average_precision(recommendations, ground_truth)
        
        # Calculate expected MAP
        # User 0: hits at positions 1,3 -> AP = (1/1 + 2/3)/2 = 0.833
        # User 1: hits at positions 1,5 -> AP = (1/1 + 2/5)/2 = 0.7
        # User 2: no hits in recommendations -> AP = 0
        expected_map = (0.833 + 0.7 + 0) / 3
        
        assert map_score == pytest.approx(expected_map, rel=0.01)
        
    def test_auc(self):
        """Test AUC calculation"""
        # Test case 1: Perfect separation
        positive_scores = np.array([0.9, 0.8, 0.7])
        negative_scores = np.array([0.3, 0.2, 0.1])
        assert auc(positive_scores, negative_scores) == 1.0
        
        # Test case 2: Random scores
        positive_scores = np.array([0.5, 0.5, 0.5])
        negative_scores = np.array([0.5, 0.5, 0.5])
        assert auc(positive_scores, negative_scores) == 0.5
        
        # Test case 3: Reverse separation (worst case)
        positive_scores = np.array([0.1, 0.2, 0.3])
        negative_scores = np.array([0.7, 0.8, 0.9])
        assert auc(positive_scores, negative_scores) == 0.0
        
        # Test case 4: Mixed scores
        positive_scores = np.array([0.9, 0.5, 0.3])
        negative_scores = np.array([0.8, 0.4, 0.2])
        # Expected: 5 correct rankings out of 9 comparisons
        assert auc(positive_scores, negative_scores) == pytest.approx(5/9, rel=1e-6)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_inputs(self):
        """Test metrics with empty inputs"""
        # Empty predictions
        assert precision_at_k(np.array([]), [1, 2, 3]) == 0.0
        assert recall_at_k(np.array([]), [1, 2, 3]) == 0.0
        assert ndcg_at_k(np.array([]), [1, 2, 3]) == 0.0
        
        # Empty ground truth
        assert precision_at_k(np.array([1, 2, 3]), []) == 0.0
        assert recall_at_k(np.array([1, 2, 3]), []) == 0.0
        assert ndcg_at_k(np.array([1, 2, 3]), []) == 0.0
        
    def test_single_item(self):
        """Test metrics with single item"""
        predictions = np.array([1])
        ground_truth = [1]
        
        assert precision_at_k(predictions, ground_truth) == 1.0
        assert recall_at_k(predictions, ground_truth) == 1.0
        assert ndcg_at_k(predictions, ground_truth) == 1.0
        assert hit_rate_at_k(predictions, ground_truth) == 1.0
        
    def test_large_k(self):
        """Test metrics when k > number of items"""
        predictions = torch.rand(100)
        ground_truth = list(range(1, 11))  # 10 relevant items
        k_values = [5, 10, 50, 100, 200]
        
        metrics = calculate_metrics(predictions, ground_truth, k_values)
        
        # Recall should plateau when k >= number of items
        assert metrics[100]['recall'] == metrics[200]['recall']
        
        # Precision should decrease as k increases beyond relevant items
        assert metrics[5]['precision'] >= metrics[50]['precision']


if __name__ == "__main__":
    pytest.main([__file__])