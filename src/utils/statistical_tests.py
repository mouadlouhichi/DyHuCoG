"""Statistical significance tests and effect size calculations"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict
import pandas as pd


def paired_t_test(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    """Perform paired t-test
    
    Args:
        scores1: First method scores
        scores2: Second method scores
        
    Returns:
        Tuple of (t_statistic, p_value)
    """
    return stats.ttest_rel(scores1, scores2)


def wilcoxon_test(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test
    
    Args:
        scores1: First method scores
        scores2: Second method scores
        
    Returns:
        Tuple of (statistic, p_value)
    """
    return stats.wilcoxon(scores1, scores2)


def cohen_d(scores1: List[float], scores2: List[float]) -> float:
    """Calculate Cohen's d effect size
    
    Args:
        scores1: First method scores
        scores2: Second method scores
        
    Returns:
        Cohen's d value
    """
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    
    # Pooled standard deviation
    n1 = len(scores1)
    n2 = len(scores2)
    var1 = np.var(scores1, ddof=1)
    var2 = np.var(scores2, ddof=1)
    
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0


def cliff_delta(scores1: List[float], scores2: List[float]) -> float:
    """Calculate Cliff's delta effect size
    
    Non-parametric effect size measure
    
    Args:
        scores1: First method scores
        scores2: Second method scores
        
    Returns:
        Cliff's delta value
    """
    n1 = len(scores1)
    n2 = len(scores2)
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    # Count dominance
    dominance = 0
    for s1 in scores1:
        for s2 in scores2:
            if s1 > s2:
                dominance += 1
            elif s1 < s2:
                dominance -= 1
    
    return dominance / (n1 * n2)


def bootstrap_confidence_interval(scores: List[float], 
                                confidence: float = 0.95,
                                n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval
    
    Args:
        scores: Method scores
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    scores = np.array(scores)
    n = len(scores)
    
    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return lower, upper


def friedman_test(scores_dict: Dict[str, List[float]]) -> Tuple[float, float]:
    """Perform Friedman test for multiple methods
    
    Args:
        scores_dict: Dictionary mapping method names to scores
        
    Returns:
        Tuple of (statistic, p_value)
    """
    # Convert to array format
    methods = list(scores_dict.keys())
    scores_array = []
    
    for method in methods:
        scores_array.append(scores_dict[method])
    
    return stats.friedmanchisquare(*scores_array)


def nemenyi_posthoc(scores_dict: Dict[str, List[float]], 
                   alpha: float = 0.05) -> pd.DataFrame:
    """Perform Nemenyi post-hoc test
    
    Args:
        scores_dict: Dictionary mapping method names to scores
        alpha: Significance level
        
    Returns:
        DataFrame with pairwise p-values
    """
    from scikit_posthocs import posthoc_nemenyi_friedman
    
    # Create data matrix
    methods = list(scores_dict.keys())
    n_samples = len(next(iter(scores_dict.values())))
    
    data = np.zeros((n_samples, len(methods)))
    for i, method in enumerate(methods):
        data[:, i] = scores_dict[method]
    
    # Perform test
    result = posthoc_nemenyi_friedman(data)
    result.index = methods
    result.columns = methods
    
    return result


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Bonferroni correction for multiple comparisons
    
    Args:
        p_values: List of p-values
        alpha: Original significance level
        
    Returns:
        List of boolean values indicating significance
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    
    return [p < adjusted_alpha for p in p_values]


def calculate_all_metrics_with_stats(results_dict: Dict[str, Dict[str, List[float]]],
                                   baseline: str = 'lightgcn') -> pd.DataFrame:
    """Calculate all statistical comparisons
    
    Args:
        results_dict: Nested dict {method: {metric: [scores]}}
        baseline: Baseline method name
        
    Returns:
        DataFrame with statistical comparisons
    """
    rows = []
    
    for method in results_dict:
        if method == baseline:
            continue
            
        for metric in results_dict[method]:
            baseline_scores = results_dict[baseline][metric]
            method_scores = results_dict[method][metric]
            
            # Calculate statistics
            mean_baseline = np.mean(baseline_scores)
            mean_method = np.mean(method_scores)
            std_baseline = np.std(baseline_scores)
            std_method = np.std(method_scores)
            
            # Statistical tests
            _, p_value_t = paired_t_test(method_scores, baseline_scores)
            _, p_value_w = wilcoxon_test(method_scores, baseline_scores)
            
            # Effect sizes
            d = cohen_d(method_scores, baseline_scores)
            delta = cliff_delta(method_scores, baseline_scores)
            
            # Confidence intervals
            ci_baseline = bootstrap_confidence_interval(baseline_scores)
            ci_method = bootstrap_confidence_interval(method_scores)
            
            # Relative improvement
            improvement = (mean_method - mean_baseline) / mean_baseline * 100
            
            rows.append({
                'Method': method,
                'Metric': metric,
                'Mean': f"{mean_method:.4f} ± {std_method:.4f}",
                'Baseline_Mean': f"{mean_baseline:.4f} ± {std_baseline:.4f}",
                'Improvement': f"{improvement:+.1f}%",
                'p_value_t': p_value_t,
                'p_value_w': p_value_w,
                'Cohen_d': f"{d:.3f}",
                'Cliff_delta': f"{delta:.3f}",
                'CI_95': f"[{ci_method[0]:.4f}, {ci_method[1]:.4f}]",
                'Significant': '✓' if p_value_w < 0.05 else ''
            })
    
    return pd.DataFrame(rows)