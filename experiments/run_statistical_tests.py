#!/usr/bin/env python
"""
Run statistical significance tests on experiment results
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, nemenyi
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Run statistical tests on results')
    
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='results/statistical_tests',
                       help='Output directory for test results')
    parser.add_argument('--metric', type=str, default='test_ndcg',
                       choices=['test_ndcg', 'test_hr', 'cold_ndcg', 'cold_hr'],
                       help='Metric to test')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level')
    
    return parser.parse_args()


def load_results(results_dir: Path) -> dict:
    """Load experiment results from directory
    
    Args:
        results_dir: Directory containing results files
        
    Returns:
        Dictionary mapping model names to results
    """
    results = {}
    
    # Try to load aggregated results first
    aggregated_path = results_dir / 'aggregated_results.json'
    if aggregated_path.exists():
        with open(aggregated_path, 'r') as f:
            return json.load(f)
    
    # Otherwise load raw results
    raw_path = results_dir / 'raw_results.json'
    if raw_path.exists():
        with open(raw_path, 'r') as f:
            raw_results = json.load(f)
            
        # Convert to aggregated format
        for model_name, runs in raw_results.items():
            metrics_dict = {}
            
            # Extract metrics from runs
            for metric in ['test_ndcg', 'test_hr', 'cold_ndcg', 'cold_hr']:
                values = []
                for run in runs:
                    if metric.startswith('test_'):
                        key = metric.replace('test_', '')
                        value = run['best_metrics']['test_metrics'][10][key]
                    else:
                        key = metric.replace('cold_', '')
                        value = run['cold_results']['cold']['metrics'][10][key]
                    values.append(value)
                    
                metrics_dict[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
                
            results[model_name] = metrics_dict
            
        return results
    
    raise FileNotFoundError(f"No results found in {results_dir}")


def perform_pairwise_tests(results: dict, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    """Perform pairwise Wilcoxon signed-rank tests
    
    Args:
        results: Results dictionary
        metric: Metric to test
        alpha: Significance level
        
    Returns:
        DataFrame with test results
    """
    models = list(results.keys())
    n_models = len(models)
    
    # Initialize results matrix
    p_values = np.ones((n_models, n_models))
    statistics = np.zeros((n_models, n_models))
    
    # Perform pairwise tests
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i >= j:
                continue
                
            values_a = results[model_a][metric]['values']
            values_b = results[model_b][metric]['values']
            
            if len(values_a) > 1 and len(values_b) > 1:
                # Wilcoxon signed-rank test
                stat, p_value = wilcoxon(values_a, values_b)
                p_values[i, j] = p_value
                p_values[j, i] = p_value
                statistics[i, j] = stat
                statistics[j, i] = -stat
                
    # Create DataFrame
    df_p = pd.DataFrame(p_values, index=models, columns=models)
    df_stat = pd.DataFrame(statistics, index=models, columns=models)
    
    return df_p, df_stat


def perform_friedman_test(results: dict, metric: str) -> tuple:
    """Perform Friedman test for multiple comparisons
    
    Args:
        results: Results dictionary
        metric: Metric to test
        
    Returns:
        Tuple of (statistic, p_value)
    """
    # Prepare data for Friedman test
    data = []
    models = list(results.keys())
    
    # Get values for each model
    for model in models:
        values = results[model][metric]['values']
        data.append(values)
        
    # Ensure all have same length
    min_length = min(len(v) for v in data)
    data = [v[:min_length] for v in data]
    
    # Perform Friedman test
    if len(data) > 2:
        stat, p_value = friedmanchisquare(*data)
        return stat, p_value
    else:
        return None, None


def perform_nemenyi_test(results: dict, metric: str, alpha: float = 0.05) -> dict:
    """Perform Nemenyi post-hoc test
    
    Args:
        results: Results dictionary
        metric: Metric to test
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Prepare data
    models = list(results.keys())
    data = []
    
    for model in models:
        values = results[model][metric]['values']
        data.append(values)
        
    # Convert to ranks
    min_length = min(len(v) for v in data)
    data_array = np.array([v[:min_length] for v in data])
    
    # Compute average ranks
    ranks = np.zeros_like(data_array)
    for j in range(data_array.shape[1]):
        ranks[:, j] = stats.rankdata(-data_array[:, j])  # Negative for descending order
        
    avg_ranks = ranks.mean(axis=1)
    
    # Critical difference for Nemenyi test
    k = len(models)
    n = min_length
    
    # Using chi-square approximation
    chi2_alpha = stats.chi2.ppf(1 - alpha, k - 1)
    cd = np.sqrt((k * (k + 1)) / (6 * n)) * np.sqrt(chi2_alpha)
    
    return {
        'models': models,
        'avg_ranks': avg_ranks,
        'critical_difference': cd
    }


def create_significance_heatmap(p_values_df: pd.DataFrame, alpha: float = 0.05,
                               output_path: Path = None):
    """Create heatmap of statistical significance
    
    Args:
        p_values_df: DataFrame with p-values
        alpha: Significance level
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(p_values_df, dtype=bool))
    
    # Create custom colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(p_values_df, mask=mask, cmap=cmap, center=alpha,
                annot=True, fmt='.4f', square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Pairwise Statistical Significance (p-values)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    # Add significance level line
    plt.axhline(y=len(p_values_df) - 0.5, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_critical_difference_diagram(nemenyi_results: dict, output_path: Path = None):
    """Create critical difference diagram
    
    Args:
        nemenyi_results: Results from Nemenyi test
        output_path: Path to save figure
    """
    models = nemenyi_results['models']
    avg_ranks = nemenyi_results['avg_ranks']
    cd = nemenyi_results['critical_difference']
    
    # Sort by average rank
    sorted_indices = np.argsort(avg_ranks)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot ranks
    y_pos = np.arange(len(sorted_models))
    ax.barh(y_pos, sorted_ranks, color='skyblue', alpha=0.8)
    
    # Add critical difference line
    best_rank = sorted_ranks[0]
    ax.axvline(x=best_rank + cd, color='red', linestyle='--', 
               label=f'Critical Difference = {cd:.3f}')
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models)
    ax.set_xlabel('Average Rank', fontsize=12)
    ax.set_title('Critical Difference Diagram (Nemenyi Test)', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add rank values
    for i, (model, rank) in enumerate(zip(sorted_models, sorted_ranks)):
        ax.text(rank + 0.05, i, f'{rank:.2f}', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_report(results: dict, tests_results: dict, output_path: Path):
    """Generate statistical test report
    
    Args:
        results: Original results
        tests_results: Test results
        output_path: Path to save report
    """
    report = []
    report.append("# Statistical Significance Test Report\n")
    report.append(f"Date: {pd.Timestamp.now()}\n")
    report.append(f"Metric: {tests_results['metric']}\n")
    report.append(f"Significance Level: α = {tests_results['alpha']}\n")
    
    # Summary statistics
    report.append("\n## Summary Statistics\n")
    report.append("| Model | Mean | Std | Min | Max | N |\n")
    report.append("|-------|------|-----|-----|-----|---|\n")
    
    for model, metrics in results.items():
        values = metrics[tests_results['metric']]['values']
        report.append(f"| {model} | {np.mean(values):.4f} | {np.std(values):.4f} | "
                     f"{np.min(values):.4f} | {np.max(values):.4f} | {len(values)} |\n")
    
    # Friedman test results
    if tests_results['friedman']['p_value'] is not None:
        report.append("\n## Friedman Test\n")
        report.append(f"- Statistic: {tests_results['friedman']['statistic']:.4f}\n")
        report.append(f"- p-value: {tests_results['friedman']['p_value']:.4f}\n")
        report.append(f"- Significant: {'Yes' if tests_results['friedman']['p_value'] < tests_results['alpha'] else 'No'}\n")
    
    # Pairwise comparisons
    report.append("\n## Pairwise Wilcoxon Tests\n")
    p_values_df = tests_results['pairwise']['p_values']
    
    significant_pairs = []
    for i in range(len(p_values_df)):
        for j in range(i+1, len(p_values_df)):
            model_a = p_values_df.index[i]
            model_b = p_values_df.columns[j]
            p_value = p_values_df.iloc[i, j]
            
            if p_value < tests_results['alpha']:
                significant_pairs.append((model_a, model_b, p_value))
    
    if significant_pairs:
        report.append("\n### Significant Differences:\n")
        for model_a, model_b, p_value in significant_pairs:
            report.append(f"- {model_a} vs {model_b}: p = {p_value:.4f}\n")
    else:
        report.append("\nNo significant pairwise differences found.\n")
    
    # Nemenyi test results
    if 'nemenyi' in tests_results:
        report.append("\n## Nemenyi Post-hoc Test\n")
        report.append(f"Critical Difference: {tests_results['nemenyi']['critical_difference']:.3f}\n")
        report.append("\nAverage Ranks:\n")
        
        for model, rank in zip(tests_results['nemenyi']['models'], 
                              tests_results['nemenyi']['avg_ranks']):
            report.append(f"- {model}: {rank:.2f}\n")
    
    # Save report
    with open(output_path, 'w') as f:
        f.writelines(report)


def main():
    args = parse_args()
    
    # Set up logger
    logger = setup_logger('statistical_tests', 
                         log_file=Path(args.output_dir) / 'statistical_tests.log')
    
    logger.info(f"Running statistical tests with parameters: {args}")
    
    # Load results
    logger.info(f"Loading results from {args.results_dir}")
    results = load_results(Path(args.results_dir))
    
    if not results:
        logger.error("No results found!")
        return
    
    logger.info(f"Loaded results for models: {list(results.keys())}")
    
    # Initialize test results
    test_results = {
        'metric': args.metric,
        'alpha': args.alpha
    }
    
    # Perform pairwise tests
    logger.info("Performing pairwise Wilcoxon tests...")
    p_values_df, statistics_df = perform_pairwise_tests(results, args.metric, args.alpha)
    test_results['pairwise'] = {
        'p_values': p_values_df,
        'statistics': statistics_df
    }
    
    # Perform Friedman test
    if len(results) > 2:
        logger.info("Performing Friedman test...")
        friedman_stat, friedman_p = perform_friedman_test(results, args.metric)
        test_results['friedman'] = {
            'statistic': friedman_stat,
            'p_value': friedman_p
        }
        
        # If Friedman test is significant, perform post-hoc tests
        if friedman_p < args.alpha:
            logger.info("Friedman test significant, performing Nemenyi post-hoc test...")
            nemenyi_results = perform_nemenyi_test(results, args.metric, args.alpha)
            test_results['nemenyi'] = nemenyi_results
    else:
        test_results['friedman'] = {'statistic': None, 'p_value': None}
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw test results
    with open(output_dir / 'test_results.json', 'w') as f:
        # Convert DataFrame to dict for JSON serialization
        save_results = test_results.copy()
        save_results['pairwise']['p_values'] = test_results['pairwise']['p_values'].to_dict()
        save_results['pairwise']['statistics'] = test_results['pairwise']['statistics'].to_dict()
        json.dump(save_results, f, indent=2)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Significance heatmap
    create_significance_heatmap(
        p_values_df, 
        args.alpha,
        output_dir / 'significance_heatmap.png'
    )
    
    # Critical difference diagram
    if 'nemenyi' in test_results:
        create_critical_difference_diagram(
            test_results['nemenyi'],
            output_dir / 'critical_difference.png'
        )
    
    # Generate report
    logger.info("Generating report...")
    generate_report(results, test_results, output_dir / 'statistical_report.md')
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("STATISTICAL TEST SUMMARY")
    logger.info("="*70)
    logger.info(f"Metric: {args.metric}")
    logger.info(f"Significance level: α = {args.alpha}")
    
    if test_results['friedman']['p_value'] is not None:
        logger.info(f"\nFriedman test p-value: {test_results['friedman']['p_value']:.4f}")
        
    # Find best performing model
    mean_values = {model: np.mean(metrics[args.metric]['values']) 
                  for model, metrics in results.items()}
    best_model = max(mean_values, key=mean_values.get)
    
    logger.info(f"\nBest performing model: {best_model} ({mean_values[best_model]:.4f})")
    
    # Report significant differences
    significant_wins = defaultdict(int)
    for i in range(len(p_values_df)):
        for j in range(i+1, len(p_values_df)):
            if p_values_df.iloc[i, j] < args.alpha:
                model_a = p_values_df.index[i]
                model_b = p_values_df.columns[j]
                
                # Determine winner
                if mean_values[model_a] > mean_values[model_b]:
                    significant_wins[model_a] += 1
                else:
                    significant_wins[model_b] += 1
    
    if significant_wins:
        logger.info("\nSignificant wins:")
        for model, wins in sorted(significant_wins.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {model}: {wins} wins")
    
    logger.info(f"\nAll results saved to {output_dir}")
    logger.info("Statistical testing completed!")


if __name__ == '__main__':
    main()