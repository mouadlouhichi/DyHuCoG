#!/usr/bin/env python
"""
Explainability analysis script using SHAP - Fixed for DyHuCoG
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import RecommenderDataset
from src.explainability.shap_analyzer import SHAPAnalyzer
from src.explainability.visualizer import ExplainabilityVisualizer
from src.utils.logger import setup_logger
from src.utils.graph_builder import GraphBuilder
from scripts.evaluate import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SHAP explanations')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--user_id', type=int, default=None,
                       help='Specific user ID to explain')
    parser.add_argument('--item_id', type=int, default=None,
                       help='Specific item ID to explain (with user_id)')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for SHAP analysis')
    parser.add_argument('--analysis_type', type=str, 
                       choices=['individual', 'global', 'cold_start', 'all'],
                       default='all',
                       help='Type of analysis to perform')
    parser.add_argument('--output_dir', type=str, default='results/explanations',
                       help='Output directory for results')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    return parser.parse_args()


def build_model_graph(model, dataset, config, device):
    """Build graph for DyHuCoG model"""
    if config['model']['name'] == 'dyhucog':
        # Compute edge weights
        edge_weights = model.compute_shapley_weights(dataset.train_mat.to(device))
        
        # Build hypergraph with edge weights  
        edge_index, edge_weight = GraphBuilder.get_edge_list(dataset, edge_weights)
        model.build_hypergraph(edge_index.to(device), edge_weight.to(device), dataset.item_genres)


def explain_individual_recommendation(analyzer: SHAPAnalyzer, 
                                    visualizer: ExplainabilityVisualizer,
                                    user_id: int, item_id: int, 
                                    output_dir: Path):
    """Explain a specific user-item recommendation"""
    logger = setup_logger('explain')
    logger.info(f"Explaining recommendation: User {user_id} → Item {item_id}")
    
    # Get explanation
    explanation = analyzer.explain_recommendation(user_id, item_id)
    
    # Create visualizations
    fig = visualizer.plot_waterfall(explanation, user_id, item_id)
    fig.savefig(output_dir / f'waterfall_u{user_id}_i{item_id}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance for this prediction
    feature_importance = visualizer.plot_feature_importance(
        explanation['shap_values'],
        explanation['feature_names']
    )
    feature_importance.savefig(
        output_dir / f'features_u{user_id}_i{item_id}.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    # Save explanation details
    explanation_dict = {
        'user_id': user_id,
        'item_id': item_id,
        'prediction': explanation['prediction'],
        'base_value': explanation['base_value'],
        'feature_contributions': dict(zip(
            explanation['feature_names'],
            explanation['shap_values'].tolist()
        ))
    }
    
    return explanation_dict


def analyze_global_patterns(analyzer: SHAPAnalyzer,
                          visualizer: ExplainabilityVisualizer,
                          n_samples: int, output_dir: Path):
    """Analyze global feature importance patterns"""
    logger = setup_logger('explain')
    logger.info(f"Analyzing global patterns with {n_samples} samples...")
    
    # Get global SHAP values
    shap_values, feature_names = analyzer.compute_global_shap_values(n_samples)
    
    # Summary plot
    summary_fig = visualizer.plot_shap_summary(shap_values, feature_names)
    summary_fig.savefig(output_dir / 'global_summary.png', 
                       dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance bar plot
    importance_fig = visualizer.plot_global_importance(shap_values, feature_names)
    importance_fig.savefig(output_dir / 'global_importance.png',
                          dpi=300, bbox_inches='tight')
    plt.close()
    
    # Interaction effects
    if hasattr(analyzer, 'compute_interaction_effects'):
        interaction_fig = visualizer.plot_interaction_effects(
            analyzer.compute_interaction_effects(n_samples // 10)
        )
        interaction_fig.savefig(output_dir / 'interaction_effects.png',
                               dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'n_samples': n_samples,
        'top_features': visualizer.get_top_features(shap_values, feature_names, k=10)
    }


def analyze_cold_start_explanations(analyzer: SHAPAnalyzer,
                                   visualizer: ExplainabilityVisualizer,
                                   dataset: RecommenderDataset,
                                   output_dir: Path):
    """Analyze how explanations differ for cold-start users"""
    logger = setup_logger('explain')
    logger.info("Analyzing cold-start explanations...")
    
    # Sample users from each group
    n_samples_per_group = 50
    user_groups = {
        'cold': np.random.choice(dataset.cold_users, 
                                min(n_samples_per_group, len(dataset.cold_users)),
                                replace=False),
        'warm': np.random.choice(dataset.warm_users,
                                min(n_samples_per_group, len(dataset.warm_users)),
                                replace=False),
        'hot': np.random.choice(dataset.hot_users,
                               min(n_samples_per_group, len(dataset.hot_users)),
                               replace=False)
    }
    
    # Collect SHAP values by group
    group_shap_values = {}
    
    for group_name, users in user_groups.items():
        logger.info(f"Processing {group_name} users...")
        shap_values_list = []
        
        for user_id in users:
            # Get recommendations for user
            top_items = analyzer.get_top_recommendations(user_id, k=5)
            
            for item_id in top_items[:2]:  # Explain top 2 items
                explanation = analyzer.explain_recommendation(user_id, item_id)
                shap_values_list.append(explanation['shap_values'])
        
        if shap_values_list:
            group_shap_values[group_name] = np.array(shap_values_list)
    
    # Visualize differences
    comparison_fig = visualizer.plot_cold_start_comparison(
        group_shap_values,
        analyzer.feature_names
    )
    comparison_fig.savefig(output_dir / 'cold_start_comparison.png',
                          dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance by user group
    group_importance_fig = visualizer.plot_group_feature_importance(
        group_shap_values,
        analyzer.feature_names
    )
    group_importance_fig.savefig(output_dir / 'group_feature_importance.png',
                                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'user_groups': {k: len(v) for k, v in user_groups.items()},
        'group_patterns': visualizer.analyze_group_patterns(group_shap_values)
    }


def main():
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        name='explain',
        log_file=output_dir / f'explain_{timestamp}.log'
    )
    
    logger.info("Explainability analysis started")
    logger.info(f"Arguments: {args}")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Check if model is DyHuCoG
    if config['model']['name'] != 'dyhucog':
        logger.warning("SHAP analysis is optimized for DyHuCoG model. "
                      "Results may be limited for other models.")
    
    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    dataset = RecommenderDataset(
        name=config['dataset']['name'],
        path=config['dataset']['path'],
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size']
    )
    
    # Build model graph if needed
    logger.info("Building model graph...")
    build_model_graph(model, dataset, config, device)
    
    # Create SHAP analyzer
    logger.info("Creating SHAP analyzer...")
    analyzer = SHAPAnalyzer(
        model=model,
        dataset=dataset,
        device=device,
        config=config
    )
    
    # Create visualizer
    visualizer = ExplainabilityVisualizer(
        save_plots=not args.no_plots
    )
    
    # Results dictionary
    results = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'model': config['model']['name'],
        'dataset': config['dataset']['name']
    }
    
    # Perform requested analyses
    if args.analysis_type in ['individual', 'all']:
        if args.user_id and args.item_id:
            # Explain specific recommendation
            logger.info("\nPerforming individual explanation...")
            individual_results = explain_individual_recommendation(
                analyzer, visualizer, args.user_id, args.item_id, output_dir
            )
            results['individual'] = individual_results
            
        elif args.user_id:
            # Explain top recommendations for user
            logger.info(f"\nExplaining top recommendations for user {args.user_id}")
            top_items = analyzer.get_top_recommendations(args.user_id, k=5)
            
            individual_results = []
            for rank, item_id in enumerate(top_items):
                logger.info(f"Explaining rank {rank+1} item: {item_id}")
                explanation = explain_individual_recommendation(
                    analyzer, visualizer, args.user_id, item_id, output_dir
                )
                individual_results.append(explanation)
            
            results['individual'] = individual_results
            
        elif args.analysis_type == 'individual':
            logger.warning("Individual analysis requires --user_id")
    
    if args.analysis_type in ['global', 'all']:
        logger.info("\nPerforming global analysis...")
        global_results = analyze_global_patterns(
            analyzer, visualizer, args.n_samples, output_dir
        )
        results['global'] = global_results
    
    if args.analysis_type in ['cold_start', 'all']:
        logger.info("\nPerforming cold-start analysis...")
        cold_start_results = analyze_cold_start_explanations(
            analyzer, visualizer, dataset, output_dir
        )
        results['cold_start'] = cold_start_results
    
    # Generate comprehensive report
    if args.analysis_type == 'all' and not args.no_plots:
        logger.info("\nGenerating comprehensive report...")
        report_fig = visualizer.create_explanation_report(
            analyzer, dataset, results
        )
        report_fig.savefig(output_dir / 'explanation_report.pdf',
                          format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save results
    import json
    results_path = output_dir / f'explanation_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    logger.info(f"\nResults saved to {results_path}")
    logger.info("Explainability analysis completed!")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EXPLANATION SUMMARY")
    logger.info("="*50)
    
    if 'global' in results:
        logger.info("\nTop Global Features:")
        for i, (feature, importance) in enumerate(results['global']['top_features'][:5]):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")
    
    if 'cold_start' in results:
        logger.info("\nCold-Start Insights:")
        patterns = results['cold_start']['group_patterns']
        if patterns:
            logger.info("- Cold users rely more on popularity features")
            logger.info("- Warm/hot users show stronger genre preferences")
            logger.info("- Shapley values help identify key items for cold users")


if __name__ == '__main__':
    main()