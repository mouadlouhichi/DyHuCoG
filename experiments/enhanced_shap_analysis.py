#!/usr/bin/env python
"""
Enhanced SHAP analysis for DyHuCoG model
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import networkx as nx
from sklearn.ensemble import RandomForestRegressor

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.dyhucog import DyHuCoG
from src.data.dataset import RecommenderDataset
from src.explainability.shap_analyzer import SHAPAnalyzer
from src.explainability.visualizer import ExplainabilityVisualizer
from src.utils.logger import setup_logger
from scripts.evaluate import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced SHAP analysis')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset name')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for SHAP analysis')
    parser.add_argument('--output_dir', type=str, default='results/enhanced_shap',
                       help='Output directory')
    
    return parser.parse_args()


class EnhancedSHAPAnalyzer:
    """Enhanced SHAP analyzer with additional visualizations"""
    
    def __init__(self, model, dataset, device, config):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.config = config
        
        # Initialize base SHAP analyzer
        self.base_analyzer = SHAPAnalyzer(model, dataset, device, config)
        self.visualizer = ExplainabilityVisualizer(save_plots=True)
        
    def visualize_shapley_graph_contribution(self, user_id, top_k=10, output_path=None):
        """Visualize how Shapley values affect the recommendation graph"""
        plt.figure(figsize=(15, 10))
        
        # Get user's items and their Shapley values
        user_items = self.dataset.train_mat[user_id].nonzero().squeeze()
        if len(user_items.shape) == 0:
            user_items = user_items.unsqueeze(0)
        
        # Get Shapley values
        with torch.no_grad():
            user_item_vector = self.dataset.train_mat[user_id]
            shapley_values = self.model.shapley_net(user_item_vector.unsqueeze(0)).squeeze()
        
        # Create graph
        G = nx.Graph()
        
        # Add user node
        G.add_node(f"U{user_id}", node_type='user', pos=(0, 0))
        
        # Add item nodes with Shapley values
        item_positions = {}
        shapley_dict = {}
        
        for idx, item_idx in enumerate(user_items[:top_k]):
            item_id = item_idx.item() + 1
            shapley_val = shapley_values[item_idx].item()
            shapley_dict[item_id] = shapley_val
            
            # Get item info
            if hasattr(self.dataset, 'items'):
                item_info = self.dataset.items[self.dataset.items['item'] == item_id].iloc[0]
                item_title = item_info['title'][:20] + "..." if len(item_info['title']) > 20 else item_info['title']
            else:
                item_title = f"Item {item_id}"
            
            # Position items in a circle
            angle = 2 * np.pi * idx / min(len(user_items), top_k)
            x, y = 3 * np.cos(angle), 3 * np.sin(angle)
            item_positions[item_id] = (x, y)
            
            G.add_node(f"I{item_id}", 
                      node_type='item',
                      title=item_title,
                      shapley=shapley_val,
                      pos=(x, y))
            
            # Add edge with Shapley weight
            edge_weight = self.model.edge_weights.get((user_id, item_id), 1.0)
            G.add_edge(f"U{user_id}", f"I{item_id}", weight=edge_weight, shapley=shapley_val)
        
        # Add genre nodes
        for idx, (item_id, pos) in enumerate(item_positions.items()):
            genres = self.dataset.item_genres.get(item_id, [])
            for genre_idx in genres:
                genre_name = self.dataset.genre_cols[genre_idx]
                if f"G{genre_name}" not in G:
                    # Position genres on outer circle
                    angle = 2 * np.pi * genre_idx / self.dataset.n_genres
                    x, y = 5 * np.cos(angle), 5 * np.sin(angle)
                    G.add_node(f"G{genre_name}", 
                              node_type='genre',
                              name=genre_name,
                              pos=(x, y))
                
                G.add_edge(f"I{item_id}", f"G{genre_name}", weight=0.5)
        
        # Draw graph
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes by type
        user_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'user']
        item_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'item']
        genre_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'genre']
        
        # Node colors based on Shapley values
        item_colors = [G.nodes[n]['shapley'] for n in item_nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='red', 
                              node_size=1000, node_shape='s', label='User')
        
        nodes = nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, node_color=item_colors,
                                      node_size=800, cmap='RdYlGn', vmin=-0.5, vmax=0.5,
                                      label='Items')
        
        nx.draw_networkx_nodes(G, pos, nodelist=genre_nodes, node_color='lightblue',
                              node_size=600, node_shape='^', label='Genres')
        
        # Draw edges with varying widths based on weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights], alpha=0.6)
        
        # Labels
        labels = {}
        for node, data in G.nodes(data=True):
            if data['node_type'] == 'user':
                labels[node] = f"User\n{user_id}"
            elif data['node_type'] == 'item':
                labels[node] = f"{data['title']}\nφ={data['shapley']:.3f}"
            else:
                labels[node] = data['name']
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add colorbar
        plt.colorbar(nodes, label='Shapley Value', orientation='horizontal', pad=0.1)
        
        plt.title(f"Recommendation Graph for User {user_id}\nwith Shapley Value Contributions")
        plt.axis('off')
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def compare_shapley_methods(self, n_samples=100, output_path=None):
        """Compare FastSHAP approximation with traditional SHAP"""
        print("Comparing Shapley value computation methods...")
        
        comparison_results = []
        
        for _ in range(n_samples):
            user_id = np.random.randint(1, self.dataset.n_users + 1)
            
            # Get user's items
            user_items = self.dataset.train_mat[user_id]
            item_indices = user_items.nonzero().squeeze()
            
            if len(item_indices) == 0:
                continue
            
            # FastSHAP approximation
            with torch.no_grad():
                fastshap_values = self.model.shapley_net(user_items.unsqueeze(0)).squeeze()
            
            # SHAP library approximation
            if self.base_analyzer.shap_explainer is not None:
                for item_idx in item_indices[:5]:  # Limit to 5 items
                    item_id = item_idx.item() + 1
                    
                    # Get SHAP explanation
                    explanation = self.base_analyzer.explain_recommendation(user_id, item_id)
                    shap_vals = explanation['shap_values']
                    
                    # Extract item-specific SHAP values
                    item_shap_sum = np.sum(shap_vals[self.dataset.n_genres+2:])
                    
                    comparison_results.append({
                        'user': user_id,
                        'item': item_id,
                        'fastshap': fastshap_values[item_idx].item(),
                        'shap_lib': item_shap_sum
                    })
        
        # Plot comparison
        if comparison_results:
            df_comp = pd.DataFrame(comparison_results)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(df_comp['fastshap'], df_comp['shap_lib'], alpha=0.6)
            plt.xlabel('FastSHAP Approximation')
            plt.ylabel('SHAP Library Values')
            plt.title('Comparison of Shapley Value Methods')
            
            # Add correlation
            corr = df_comp['fastshap'].corr(df_comp['shap_lib'])
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, verticalalignment='top')
            
            # Add diagonal line
            min_val = min(df_comp['fastshap'].min(), df_comp['shap_lib'].min())
            max_val = max(df_comp['fastshap'].max(), df_comp['shap_lib'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
    def analyze_cold_start_shapley(self, n_cold_users=50, output_path=None):
        """Analyze how Shapley values help with cold-start users"""
        print("Analyzing Shapley values for cold-start mitigation...")
        
        cold_results = []
        warm_results = []
        
        # Sample cold and warm users
        cold_sample = np.random.choice(self.dataset.cold_users, 
                                      min(n_cold_users, len(self.dataset.cold_users)))
        warm_sample = np.random.choice(self.dataset.warm_users, 
                                      min(n_cold_users, len(self.dataset.warm_users)))
        
        with torch.no_grad():
            # Analyze cold users
            for user_id in cold_sample:
                user_items = self.dataset.train_mat[user_id]
                if user_items.sum() > 0:
                    shapley_values = self.model.shapley_net(user_items.unsqueeze(0)).squeeze()
                    
                    # Get non-zero Shapley values
                    nonzero_shapley = shapley_values[user_items > 0]
                    if len(nonzero_shapley) > 0:
                        cold_results.append({
                            'user': user_id,
                            'n_items': int(user_items.sum()),
                            'mean_shapley': nonzero_shapley.mean().item(),
                            'std_shapley': nonzero_shapley.std().item() if len(nonzero_shapley) > 1 else 0,
                            'max_shapley': nonzero_shapley.max().item()
                        })
            
            # Analyze warm users
            for user_id in warm_sample:
                user_items = self.dataset.train_mat[user_id]
                if user_items.sum() > 0:
                    shapley_values = self.model.shapley_net(user_items.unsqueeze(0)).squeeze()
                    
                    nonzero_shapley = shapley_values[user_items > 0]
                    if len(nonzero_shapley) > 0:
                        warm_results.append({
                            'user': user_id,
                            'n_items': int(user_items.sum()),
                            'mean_shapley': nonzero_shapley.mean().item(),
                            'std_shapley': nonzero_shapley.std().item() if len(nonzero_shapley) > 1 else 0,
                            'max_shapley': nonzero_shapley.max().item()
                        })
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if cold_results and warm_results:
            df_cold = pd.DataFrame(cold_results)
            df_warm = pd.DataFrame(warm_results)
            
            # 1. Mean Shapley values distribution
            axes[0, 0].hist(df_cold['mean_shapley'], bins=20, alpha=0.5, label='Cold Users', color='blue')
            axes[0, 0].hist(df_warm['mean_shapley'], bins=20, alpha=0.5, label='Warm Users', color='red')
            axes[0, 0].set_xlabel('Mean Shapley Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Mean Shapley Values')
            axes[0, 0].legend()
            
            # 2. Shapley vs Number of Items
            axes[0, 1].scatter(df_cold['n_items'], df_cold['mean_shapley'], 
                              alpha=0.6, label='Cold Users', color='blue')
            axes[0, 1].scatter(df_warm['n_items'], df_warm['mean_shapley'], 
                              alpha=0.6, label='Warm Users', color='red')
            axes[0, 1].set_xlabel('Number of Items')
            axes[0, 1].set_ylabel('Mean Shapley Value')
            axes[0, 1].set_title('Shapley Values vs User Activity')
            axes[0, 1].legend()
            
            # 3. Shapley standard deviation
            axes[1, 0].hist(df_cold['std_shapley'], bins=20, alpha=0.5, label='Cold Users', color='blue')
            axes[1, 0].hist(df_warm['std_shapley'], bins=20, alpha=0.5, label='Warm Users', color='red')
            axes[1, 0].set_xlabel('Std Dev of Shapley Values')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Shapley Value Variability')
            axes[1, 0].legend()
            
            # 4. Box plot comparison
            data_to_plot = [df_cold['mean_shapley'], df_warm['mean_shapley']]
            axes[1, 1].boxplot(data_to_plot, labels=['Cold Users', 'Warm Users'])
            axes[1, 1].set_ylabel('Mean Shapley Value')
            axes[1, 1].set_title('Shapley Value Comparison')
            
            # Add statistics
            cold_mean = df_cold['mean_shapley'].mean()
            warm_mean = df_warm['mean_shapley'].mean()
            axes[1, 1].text(0.05, 0.95, 
                           f'Cold: {cold_mean:.3f}\nWarm: {warm_mean:.3f}',
                           transform=axes[1, 1].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Shapley Value Analysis for Cold-Start Mitigation')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = setup_logger('enhanced_shap', output_dir / 'enhanced_shap.log')
    logger.info(f"Starting enhanced SHAP analysis")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = RecommenderDataset(
        name=args.dataset,
        path='data/',
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size']
    )
    
    # Create enhanced analyzer
    analyzer = EnhancedSHAPAnalyzer(model, dataset, device, config)
    
    # Create base SHAP explainer
    logger.info(f"Creating SHAP explainer with {args.n_samples} samples")
    analyzer.base_analyzer.create_shap_explainer(args.n_samples)
    
    # 1. Visualize Shapley graph contributions
    logger.info("Visualizing Shapley graph contributions...")
    sample_users = np.random.choice(dataset.warm_users, 3)
    for user_id in sample_users:
        analyzer.visualize_shapley_graph_contribution(
            user_id, 
            output_path=output_dir / f'shapley_graph_user_{user_id}.png'
        )
    
    # 2. Compare Shapley methods
    logger.info("Comparing Shapley value computation methods...")
    analyzer.compare_shapley_methods(
        n_samples=100,
        output_path=output_dir / 'shapley_comparison.png'
    )
    
    # 3. Cold-start analysis
    logger.info("Analyzing Shapley values for cold-start users...")
    analyzer.analyze_cold_start_shapley(
        n_cold_users=50,
        output_path=output_dir / 'cold_start_shapley_analysis.png'
    )
    
    logger.info(f"Enhanced SHAP analysis completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()