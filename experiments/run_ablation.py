#!/usr/bin/env python
"""
Run ablation study for DyHuCoG components
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import RecommenderDataset
from src.models.dyhucog import DyHuCoG
from src.models.cooperative_game import CooperativeGameDAE, ShapleyValueNetwork
from src.utils.trainer import Trainer
from src.utils.logger import setup_logger, ExperimentLogger
from src.utils.graph_builder import GraphBuilder
from config.model_config import get_model_config


def parse_args():
    parser = argparse.ArgumentParser(description='Run ablation study')
    
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       help='Dataset name')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file')
    parser.add_argument('--output_dir', type=str, default='results/ablation',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    
    return parser.parse_args()


class DyHuCoGNoShapley(DyHuCoG):
    """DyHuCoG without Shapley value weighting"""
    
    def compute_shapley_weights(self, train_mat: torch.Tensor):
        """Use uniform weights instead of Shapley values"""
        self.edge_weights = defaultdict(lambda: 1.0)
        return self.edge_weights


class DyHuCoGNoGenre(DyHuCoG):
    """DyHuCoG without genre information (no hypergraph)"""
    
    def build_hypergraph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                        item_genres: dict):
        """Build only user-item graph without genres"""
        # Filter out genre connections
        mask = edge_index[1] < self.n_users + self.n_items
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]
        
        # Update node count
        self.n_nodes = self.n_users + self.n_items
        
        # Create adjacency matrix
        self.adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, (self.n_nodes, self.n_nodes)
        )
        
        # Normalize
        self.adj = GraphBuilder.normalize_adj(self.adj)


class DyHuCoGNoAttention(DyHuCoG):
    """DyHuCoG without attention mechanism"""
    
    def __init__(self, n_users: int, n_items: int, n_genres: int, config: dict):
        super().__init__(n_users, n_items, n_genres, config)
        self.use_attention = False
        
    def predict(self, users: torch.Tensor, items: torch.Tensor):
        """Predict without attention"""
        emb = self.forward()
        
        user_emb = emb[users - 1]
        item_emb = emb[self.n_users + items - 1]
        
        # Only inner product
        score = (user_emb * item_emb).sum(dim=1)
        
        return score


class DyHuCoGNoCooperative(DyHuCoG):
    """DyHuCoG without cooperative game components"""
    
    def compute_shapley_weights(self, train_mat: torch.Tensor):
        """Skip Shapley computation"""
        self.edge_weights = defaultdict(lambda: 1.0)
        return self.edge_weights
        
    def train_cooperative_components(self, train_loader):
        """Skip cooperative component training"""
        pass


class DyHuCoGStaticWeights(DyHuCoG):
    """DyHuCoG with static learned weights instead of dynamic Shapley"""
    
    def __init__(self, n_users: int, n_items: int, n_genres: int, config: dict):
        super().__init__(n_users, n_items, n_genres, config)
        
        # Learnable edge weight parameters
        self.edge_weight_params = nn.Parameter(
            torch.ones(n_users * n_items) * 0.1
        )
        
    def compute_shapley_weights(self, train_mat: torch.Tensor):
        """Use learned static weights"""
        self.edge_weights = {}
        
        idx = 0
        for user in range(1, self.n_users + 1):
            for item in range(1, self.n_items + 1):
                if train_mat[user, item] > 0:
                    weight = torch.sigmoid(self.edge_weight_params[idx]).item() + 0.5
                    self.edge_weights[(user, item)] = weight
                    idx += 1
                    
        return self.edge_weights


def run_ablation_variant(variant_name: str, variant_class, dataset: RecommenderDataset,
                        config: dict, device: torch.device, logger) -> dict:
    """Run experiment for an ablation variant
    
    Args:
        variant_name: Name of the variant
        variant_class: Model class for variant
        dataset: Dataset object
        config: Configuration dictionary
        device: Torch device
        logger: Logger instance
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\nRunning ablation: {variant_name}")
    
    # Create model
    model = variant_class(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_genres=dataset.n_genres,
        config=config['model']
    ).to(device)
    
    # Create output directories
    checkpoint_dir = Path(config['experiment']['checkpoint_dir']) / f"ablation_{variant_name}"
    log_dir = Path(config['experiment']['log_dir']) / f"ablation_{variant_name}"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Train model
    history = trainer.train()
    
    # Get results
    best_metrics = trainer.best_metrics
    
    # Evaluate on cold users
    from src.utils.evaluator import Evaluator
    evaluator = Evaluator(model, dataset, device)
    
    cold_metrics, _, _ = evaluator.evaluate(
        user_list=dataset.cold_users,
        k_values=config['evaluation']['k_values']
    )
    
    return {
        'variant': variant_name,
        'best_val_ndcg': best_metrics['val_metrics'][10]['ndcg'],
        'best_test_ndcg': best_metrics['test_metrics'][10]['ndcg'],
        'best_test_hr': best_metrics['test_metrics'][10]['hit_rate'],
        'cold_ndcg': cold_metrics[10]['ndcg'],
        'cold_hr': cold_metrics[10]['hit_rate'],
        'training_time': sum(history['epoch_times'])
    }


def main():
    args = parse_args()
    
    # Set up experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=f"ablation_{args.dataset}",
        log_dir=Path(args.output_dir)
    )
    
    # Load configuration
    config = get_model_config(args.config).to_dict()
    config['dataset']['name'] = args.dataset
    config['model']['name'] = 'dyhucog'
    
    exp_logger.log_config(config)
    
    # Get device
    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available() 
        else args.device if args.device != 'auto' 
        else 'cpu'
    )
    
    exp_logger.logger.info(f"Using device: {device}")
    
    # Load dataset
    exp_logger.logger.info(f"Loading dataset: {args.dataset}")
    dataset = RecommenderDataset(
        name=args.dataset,
        path='data/',
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size']
    )
    
    # Define ablation variants
    variants = {
        'Full': DyHuCoG,
        'NoShapley': DyHuCoGNoShapley,
        'NoGenre': DyHuCoGNoGenre,
        'NoAttention': DyHuCoGNoAttention,
        'NoCooperative': DyHuCoGNoCooperative,
        'StaticWeights': DyHuCoGStaticWeights
    }
    
    # Run ablation study
    results = {}
    
    for variant_name, variant_class in variants.items():
        # Set seed for reproducibility
        torch.manual_seed(config['experiment']['seed'])
        np.random.seed(config['experiment']['seed'])
        
        # Run variant
        variant_results = run_ablation_variant(
            variant_name=variant_name,
            variant_class=variant_class,
            dataset=dataset,
            config=config,
            device=device,
            logger=exp_logger.logger
        )
        
        results[variant_name] = variant_results
        
        # Log results
        exp_logger.log_metrics(variant_results, step=0, prefix=variant_name)
        
    # Print summary
    exp_logger.logger.info("\n" + "="*70)
    exp_logger.logger.info("ABLATION STUDY RESULTS")
    exp_logger.logger.info("="*70)
    
    # Create comparison table
    exp_logger.logger.info("\n%-20s %-12s %-12s %-12s %-12s" % 
                          ("Variant", "Val NDCG", "Test NDCG", "Test HR", "Cold NDCG"))
    exp_logger.logger.info("-" * 70)
    
    for variant_name, res in results.items():
        exp_logger.logger.info("%-20s %-12.4f %-12.4f %-12.4f %-12.4f" % 
                              (variant_name, 
                               res['best_val_ndcg'],
                               res['best_test_ndcg'],
                               res['best_test_hr'],
                               res['cold_ndcg']))
    
    # Calculate relative improvements
    full_results = results['Full']
    
    exp_logger.logger.info("\n" + "="*70)
    exp_logger.logger.info("RELATIVE PERFORMANCE (vs Full Model)")
    exp_logger.logger.info("="*70)
    
    for variant_name, res in results.items():
        if variant_name == 'Full':
            continue
            
        test_drop = ((full_results['best_test_ndcg'] - res['best_test_ndcg']) / 
                    full_results['best_test_ndcg'] * 100)
        cold_drop = ((full_results['cold_ndcg'] - res['cold_ndcg']) / 
                    full_results['cold_ndcg'] * 100)
        
        exp_logger.logger.info(f"\n{variant_name}:")
        exp_logger.logger.info(f"  Test NDCG drop: {test_drop:.1f}%")
        exp_logger.logger.info(f"  Cold NDCG drop: {cold_drop:.1f}%")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Save latex table
    latex_table = generate_latex_table(results)
    with open(output_dir / 'ablation_table.tex', 'w') as f:
        f.write(latex_table)
        
    exp_logger.log_best_results(results)
    exp_logger.save_results()
    
    exp_logger.logger.info(f"\nResults saved to {output_dir}")
    exp_logger.logger.info("Ablation study completed!")


def generate_latex_table(results: dict) -> str:
    """Generate LaTeX table for ablation results"""
    latex = """\\begin{table}[h]
\\centering
\\caption{Ablation Study Results on MovieLens-100K}
\\label{tab:ablation}
\\begin{tabular}{lcccc}
\\toprule
Variant & Val NDCG@10 & Test NDCG@10 & Test HR@10 & Cold NDCG@10 \\\\
\\midrule
"""
    
    for variant_name, res in results.items():
        latex += f"{variant_name} & {res['best_val_ndcg']:.4f} & "
        latex += f"{res['best_test_ndcg']:.4f} & {res['best_test_hr']:.4f} & "
        latex += f"{res['cold_ndcg']:.4f} \\\\\n"
        
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex


if __name__ == '__main__':
    main()