"""Complexity and scalability analysis"""

import time
import torch
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class ComplexityAnalyzer:
    """Analyze computational complexity and scalability"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
    def measure_model_complexity(self, model_class, config: Dict,
                               n_users_list: list, n_items_list: list) -> Dict:
        """Measure model complexity across different scales
        
        Args:
            model_class: Model class to analyze
            config: Model configuration
            n_users_list: List of user counts to test
            n_items_list: List of item counts to test
            
        Returns:
            Dictionary with complexity measurements
        """
        results = {
            'forward_time': {},
            'backward_time': {},
            'memory_usage': {},
            'parameter_count': {}
        }
        
        for n_users, n_items in zip(n_users_list, n_items_list):
            key = f"{n_users}x{n_items}"
            
            # Create model
            if hasattr(model_class, '__name__') and 'DyHuCoG' in model_class.__name__:
                model = model_class(n_users, n_items, n_genres=10, config=config)
            else:
                model = model_class(n_users, n_items, config['latent_dim'], 
                                  config['n_layers'])
            model = model.to(self.device)
            
            # Parameter count
            param_count = sum(p.numel() for p in model.parameters())
            results['parameter_count'][key] = param_count
            
            # Create dummy data
            batch_size = 1024
            users = torch.randint(1, n_users + 1, (batch_size,), device=self.device)
            items = torch.randint(1, n_items + 1, (batch_size,), device=self.device)
            
            # Build dummy adjacency
            edge_index = torch.stack([
                torch.randint(0, n_users + n_items, (10000,)),
                torch.randint(0, n_users + n_items, (10000,))
            ], dim=0).to(self.device)
            edge_weight = torch.ones(10000, device=self.device)
            adj = torch.sparse_coo_tensor(
                edge_index, edge_weight,
                (n_users + n_items, n_users + n_items)
            )
            
            # Measure forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            if hasattr(model, 'predict'):
                if hasattr(model, 'adj'):
                    scores = model.predict(users, items, adj)
                else:
                    scores = model.predict(users, items)
            else:
                scores = model(users, items)
                
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            forward_time = time.time() - start_time
            results['forward_time'][key] = forward_time
            
            # Measure backward pass
            loss = scores.mean()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            loss.backward()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            backward_time = time.time() - start_time
            results['backward_time'][key] = backward_time
            
            # Memory usage (GPU)
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                results['memory_usage'][key] = memory_mb
                torch.cuda.reset_peak_memory_stats()
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return results
    
    def analyze_asymptotic_complexity(self, model_name: str) -> Dict:
        """Analyze theoretical complexity
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with complexity analysis
        """
        complexities = {
            'lightgcn': {
                'time_forward': 'O(|E| * d * L)',
                'time_backward': 'O(|E| * d * L)',
                'space': 'O((|U| + |I|) * d)',
                'description': 'Linear in edges and layers'
            },
            'ngcf': {
                'time_forward': 'O(|E| * d * L)',
                'time_backward': 'O(|E| * d * L)',
                'space': 'O((|U| + |I|) * d * L)',
                'description': 'Linear in edges and layers, concatenates embeddings'
            },
            'sgl': {
                'time_forward': 'O(|E| * d * L + B^2 * d)',
                'time_backward': 'O(|E| * d * L + B^2 * d)',
                'space': 'O((|U| + |I|) * d)',
                'description': 'Additional contrastive loss computation'
            },
            'dyhucog': {
                'time_forward': 'O(|E| * d * L + |U| * |I_u| * S)',
                'time_backward': 'O(|E| * d * L + |U| * |I_u| * S)',
                'space': 'O((|U| + |I| + |G|) * d + |E|)',
                'description': 'Additional Shapley computation and hypergraph'
            }
        }
        
        return complexities.get(model_name, {})
    
    def plot_scalability_results(self, results: Dict, output_path: Path):
        """Plot scalability analysis results
        
        Args:
            results: Results from measure_model_complexity
            output_path: Path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        scales = list(results['forward_time'].keys())
        n_users = [int(s.split('x')[0]) for s in scales]
        
        # 1. Forward pass time
        ax = axes[0, 0]
        for metric in ['forward_time']:
            values = [results[metric][s] for s in scales]
            ax.plot(n_users, values, 'o-', label=metric, linewidth=2)
        ax.set_xlabel('Number of Users')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Forward Pass Scalability')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 2. Backward pass time
        ax = axes[0, 1]
        values = [results['backward_time'][s] for s in scales]
        ax.plot(n_users, values, 'o-', color='orange', linewidth=2)
        ax.set_xlabel('Number of Users')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Backward Pass Scalability')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 3. Memory usage
        ax = axes[1, 0]
        if 'memory_usage' in results and results['memory_usage']:
            values = [results['memory_usage'][s] for s in scales]
            ax.plot(n_users, values, 'o-', color='green', linewidth=2)
            ax.set_xlabel('Number of Users')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('GPU Memory Usage')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # 4. Parameter count
        ax = axes[1, 1]
        values = [results['parameter_count'][s] for s in scales]
        ax.plot(n_users, values, 'o-', color='red', linewidth=2)
        ax.set_xlabel('Number of Users')
        ax.set_ylabel('Parameters')
        ax.set_title('Model Parameter Count')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Scalability Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_complexity_table(self, models: list) -> str:
        """Generate LaTeX table of complexity analysis
        
        Args:
            models: List of model names
            
        Returns:
            LaTeX table string
        """
        latex = """\\begin{table}[h]
\\centering
\\caption{Computational Complexity Analysis}
\\label{tab:complexity}
\\begin{tabular}{lccc}
\\toprule
Model & Time (Forward) & Time (Backward) & Space \\\\
\\midrule
"""
        
        for model in models:
            complexity = self.analyze_asymptotic_complexity(model)
            if complexity:
                latex += f"{model.upper()} & {complexity['time_forward']} & "
                latex += f"{complexity['time_backward']} & {complexity['space']} \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return latex