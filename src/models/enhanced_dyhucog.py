# Enhanced DyHuCoG with SHAP Integration
# Additions to the original code for better explainability and analysis

import shap
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Add these imports to the original imports
# pip install shap networkx

class EnhancedDyHuCoG(DyHuCoG):
    """Enhanced DyHuCoG with SHAP-based explainability"""
    
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        self.shap_explainer = None
        self.item_features = None
        self.user_features = None
        self.build_feature_representations()
        
    def build_feature_representations(self):
        """Build feature representations for SHAP analysis"""
        # Item features: genres + popularity + interaction count
        self.item_features = torch.zeros((self.n_items + 1, self.n_genres + 2), device=device)
        
        for item_id in range(1, self.n_items + 1):
            # Genre features
            genres = self.dataset.item_genres.get(item_id, [])
            for g in genres:
                self.item_features[item_id, g] = 1
            
            # Popularity feature
            item_count = len(self.dataset.train_ratings[self.dataset.train_ratings['item'] == item_id])
            self.item_features[item_id, -2] = item_count / len(self.dataset.train_ratings)
            
            # Recency feature (normalized timestamp)
            item_ratings = self.dataset.train_ratings[self.dataset.train_ratings['item'] == item_id]
            if len(item_ratings) > 0:
                avg_timestamp = item_ratings['timestamp'].mean()
                self.item_features[item_id, -1] = avg_timestamp / self.dataset.ratings['timestamp'].max()
        
        # User features: interaction count, avg rating time, genre preferences
        self.user_features = torch.zeros((self.n_users + 1, self.n_genres + 2), device=device)
        
        for user_id in range(1, self.n_users + 1):
            user_items = self.dataset.train_mat[user_id].nonzero().squeeze()
            if len(user_items.shape) == 0:
                continue
                
            # Genre preferences
            genre_counts = torch.zeros(self.n_genres, device=device)
            for item in user_items:
                if item > 0:
                    genres = self.dataset.item_genres.get(item.item() + 1, [])
                    for g in genres:
                        genre_counts[g] += 1
            
            if genre_counts.sum() > 0:
                self.user_features[user_id, :self.n_genres] = genre_counts / genre_counts.sum()
            
            # Interaction count feature
            self.user_features[user_id, -2] = len(user_items) / self.n_items
            
            # Activity recency
            user_ratings = self.dataset.train_ratings[self.dataset.train_ratings['user'] == user_id]
            if len(user_ratings) > 0:
                avg_timestamp = user_ratings['timestamp'].mean()
                self.user_features[user_id, -1] = avg_timestamp / self.dataset.ratings['timestamp'].max()
    
    def create_shap_explainer(self, sample_size=1000):
        """Create SHAP explainer for recommendation predictions"""
        print("Creating SHAP explainer...")
        
        # Sample users and items for background data
        sample_users = np.random.choice(range(1, self.n_users + 1), sample_size)
        sample_items = np.random.choice(range(1, self.n_items + 1), sample_size)
        
        # Create feature matrix for samples
        X_background = []
        y_background = []
        
        with torch.no_grad():
            for u, i in zip(sample_users, sample_items):
                # Combine user and item features
                user_feat = self.user_features[u].cpu().numpy()
                item_feat = self.item_features[i].cpu().numpy()
                combined_feat = np.concatenate([user_feat, item_feat])
                X_background.append(combined_feat)
                
                # Get prediction
                u_tensor = torch.tensor([u], device=device)
                i_tensor = torch.tensor([i], device=device)
                score = self.predict(u_tensor, i_tensor).item()
                y_background.append(score)
        
        X_background = np.array(X_background)
        y_background = np.array(y_background)
        
        # Train a simple model for SHAP
        self.surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.surrogate_model.fit(X_background, y_background)
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.surrogate_model)
        
        # Store feature names
        genre_names = self.dataset.genre_cols
        user_feature_names = [f"User_{g}" for g in genre_names] + ["User_Activity", "User_Recency"]
        item_feature_names = [f"Item_{g}" for g in genre_names] + ["Item_Popularity", "Item_Recency"]
        self.feature_names = user_feature_names + item_feature_names
    
    def explain_recommendation(self, user_id, item_id, plot=True):
        """Explain a specific recommendation using SHAP"""
        if self.shap_explainer is None:
            self.create_shap_explainer()
        
        # Get features
        user_feat = self.user_features[user_id].cpu().numpy()
        item_feat = self.item_features[item_id].cpu().numpy()
        combined_feat = np.concatenate([user_feat, item_feat]).reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(combined_feat)
        
        if plot:
            # Create custom plot
            plt.figure(figsize=(12, 6))
            
            # Get item title
            item_title = self.dataset.items[self.dataset.items['item'] == item_id]['title'].values[0]
            plt.suptitle(f'Recommendation Explanation\nUser {user_id} → "{item_title}"', fontsize=14)
            
            # SHAP waterfall plot
            shap.waterfall_plot(
                shap.Explanation(values=shap_values[0], 
                                base_values=self.shap_explainer.expected_value,
                                data=combined_feat[0],
                                feature_names=self.feature_names),
                max_display=15
            )
            
            plt.tight_layout()
            
        return shap_values[0]
    
    def analyze_feature_importance_global(self, n_samples=1000):
        """Global feature importance analysis"""
        if self.shap_explainer is None:
            self.create_shap_explainer()
        
        print("Analyzing global feature importance...")
        
        # Sample data
        sample_users = np.random.choice(range(1, self.n_users + 1), n_samples)
        sample_items = np.random.choice(range(1, self.n_items + 1), n_samples)
        
        # Create feature matrix
        X_sample = []
        for u, i in zip(sample_users, sample_items):
            user_feat = self.user_features[u].cpu().numpy()
            item_feat = self.item_features[i].cpu().numpy()
            combined_feat = np.concatenate([user_feat, item_feat])
            X_sample.append(combined_feat)
        
        X_sample = np.array(X_sample)
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        # Plot summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title("Global Feature Importance for Recommendations")
        plt.tight_layout()
        plt.savefig('results/shap_global_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_values
    
    def visualize_shapley_graph_contribution(self, user_id, top_k=10):
        """Visualize how Shapley values affect the recommendation graph"""
        plt.figure(figsize=(15, 10))
        
        # Get user's items and their Shapley values
        user_items = self.dataset.train_mat[user_id].nonzero().squeeze()
        if len(user_items.shape) == 0:
            user_items = user_items.unsqueeze(0)
        
        # Get Shapley values
        with torch.no_grad():
            user_item_vector = self.dataset.train_mat[user_id]
            shapley_values = self.shapley_net(user_item_vector.unsqueeze(0)).squeeze()
        
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
            item_info = self.dataset.items[self.dataset.items['item'] == item_id].iloc[0]
            item_title = item_info['title'][:20] + "..." if len(item_info['title']) > 20 else item_info['title']
            
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
            edge_weight = self.edge_weights.get((user_id, item_id), 1.0)
            G.add_edge(f"U{user_id}", f"I{item_id}", weight=edge_weight, shapley=shapley_val)
        
        # Add genre nodes
        genre_positions = {}
        for idx, (item_id, pos) in enumerate(item_positions.items()):
            genres = self.dataset.item_genres.get(item_id, [])
            for genre_idx in genres:
                genre_name = self.dataset.genre_cols[genre_idx]
                if f"G{genre_name}" not in G:
                    # Position genres on outer circle
                    angle = 2 * np.pi * genre_idx / self.n_genres
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
        plt.savefig(f'results/shapley_graph_user_{user_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_shapley_methods(self, n_samples=100):
        """Compare FastSHAP approximation with SHAP library"""
        print("Comparing Shapley value methods...")
        
        comparison_results = []
        
        for _ in range(n_samples):
            user_id = np.random.randint(1, self.n_users + 1)
            
            # Get user's items
            user_items = self.dataset.train_mat[user_id]
            item_indices = user_items.nonzero().squeeze()
            
            if len(item_indices) == 0:
                continue
            
            # FastSHAP approximation
            with torch.no_grad():
                fastshap_values = self.shapley_net(user_items.unsqueeze(0)).squeeze()
            
            # SHAP library approximation (for a subset of items)
            if self.shap_explainer is not None:
                for item_idx in item_indices[:5]:  # Limit to 5 items for speed
                    item_id = item_idx.item() + 1
                    
                    # Get SHAP explanation
                    shap_vals = self.explain_recommendation(user_id, item_id, plot=False)
                    
                    # Extract item-specific SHAP values
                    item_shap_sum = np.sum(shap_vals[self.n_genres+2:])  # Sum of item features
                    
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
            plt.savefig('results/shapley_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def analyze_cold_start_shapley(self, n_cold_users=50):
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
                    shapley_values = self.shapley_net(user_items.unsqueeze(0)).squeeze()
                    
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
                    shapley_values = self.shapley_net(user_items.unsqueeze(0)).squeeze()
                    
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
        plt.savefig('results/cold_start_shapley_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_enhanced_experiments_with_shap(dataset, config):
    """Run experiments with SHAP analysis"""
    print("\n" + "="*50)
    print("Running Enhanced DyHuCoG with SHAP Analysis")
    print("="*50)
    
    # Train Enhanced DyHuCoG
    model = EnhancedDyHuCoG(dataset, config).to(device)
    history = train_model(model, dataset, "Enhanced DyHuCoG")
    
    # Create SHAP explainer
    model.create_shap_explainer()
    
    # 1. Global feature importance
    print("\n1. Analyzing global feature importance...")
    shap_values = model.analyze_feature_importance_global()
    
    # 2. Individual recommendation explanations
    print("\n2. Explaining individual recommendations...")
    # Find a user with good test performance
    test_users = list(dataset.test_dict.keys())[:10]
    for user_id in test_users[:3]:  # Explain for 3 users
        test_items = dataset.test_dict[user_id]
        if test_items:
            # Get top recommendation
            items = torch.arange(1, dataset.n_items + 1, device=device)
            users_tensor = torch.full_like(items, user_id, device=device)
            
            with torch.no_grad():
                scores = model.predict(users_tensor, items)
                _, top_items = torch.topk(scores, 5)
            
            # Explain top recommendation
            top_item = top_items[0].item() + 1
            print(f"\nExplaining recommendation for User {user_id} → Item {top_item}")
            model.explain_recommendation(user_id, top_item)
    
    # 3. Visualize Shapley graph contributions
    print("\n3. Visualizing Shapley value contributions in graph...")
    sample_users = np.random.choice(dataset.warm_users, 3)
    for user_id in sample_users:
        model.visualize_shapley_graph_contribution(user_id)
    
    # 4. Compare Shapley methods
    print("\n4. Comparing Shapley value computation methods...")
    model.compare_shapley_methods()
    
    # 5. Cold-start analysis
    print("\n5. Analyzing Shapley values for cold-start users...")
    model.analyze_cold_start_shapley()
    
    # 6. Feature interaction analysis
    print("\n6. Creating SHAP interaction plots...")
    if hasattr(model.shap_explainer, 'shap_interaction_values'):
        # Sample some data
        n_samples = 100
        sample_users = np.random.choice(range(1, dataset.n_users + 1), n_samples)
        sample_items = np.random.choice(range(1, dataset.n_items + 1), n_samples)
        
        X_sample = []
        for u, i in zip(sample_users, sample_items):
            user_feat = model.user_features[u].cpu().numpy()
            item_feat = model.item_features[i].cpu().numpy()
            combined_feat = np.concatenate([user_feat, item_feat])
            X_sample.append(combined_feat)
        
        X_sample = np.array(X_sample)
        
        # Get interaction values
        shap_interaction_values = model.shap_explainer.shap_interaction_values(X_sample)
        
        # Plot top interactions
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_interaction_values, X_sample, 
                         feature_names=model.feature_names,
                         max_display=10)
        plt.title("SHAP Feature Interactions")
        plt.tight_layout()
        plt.savefig('results/shap_interactions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return model, history


# Add to main() function:
def main_with_shap():
    """Extended main function with SHAP analysis"""
    # Run original main experiments
    main()
    
    # Load dataset again for SHAP analysis
    data_path = 'ml-100k/'
    dataset = RecommenderDataset(data_path)
    
    # Run enhanced experiments with SHAP
    print("\n" + "="*70)
    print("EXTENDED ANALYSIS WITH SHAP")
    print("="*70)
    
    enhanced_model, enhanced_history = run_enhanced_experiments_with_shap(dataset, config)
    
    print("\nSHAP analysis completed! Check the 'results/' folder for visualizations.")


# Additional helper function for paper-ready SHAP plots
def create_paper_figures(model, dataset):
    """Create publication-quality figures for the paper"""
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Figure 1: Shapley Value Distribution Across User Groups
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    user_groups = [
        ('Cold Users', dataset.cold_users[:50]),
        ('Warm Users', dataset.warm_users[:50]),
        ('Hot Users', dataset.hot_users[:50])
    ]
    
    for idx, (group_name, users) in enumerate(user_groups):
        shapley_values = []
        
        with torch.no_grad():
            for user in users:
                user_items = dataset.train_mat[user]
                if user_items.sum() > 0:
                    sv = model.shapley_net(user_items.unsqueeze(0)).squeeze()
                    shapley_values.extend(sv[user_items > 0].cpu().numpy())
        
        axes[idx].hist(shapley_values, bins=30, alpha=0.7, color=f'C{idx}', edgecolor='black')
        axes[idx].set_title(group_name)
        axes[idx].set_xlabel('Shapley Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Shapley Value Distributions Across User Groups')
    plt.tight_layout()
    plt.savefig('results/paper_shapley_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Paper-ready figures created in results/ folder")


if __name__ == "__main__":
    # Run extended experiments with SHAP
    main_with_shap()