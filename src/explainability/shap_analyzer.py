"""SHAP-based explainability analysis for DyHuCoG"""

import torch
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """SHAP analysis for recommendation models
    
    Args:
        model: Recommendation model
        dataset: Dataset object
        device: Torch device
        config: Configuration dictionary
    """
    
    def __init__(self, model, dataset, device: torch.device, config: Dict):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.config = config
        
        # Feature representations
        self.feature_names = None
        self.user_features = None
        self.item_features = None
        
        # SHAP explainer
        self.shap_explainer = None
        self.surrogate_model = None
        
        # Build features
        self._build_feature_representations()
        
    def _build_feature_representations(self):
        """Build feature representations for SHAP analysis"""
        logger.info("Building feature representations...")
        
        # Item features: genres + popularity + recency
        n_extra_features = 2  # popularity and recency
        self.item_features = torch.zeros(
            (self.dataset.n_items + 1, self.dataset.n_genres + n_extra_features),
            device=self.device
        )
        
        # Genre features
        for item_id, genres in self.dataset.item_genres.items():
            for genre_idx in genres:
                self.item_features[item_id, genre_idx] = 1
                
        # Popularity feature
        item_counts = self.dataset.train_ratings['item'].value_counts()
        for item_id in range(1, self.dataset.n_items + 1):
            count = item_counts.get(item_id, 0)
            self.item_features[item_id, -2] = count / len(self.dataset.train_ratings)
            
        # Recency feature
        for item_id in range(1, self.dataset.n_items + 1):
            item_ratings = self.dataset.train_ratings[
                self.dataset.train_ratings['item'] == item_id
            ]
            if len(item_ratings) > 0:
                avg_timestamp = item_ratings['timestamp'].mean()
                max_timestamp = self.dataset.ratings['timestamp'].max()
                self.item_features[item_id, -1] = avg_timestamp / max_timestamp
                
        # User features: genre preferences + activity level + recency
        self.user_features = torch.zeros(
            (self.dataset.n_users + 1, self.dataset.n_genres + n_extra_features),
            device=self.device
        )
        
        for user_id in range(1, self.dataset.n_users + 1):
            # Get user's items
            user_items = self.dataset.train_mat[user_id].nonzero().squeeze()
            if len(user_items.shape) == 0:
                continue
                
            # Genre preferences
            genre_counts = torch.zeros(self.dataset.n_genres, device=self.device)
            for item_idx in user_items:
                item_id = item_idx.item() + 1
                genres = self.dataset.item_genres.get(item_id, [])
                for g in genres:
                    genre_counts[g] += 1
                    
            if genre_counts.sum() > 0:
                self.user_features[user_id, :self.dataset.n_genres] = (
                    genre_counts / genre_counts.sum()
                )
                
            # Activity level
            self.user_features[user_id, -2] = len(user_items) / self.dataset.n_items
            
            # Recency
            user_ratings = self.dataset.train_ratings[
                self.dataset.train_ratings['user'] == user_id
            ]
            if len(user_ratings) > 0:
                avg_timestamp = user_ratings['timestamp'].mean()
                max_timestamp = self.dataset.ratings['timestamp'].max()
                self.user_features[user_id, -1] = avg_timestamp / max_timestamp
                
        # Feature names
        user_feature_names = [f"User_{g}" for g in self.dataset.genre_cols] + \
                           ["User_Activity", "User_Recency"]
        item_feature_names = [f"Item_{g}" for g in self.dataset.genre_cols] + \
                           ["Item_Popularity", "Item_Recency"]
        self.feature_names = user_feature_names + item_feature_names
        
    def create_shap_explainer(self, sample_size: int = 1000):
        """Create SHAP explainer with surrogate model
        
        Args:
            sample_size: Number of samples for background data
        """
        logger.info(f"Creating SHAP explainer with {sample_size} samples...")
        
        # Sample user-item pairs
        sample_users = np.random.choice(range(1, self.dataset.n_users + 1), sample_size)
        sample_items = np.random.choice(range(1, self.dataset.n_items + 1), sample_size)
        
        # Create feature matrix
        X_background = []
        y_background = []
        
        self.model.eval()
        with torch.no_grad():
            for u, i in zip(sample_users, sample_items):
                # Combine features
                user_feat = self.user_features[u].cpu().numpy()
                item_feat = self.item_features[i].cpu().numpy()
                combined_feat = np.concatenate([user_feat, item_feat])
                X_background.append(combined_feat)
                
                # Get prediction
                u_tensor = torch.tensor([u], device=self.device)
                i_tensor = torch.tensor([i], device=self.device)
                
                if hasattr(self.model, 'predict'):
                    score = self.model.predict(u_tensor, i_tensor).item()
                else:
                    score = self.model(u_tensor, i_tensor).item()
                    
                y_background.append(score)
                
        X_background = np.array(X_background)
        y_background = np.array(y_background)
        
        # Train surrogate model
        logger.info("Training surrogate model...")
        self.surrogate_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.config['experiment']['seed']
        )
        self.surrogate_model.fit(X_background, y_background)
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.surrogate_model)
        
        logger.info("SHAP explainer created successfully")
        
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict:
        """Explain a specific recommendation
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Dictionary with explanation details
        """
        if self.shap_explainer is None:
            self.create_shap_explainer()
            
        # Get features
        user_feat = self.user_features[user_id].cpu().numpy()
        item_feat = self.item_features[item_id].cpu().numpy()
        combined_feat = np.concatenate([user_feat, item_feat]).reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(combined_feat)[0]
        
        # Get prediction
        with torch.no_grad():
            u_tensor = torch.tensor([user_id], device=self.device)
            i_tensor = torch.tensor([item_id], device=self.device)
            
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(u_tensor, i_tensor).item()
            else:
                prediction = self.model(u_tensor, i_tensor).item()
                
        # Get item title if available
        item_title = f"Item {item_id}"
        if hasattr(self.dataset, 'items'):
            item_data = self.dataset.items[self.dataset.items['item'] == item_id]
            if not item_data.empty:
                item_title = item_data.iloc[0].get('title', item_title)
                
        return {
            'user_id': user_id,
            'item_id': item_id,
            'item_title': item_title,
            'prediction': prediction,
            'base_value': self.shap_explainer.expected_value,
            'shap_values': shap_values,
            'feature_values': combined_feat[0],
            'feature_names': self.feature_names
        }
        
    def compute_global_shap_values(self, n_samples: int = 1000) -> Tuple[np.ndarray, List[str]]:
        """Compute global SHAP values across samples
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            Tuple of (shap_values_matrix, feature_names)
        """
        if self.shap_explainer is None:
            self.create_shap_explainer()
            
        logger.info(f"Computing global SHAP values for {n_samples} samples...")
        
        # Sample user-item pairs
        sample_users = np.random.choice(range(1, self.dataset.n_users + 1), n_samples)
        sample_items = np.random.choice(range(1, self.dataset.n_items + 1), n_samples)
        
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
        
        return shap_values, self.feature_names
        
    def get_top_recommendations(self, user_id: int, k: int = 10) -> List[int]:
        """Get top-k recommendations for a user
        
        Args:
            user_id: User ID
            k: Number of recommendations
            
        Returns:
            List of recommended item IDs
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get predictions for all items
            items = torch.arange(1, self.dataset.n_items + 1, device=self.device)
            users = torch.full_like(items, user_id, device=self.device)
            
            if hasattr(self.model, 'predict'):
                scores = self.model.predict(users, items)
            else:
                scores = self.model(users, items)
                
            # Filter out training items
            train_items = torch.where(self.dataset.train_mat[user_id] > 0)[0]
            if len(train_items) > 0:
                scores[train_items - 1] = -float('inf')
                
            # Get top-k
            _, top_k_indices = torch.topk(scores, k)
            top_k_items = (top_k_indices + 1).cpu().numpy()
            
        return top_k_items.tolist()
        
    def analyze_feature_interactions(self, n_samples: int = 100) -> np.ndarray:
        """Analyze feature interactions using SHAP
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            Interaction matrix
        """
        if not hasattr(self.shap_explainer, 'shap_interaction_values'):
            logger.warning("Tree explainer does not support interaction values")
            return None
            
        # Sample data
        sample_users = np.random.choice(range(1, self.dataset.n_users + 1), n_samples)
        sample_items = np.random.choice(range(1, self.dataset.n_items + 1), n_samples)
        
        X_sample = []
        for u, i in zip(sample_users, sample_items):
            user_feat = self.user_features[u].cpu().numpy()
            item_feat = self.item_features[i].cpu().numpy()
            combined_feat = np.concatenate([user_feat, item_feat])
            X_sample.append(combined_feat)
            
        X_sample = np.array(X_sample)
        
        # Get interaction values
        interaction_values = self.shap_explainer.shap_interaction_values(X_sample)
        
        # Average across samples
        mean_interactions = np.mean(np.abs(interaction_values), axis=0)
        
        return mean_interactions