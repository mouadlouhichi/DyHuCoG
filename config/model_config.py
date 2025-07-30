"""Model configuration utilities"""

from typing import Dict, Any
import yaml
from pathlib import Path


class ModelConfig:
    """Model configuration container"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['model', 'training', 'dataset', 'evaluation']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration section: {key}")
        
        # Validate model config
        model_config = self.config['model']
        if 'name' not in model_config:
            raise ValueError("Model name not specified")
        
        if model_config['name'] == 'dyhucog':
            required_dyhucog_params = ['latent_dim', 'n_layers', 'dae_hidden', 'shapley_hidden']
            for param in required_dyhucog_params:
                if param not in model_config:
                    raise ValueError(f"Missing required DyHuCoG parameter: {param}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.config.copy()


def get_model_config(config_path: str = 'config/config.yaml', 
                     overrides: Dict[str, Any] = None) -> ModelConfig:
    """Load model configuration from file
    
    Args:
        config_path: Path to configuration file
        overrides: Dictionary of values to override
        
    Returns:
        ModelConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            keys = key.split('.')
            d = config_dict
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
    
    return ModelConfig(config_dict)


# Default configurations for different models
DEFAULT_CONFIGS = {
    'dyhucog': {
        'model': {
            'name': 'dyhucog',
            'latent_dim': 64,
            'n_layers': 3,
            'dropout': 0.1,
            'dae_hidden': 128,
            'dae_epochs': 30,
            'shapley_hidden': 128,
            'shapley_epochs': 30,
            'n_shapley_samples': 10,
            'use_attention': True,
            'use_genres': True
        }
    },
    'lightgcn': {
        'model': {
            'name': 'lightgcn',
            'latent_dim': 64,
            'n_layers': 3,
            'dropout': 0.0
        }
    },
    'ngcf': {
        'model': {
            'name': 'ngcf',
            'latent_dim': 64,
            'n_layers': 3,
            'dropout': 0.1
        }
    }
}