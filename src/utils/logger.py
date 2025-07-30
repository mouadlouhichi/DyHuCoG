"""Logging utilities"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str, log_file: Optional[Path] = None,
                 level: str = 'INFO', format_string: Optional[str] = None) -> logging.Logger:
    """Set up logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TensorBoardLogger:
    """Logger for TensorBoard metrics"""
    
    def __init__(self, log_dir: Path, enabled: bool = True):
        """Initialize TensorBoard logger
        
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether to enable logging
        """
        self.enabled = enabled
        
        if self.enabled:
            from torch.utils.tensorboard import SummaryWriter
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
            
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars"""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
            
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
            
    def log_graph(self, model, input_to_model):
        """Log model graph"""
        if self.writer:
            self.writer.add_graph(model, input_to_model)
            
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        if self.writer:
            self.writer.add_text(tag, text, step)
            
    def close(self):
        """Close the writer"""
        if self.writer:
            self.writer.close()


class ExperimentLogger:
    """Logger for experiment tracking"""
    
    def __init__(self, experiment_name: str, log_dir: Path):
        """Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        # Set up logger
        self.logger = setup_logger(
            f"exp_{experiment_name}",
            log_file=self.log_file,
            level='INFO'
        )
        
        # Metrics storage
        self.metrics = {}
        self.start_time = datetime.now()
        
    def log_config(self, config: dict):
        """Log experiment configuration"""
        self.logger.info("Experiment Configuration:")
        self._log_dict(config, indent=2)
        
    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """Log metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Current step/epoch
            prefix: Prefix for metric names
        """
        # Store metrics
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if full_key not in self.metrics:
                self.metrics[full_key] = []
            self.metrics[full_key].append((step, value))
            
        # Log to file
        self.logger.info(f"Step {step} - {prefix} metrics:")
        self._log_dict(metrics, indent=2)
        
    def log_best_results(self, results: dict):
        """Log best results"""
        self.logger.info("Best Results:")
        self._log_dict(results, indent=2)
        
    def save_results(self):
        """Save all results to file"""
        import json
        
        results = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': str(datetime.now() - self.start_time),
            'metrics': self.metrics
        }
        
        results_file = self.log_dir / f"{self.experiment_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Results saved to {results_file}")
        
    def _log_dict(self, d: dict, indent: int = 0):
        """Log dictionary with indentation"""
        for key, value in d.items():
            if isinstance(value, dict):
                self.logger.info(" " * indent + f"{key}:")
                self._log_dict(value, indent + 2)
            else:
                self.logger.info(" " * indent + f"{key}: {value}")