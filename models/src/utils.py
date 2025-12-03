"""
Utilities Module

Helper functions for model training and evaluation.
"""

import numpy as np
import random
import os
import logging
from pathlib import Path


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Setup logger for module"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if path provided
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def format_metric(value: float, decimals: int = 4) -> str:
    """Format metric value for display"""
    return f"{value:.{decimals}f}"


if __name__ == '__main__':
    # Test utilities
    set_random_seed(42)
    logger = setup_logger('test', 'logs/test.log')
    logger.info("Utilities test successful")
    print(f"Formatted metric: {format_metric(0.12345678, 3)}")
