"""
Training Configuration for Maximum Accuracy

Optimized hyperparameters prioritizing accuracy over training time/cost.
"""

# Baseline Training (Complete Data) - Maximum Accuracy
BASELINE_CONFIG = {
    'lambda_weight': 0.0,              # Disabled - incompatible with log transform
    'n_estimators': 2000,              # More trees = better accuracy (vs 500 default)
    'max_depth': 12,                   # Deeper trees (vs 8 default)
    'learning_rate': 0.03,             # Slower learning for better convergence (vs 0.05)
    'subsample': 0.9,                  # Higher subsample (vs 0.8)
    'colsample_bytree': 0.9,           # More features per tree (vs 0.8)
    'min_child_weight': 1,             # Allow finer splits
    'gamma': 0,                        # No regularization penalty initially
    'reg_alpha': 0,                    # L1 regularization (tune if needed)
    'reg_lambda': 1,                   # L2 regularization
    'tree_method': 'hist',             # Histogram-based (XGBoost 2.0+)
    'device': 'cuda',                  # GPU acceleration (XGBoost 2.0+ syntax)
    'early_stopping_rounds': 100,      # More patience (vs 50)
    'random_state': 42
}

# Robustness Training (With Artificial Missing Values) - Maximum Accuracy
ROBUSTNESS_CONFIG = {
    'lambda_weight': 0.15,             # Slightly higher constraint weight
    'n_estimators': 2500,              # Even more trees for complex patterns
    'max_depth': 14,                   # Deeper for handling missing patterns
    'learning_rate': 0.02,             # Even slower for stability
    'subsample': 0.85,                 # Slightly lower to prevent overfitting
    'colsample_bytree': 0.85,
    'min_child_weight': 1,
    'gamma': 0.1,                      # Small regularization
    'reg_alpha': 0.1,
    'reg_lambda': 1.5,
    'tree_method': 'hist',             # Histogram-based (XGBoost 2.0+)
    'device': 'cuda',                  # GPU acceleration (XGBoost 2.0+ syntax)
    'early_stopping_rounds': 150,      # Even more patience
    'random_state': 42
}

# Missing Value Augmentation Settings
MISSING_AUGMENTATION = {
    # Probability of introducing missing values during training
    'weight_missing_prob': 0.15,       # 15% of samples will have missing weight
    'distance_missing_prob': 0.25,     # 25% missing distance (most common in practice)
    'materials_missing_prob': 0.10,    # 10% missing materials
    
    # For robustness testing (reduced for faster iteration)
    'test_missing_levels': [0.0, 0.2, 0.4],  # 3 levels instead of 6
    'test_n_trials': 3,                       # 3 trials instead of 5 (total: 9 vs 30)
}

# Hyperparameter Tuning Grid (if enabled)
TUNING_GRID = {
    'lambda_weight': [0.05, 0.1, 0.15, 0.2],
    'max_depth': [10, 12, 14, 16],
    'learning_rate': [0.02, 0.03, 0.04],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.85, 0.9],
    'colsample_bytree': [0.8, 0.85, 0.9]
}

# Success Criteria
ACCURACY_TARGETS = {
    'baseline': {
        'r2_min': 0.90,              # R² > 0.90 (Excellent tier)
        'mae_carbon_max': 0.10,      # MAE < 0.10 kg CO2e
        'mae_water_max': 150,        # MAE < 150L
        'constraint_violation_max': 0.01  # < 0.01 kg CO2e
    },
    'robustness_30pct_missing': {
        'r2_min': 0.80,              # R² > 0.80 with 30% missing
        'degradation_max': 0.15,     # < 15% performance drop
        'mae_carbon_max': 0.15,      # MAE < 0.15 kg CO2e
        'mae_water_max': 200,        # MAE < 200L
    }
}
