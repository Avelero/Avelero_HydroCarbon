"""
Training Configuration for PRODUCTION-READY MODEL

Optimized hyperparameters balancing accuracy and generalization.
Formula features DISABLED to prevent data leakage.
Target encoding uses K-Fold CV to prevent leakage.
"""

# Baseline Training (Complete Data) - PRODUCTION-READY (no data leakage)
BASELINE_CONFIG = {
    'lambda_weight': 0.0,              # Disabled for baseline (enable for physics constraint)
    'n_estimators': 3000,              # Enough trees for convergence
    'max_depth': 10,                   # Moderate depth to prevent overfitting
    'learning_rate': 0.03,             # Balanced learning rate
    'subsample': 0.8,                  # Use 80% of data per tree (regularization)
    'colsample_bytree': 0.8,           # Use 80% of features per tree
    'min_child_weight': 5,             # Require more samples per leaf (regularization)
    'gamma': 0.1,                      # Minimum loss reduction for split (regularization)
    'reg_alpha': 0.1,                  # L1 regularization (feature selection)
    'reg_lambda': 1.0,                 # L2 regularization (weight decay)
    'tree_method': 'hist',             # Histogram-based (XGBoost 2.0+)
    'device': 'cuda',                  # GPU acceleration
    'early_stopping_rounds': 100,      # Stop if no improvement for 100 rounds
    'random_state': 42,
    # GPU optimization
    'max_bin': 512,                    # Moderate histogram resolution
    'grow_policy': 'depthwise',        # Standard depth-first growth
    'max_leaves': 0,                   # Unlimited leaves (controlled by max_depth)
    'sampling_method': 'gradient_based',  # GPU-accelerated sampling
}

# Robustness Training (With Artificial Missing Values) - PRODUCTION-READY
# Uses higher augmentation rates to learn fallback features
ROBUSTNESS_CONFIG = {
    'lambda_weight': 0.0,              # Disabled (physics constraint not needed)
    'n_estimators': 3000,              # Enough trees for convergence
    'max_depth': 12,                   # Slightly deeper for learning fallback patterns
    'learning_rate': 0.03,             # Same as baseline
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,                      # Regularization
    'reg_alpha': 0.1,                  # L1 regularization
    'reg_lambda': 1.0,                 # L2 regularization (same as baseline)
    'tree_method': 'hist',
    'device': 'cuda',
    'early_stopping_rounds': 100,
    'random_state': 42,
    # GPU optimization
    'max_bin': 512,
    'grow_policy': 'depthwise',
    'max_leaves': 0,
    'sampling_method': 'gradient_based',
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

# Success Criteria - REALISTIC TARGETS (without data leakage)
# Note: R² 0.85-0.95 is excellent for real-world carbon footprint prediction
# Previous R² > 0.999 was due to data leakage, not real model performance
ACCURACY_TARGETS = {
    'baseline': {
        'r2_min': 0.80,              # R² > 0.80 (Good tier - realistic target)
        'mae_carbon_max': 0.50,      # MAE < 0.50 kg CO2e (realistic)
        'mae_water_max': 500,        # MAE < 500L (realistic)
        'constraint_violation_max': 0.05  # < 0.05 kg CO2e
    },
    'robustness_30pct_missing': {
        'r2_min': 0.70,              # R² > 0.70 with 30% missing
        'degradation_max': 0.20,     # < 20% performance drop
        'mae_carbon_max': 0.75,      # MAE < 0.75 kg CO2e
        'mae_water_max': 750,        # MAE < 750L
    }
}
