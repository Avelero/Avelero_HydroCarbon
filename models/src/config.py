"""
Training Configuration for MAXIMUM ACCURACY

Optimized hyperparameters prioritizing accuracy - no compromises on time/VRAM.
Target: R² > 0.999 on all targets
"""

# Baseline Training (Complete Data) - MAXIMUM ACCURACY (no compromises)
BASELINE_CONFIG = {
    'lambda_weight': 0.0,              # Disabled - incompatible with log transform
    'n_estimators': 5000,              # Many more trees (was 2000) - early stopping will find optimal
    'max_depth': 15,                   # Very deep trees (was 12)
    'learning_rate': 0.01,             # Very slow learning for best convergence (was 0.03)
    'subsample': 0.95,                 # Use almost all data (was 0.9)
    'colsample_bytree': 0.95,          # Use almost all features (was 0.9)
    'min_child_weight': 1,             # Allow finest splits
    'gamma': 0,                        # No regularization - maximize fit
    'reg_alpha': 0,                    # No L1 regularization
    'reg_lambda': 0.5,                 # Minimal L2 regularization
    'tree_method': 'hist',             # Histogram-based (XGBoost 2.0+)
    'device': 'cuda',                  # GPU acceleration
    'early_stopping_rounds': 200,      # Very patient (was 100) - wait for convergence
    'random_state': 42,
    # GPU optimization for maximum accuracy
    'max_bin': 4096,                   # Maximum histogram resolution (was 2048)
    'grow_policy': 'lossguide',        # Best split selection
    'max_leaves': 1024,                # More leaves = finer granularity (was 512)
    'sampling_method': 'gradient_based',  # GPU-accelerated sampling
}

# Robustness Training (With Artificial Missing Values) - MAXIMUM ACCURACY
ROBUSTNESS_CONFIG = {
    'lambda_weight': 0.15,             # Physics constraint weight
    'n_estimators': 5000,              # Many trees (was 2500)
    'max_depth': 16,                   # Very deep (was 14)
    'learning_rate': 0.01,             # Very slow (was 0.02)
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'gamma': 0.05,                     # Minimal regularization (was 0.1)
    'reg_alpha': 0.05,                 # Minimal L1 (was 0.1)
    'reg_lambda': 1.0,                 # Moderate L2 (was 1.5)
    'tree_method': 'hist',
    'device': 'cuda',
    'early_stopping_rounds': 250,      # Very patient (was 150)
    'random_state': 42,
    # GPU optimization for maximum accuracy
    'max_bin': 4096,
    'grow_policy': 'lossguide',
    'max_leaves': 1024,
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
