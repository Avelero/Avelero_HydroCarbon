"""
Data Loader Module

Loads preprocessed CSV data from data_splitter output with one-hot encoded materials.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# Column definitions
TARGET_COLUMNS = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']

# Material columns (34 one-hot encoded columns)
MATERIAL_COLUMNS = [
    'acrylic', 'cashmere', 'coated_fabric_pu', 'cotton_conventional', 'cotton_organic',
    'cotton_recycled', 'down_feather', 'down_synthetic', 'elastane', 'eva', 'hemp',
    'jute', 'leather_bovine', 'leather_ovine', 'leather_synthetic', 'linen_flax',
    'lyocell_tencel', 'metal_brass', 'metal_gold', 'metal_silver', 'metal_steel',
    'modal', 'natural_rubber', 'polyamide_6', 'polyamide_66', 'polyamide_recycled',
    'polyester_recycled', 'polyester_virgin', 'rubber_synthetic', 'silk', 'tpu',
    'viscose', 'wool_generic', 'wool_merino'
]

# Categorical columns
CATEGORICAL_COLUMNS = ['gender', 'parent_category', 'category']

# Numerical columns
NUMERICAL_COLUMNS = ['weight_kg', 'total_distance_km']

# All feature columns (excluding product_name and manufacturer_country per plan)
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + MATERIAL_COLUMNS


def load_data(
    train_path: str = '/home/tr4moryp/Projects/bulk_product_generator/data/data_splitter/output/train.csv',
    val_path: str = '/home/tr4moryp/Projects/bulk_product_generator/data/data_splitter/output/validate.csv',
    sample_size: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation data from CSV files.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV  
        sample_size: If set, load only this many rows (for quick testing)
        
    Returns:
        X_train, y_train, X_val, y_val
    """
    print(f"Loading training data from {train_path}...")
    if sample_size:
        print(f"  [Quick test mode: loading only {sample_size} samples]")
        train_df = pd.read_csv(train_path, nrows=sample_size, on_bad_lines='warn', quotechar='"')
        val_df = pd.read_csv(val_path, nrows=sample_size // 4, on_bad_lines='warn', quotechar='"')
    else:
        train_df = pd.read_csv(train_path, on_bad_lines='warn', quotechar='"')
        val_df = pd.read_csv(val_path, on_bad_lines='warn', quotechar='"')
    
    print(f"  Loaded {len(train_df):,} training samples")
    print(f"  Loaded {len(val_df):,} validation samples")
    
    # Validate columns exist
    missing_features = set(FEATURE_COLUMNS) - set(train_df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    missing_targets = set(TARGET_COLUMNS) - set(train_df.columns)
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    
    # Split features and targets
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMNS].copy()
    
    X_val = val_df[FEATURE_COLUMNS].copy()
    y_val = val_df[TARGET_COLUMNS].copy()
    
    # Data validation
    print(f"\n✓ Data loaded successfully")
    print(f"  Features: {len(FEATURE_COLUMNS)} ({len(CATEGORICAL_COLUMNS)} categorical, "
          f"{len(NUMERICAL_COLUMNS)} numerical, {len(MATERIAL_COLUMNS)} materials)")
    print(f"  Targets: {len(TARGET_COLUMNS)}")
    print(f"  Missing values in training:")
    for col in NUMERICAL_COLUMNS:
        missing_pct = (X_train[col].isna().sum() / len(X_train)) * 100
        if missing_pct > 0:
            print(f"    {col}: {missing_pct:.1f}%")
    
    return X_train, y_train, X_val, y_val


def get_material_dataset_path():
    """Get path to material emission factors dataset"""
    return '/home/tr4moryp/Projects/bulk_product_generator/data/data_calculations/input/material_dataset_final.csv'


if __name__ == '__main__':
    # Test the data loader
    X_train, y_train, X_val, y_val = load_data(sample_size=1000)
    
    print("\nSample features:")
    print(X_train.head(2))
    print("\nSample targets:")
    print(y_train.head(2))
    print("\nData types:")
    print(X_train.dtypes.value_counts())
