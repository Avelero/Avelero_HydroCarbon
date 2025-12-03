"""
Preprocessor Module

Feature engineering and preprocessing pipeline for ML model training.
Handles categorical encoding, missing indicators, scaling, and feature preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path
from typing import Tuple

from .data_loader import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, MATERIAL_COLUMNS


class FootprintPreprocessor:
    """
    Comprehensive preprocessing pipeline for footprint prediction.
    
    Workflow:
    1. Create interaction features (combined patterns)
    2. Create missing value indicators
    3. Encode categorical features (label + target encoding)
    4. Scale numerical features
    5. Log-transform formula features (to match log-transformed targets)
    """
    
    FORMULA_FEATURES = ['formula_carbon_material', 'formula_carbon_transport', 'formula_water_total']
    
    # Target columns for target encoding
    TARGET_ENCODING_COLS = ['carbon_material', 'carbon_transport', 'water_total']
    
    def __init__(self, log_transform_formula: bool = True):
        """
        Args:
            log_transform_formula: If True, apply log1p to formula features.
                                  Should match whether targets are log-transformed.
        """
        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.log_transform_formula = log_transform_formula
        self.formula_mins = {}  # Store mins for log transform
        self.target_encoding_maps = {}  # Store target encoding mappings
        self.is_fitted = False
    
    def log_transform_formula_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply log transformation to formula features.
        
        CRITICAL: This ensures formula features are in the same space as log-transformed targets.
        Without this, a 1.0 correlation in original space becomes ~0.53 in log space!
        
        Transform: log1p(x - min + 1) to match trainer._scale_targets()
        
        Args:
            df: DataFrame with formula features
            fit: If True, compute and store min values
            
        Returns:
            DataFrame with log-transformed formula features
        """
        if not self.log_transform_formula:
            return df
        
        df = df.copy()
        
        for col in self.FORMULA_FEATURES:
            if col not in df.columns:
                continue
            
            if fit:
                # Store min value for this feature
                self.formula_mins[col] = df[col].min()
            
            min_val = self.formula_mins.get(col, 0)
            
            # Shift to make all values positive (same as trainer._scale_targets)
            shifted = df[col] - min_val + 1.0
            
            # Apply log1p transformation (same as trainer)
            df[col] = np.log1p(shifted)
        
        if fit:
            print(f"[LOG TRANSFORM] Formula features log-transformed (matching target space)")
            for col in self.FORMULA_FEATURES:
                if col in df.columns:
                    print(f"  {col}: min={self.formula_mins.get(col, 'N/A'):.4f}")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features that capture combined patterns.
        
        These help the model learn relationships like:
        - "T-shirts typically use cotton" (category × material)
        - "Heavy leather items" (weight × material)
        - "Large footwear items" (category × weight)
        
        This is crucial for when formula features are missing - the model
        can still infer footprints from these combined patterns.
        """
        df = df.copy()
        
        # 1. Primary material info (which material dominates)
        df['primary_material_pct'] = df[MATERIAL_COLUMNS].max(axis=1)
        df['material_diversity'] = (df[MATERIAL_COLUMNS] > 0).sum(axis=1)  # Count of materials used
        
        # 2. Weight-Material interactions
        # Weight × primary material percentage (heavier items with more of one material)
        df['weight_x_primary_pct'] = df['weight_kg'].fillna(0) * df['primary_material_pct']
        
        # Weight × total material (sanity check feature)
        df['weight_x_total_material'] = df['weight_kg'].fillna(0) * df[MATERIAL_COLUMNS].sum(axis=1)
        
        # 3. Category-encoded interactions (after encoding)
        # These will be created after categorical encoding
        
        # 4. Material group aggregates (high-impact vs low-impact materials)
        # High carbon materials: leather, wool, silk, cashmere
        high_carbon_cols = ['leather_bovine', 'leather_ovine', 'wool_generic', 'wool_merino', 
                           'silk', 'cashmere', 'down_feather']
        existing_high_carbon = [c for c in high_carbon_cols if c in df.columns]
        df['high_carbon_material_pct'] = df[existing_high_carbon].sum(axis=1) if existing_high_carbon else 0
        
        # Synthetic materials: polyester, polyamide, acrylic
        synthetic_cols = ['polyester_virgin', 'polyester_recycled', 'polyamide_6', 'polyamide_66',
                         'polyamide_recycled', 'acrylic']
        existing_synthetic = [c for c in synthetic_cols if c in df.columns]
        df['synthetic_material_pct'] = df[existing_synthetic].sum(axis=1) if existing_synthetic else 0
        
        # Natural materials: cotton, linen, hemp
        natural_cols = ['cotton_conventional', 'cotton_organic', 'cotton_recycled', 
                       'linen_flax', 'hemp', 'jute']
        existing_natural = [c for c in natural_cols if c in df.columns]
        df['natural_material_pct'] = df[existing_natural].sum(axis=1) if existing_natural else 0
        
        # 5. Weight × material group interactions
        df['weight_x_high_carbon'] = df['weight_kg'].fillna(0) * df['high_carbon_material_pct']
        df['weight_x_synthetic'] = df['weight_kg'].fillna(0) * df['synthetic_material_pct']
        df['weight_x_natural'] = df['weight_kg'].fillna(0) * df['natural_material_pct']
        
        print(f"[INTERACTION] Created {10} interaction features")
        
        return df
        
    def create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary features indicating which values are missing.
        
        Critical indicators (formula-dependent):
        - weight_kg_missing
        - total_distance_km_missing  
        - materials_missing (any material column)
        
        Optional indicators (non-formula features):
        - category_missing
        - parent_category_missing
        - gender_missing
        """
        # Critical indicators
        df['weight_kg_missing'] = df['weight_kg'].isna().astype(int)
        df['total_distance_km_missing'] = df['total_distance_km'].isna().astype(int)
        
        # Materials missing = all material columns are zero
        df['materials_missing'] = (df[MATERIAL_COLUMNS].sum(axis=1) == 0).astype(int)
        
        # Optional categorical indicators
        df['category_missing'] = df['category'].isna().astype(int)
        df['parent_category_missing'] = df['parent_category'].isna().astype(int)
        df['gender_missing'] = df['gender'].isna().astype(int)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        Handles unknown categories during inference.
        
        Note: manufacturer_country excluded as has no direct physical relationship with emissions
        """
        df = df.copy()
        
        for col in CATEGORICAL_COLUMNS:
            # Fill NaN with 'Unknown'
            df[col] = df[col].fillna('Unknown')
            
            if fit:
                # Fit encoder
                self.categorical_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.categorical_encoders[col].fit_transform(df[col])
            else:
                # Transform, handling unseen categories
                def safe_transform(value):
                    if value in self.categorical_encoders[col].classes_:
                        return self.categorical_encoders[col].transform([value])[0]
                    else:
                        # Unknown category -> encode as -1
                        return -1
                
                df[f'{col}_encoded'] = df[col].apply(safe_transform)
        
        return df
    
    def target_encode_categories(
        self, 
        df: pd.DataFrame, 
        y: pd.DataFrame = None, 
        fit: bool = True,
        n_folds: int = 5
    ) -> pd.DataFrame:
        """
        Apply target encoding to categorical features with K-FOLD CROSS-VALIDATION.
        
        CRITICAL: Naive target encoding causes data leakage because you're using
        the target values of each row to compute features for that same row.
        
        Solution: Use K-Fold CV during training:
        - Split data into K folds
        - For each fold, compute encoding from the OTHER folds only
        - This prevents each row from "seeing" its own target value
        
        For inference (fit=False), use the global encoding maps.
        
        Uses smoothing to prevent overfitting on rare categories:
            encoded = (count * category_mean + smoothing * global_mean) / (count + smoothing)
        
        Args:
            df: Features DataFrame
            y: Targets DataFrame (required for fit=True)
            fit: If True, compute and store encoding maps using K-fold CV
            n_folds: Number of folds for cross-validation (default: 5)
            
        Returns:
            DataFrame with target-encoded features added
        """
        from sklearn.model_selection import KFold
        
        df = df.copy()
        smoothing = 20  # Increased smoothing for better regularization
        
        if fit:
            if y is None:
                raise ValueError("y (targets) must be provided for fit=True")
            
            self.target_encoding_maps = {}
            
            # Initialize encoded columns with NaN
            encoded_cols = {}
            for cat_col in CATEGORICAL_COLUMNS:
                for target_col in self.TARGET_ENCODING_COLS:
                    feature_name = f'{cat_col}_target_{target_col.replace("_", "")}'
                    encoded_cols[feature_name] = np.full(len(df), np.nan)
            
            # K-Fold cross-validation for encoding
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
                # For each fold, compute encoding from train_idx only
                for cat_col in CATEGORICAL_COLUMNS:
                    cat_values_train = df[cat_col].iloc[train_idx].fillna('Unknown')
                    cat_values_val = df[cat_col].iloc[val_idx].fillna('Unknown')
                    
                    for target_col in self.TARGET_ENCODING_COLS:
                        # Compute stats from training fold only
                        global_mean = y[target_col].iloc[train_idx].mean()
                        
                        category_stats = pd.DataFrame({
                            'category': cat_values_train,
                            'target': y[target_col].iloc[train_idx]
                        }).groupby('category').agg({
                            'target': ['mean', 'count']
                        })
                        category_stats.columns = ['mean', 'count']
                        
                        # Apply smoothing
                        category_stats['smoothed_mean'] = (
                            (category_stats['count'] * category_stats['mean'] + smoothing * global_mean) 
                            / (category_stats['count'] + smoothing)
                        )
                        
                        # Apply to validation fold
                        encoding_map = category_stats['smoothed_mean'].to_dict()
                        feature_name = f'{cat_col}_target_{target_col.replace("_", "")}'
                        
                        for i, idx in enumerate(val_idx):
                            cat_val = cat_values_val.iloc[i]
                            encoded_cols[feature_name][idx] = encoding_map.get(cat_val, global_mean)
            
            # Apply the K-fold encoded values
            for col_name, values in encoded_cols.items():
                df[col_name] = values
            
            # Now compute GLOBAL encoding maps for inference (using ALL data)
            # This is safe because we only use these maps during inference, not training
            for cat_col in CATEGORICAL_COLUMNS:
                self.target_encoding_maps[cat_col] = {}
                cat_values = df[cat_col].fillna('Unknown')
                
                for target_col in self.TARGET_ENCODING_COLS:
                    global_mean = y[target_col].mean()
                    
                    category_stats = pd.DataFrame({
                        'category': cat_values,
                        'target': y[target_col]
                    }).groupby('category').agg({
                        'target': ['mean', 'count']
                    })
                    category_stats.columns = ['mean', 'count']
                    
                    category_stats['smoothed_mean'] = (
                        (category_stats['count'] * category_stats['mean'] + smoothing * global_mean) 
                        / (category_stats['count'] + smoothing)
                    )
                    
                    encoding_map = category_stats['smoothed_mean'].to_dict()
                    encoding_map['__global_mean__'] = global_mean
                    self.target_encoding_maps[cat_col][target_col] = encoding_map
            
            n_features = len(CATEGORICAL_COLUMNS) * len(self.TARGET_ENCODING_COLS)
            print(f"[TARGET ENCODING] Created {n_features} target-encoded features (K-Fold CV, {n_folds} folds)")
        
        else:
            # Apply stored global encoding maps (for inference)
            for cat_col in CATEGORICAL_COLUMNS:
                cat_values = df[cat_col].fillna('Unknown')
                
                for target_col in self.TARGET_ENCODING_COLS:
                    encoding_map = self.target_encoding_maps[cat_col][target_col]
                    global_mean = encoding_map['__global_mean__']
                    
                    feature_name = f'{cat_col}_target_{target_col.replace("_", "")}'
                    df[feature_name] = cat_values.map(
                        lambda x: encoding_map.get(x, global_mean)
                    )
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        Material percentages already normalized (0-1 range).
        
        Handles missing values with median imputation before scaling.
        """
        df = df.copy()
        
        # Impute missing numerical values with median
        for col in NUMERICAL_COLUMNS:
            if df[col].isna().any():
                if fit:
                    median_val = df[col].median()
                    self.numerical_scaler.__dict__[f'{col}_median'] = median_val
                    df[col] = df[col].fillna(median_val)
                else:
                    median_val = self.numerical_scaler.__dict__.get(f'{col}_median', 0)
                    df[col] = df[col].fillna(median_val)
        
        # Scale numerical features
        if fit:
            df[NUMERICAL_COLUMNS] = self.numerical_scaler.fit_transform(df[NUMERICAL_COLUMNS])
        else:
            df[NUMERICAL_COLUMNS] = self.numerical_scaler.transform(df[NUMERICAL_COLUMNS])
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fit preprocessor on training data and transform.
        
        Args:
            X: Training features DataFrame
            y: Training targets DataFrame (required for target encoding)
            
        Returns:
            Preprocessed features DataFrame
        """
        print("Preprocessing training data (fit_transform)...")
        
        # 1. Create interaction features (before encoding, needs raw materials)
        X = self.create_interaction_features(X)
        
        # 2. Create missing indicators
        X = self.create_missing_indicators(X)
        
        # 3. Encode categorical features (label encoding)
        X = self.encode_categorical_features(X, fit=True)
        
        # 4. Target encode categories (uses y to compute mean footprints per category)
        if y is not None:
            X = self.target_encode_categories(X, y, fit=True)
        else:
            print("[WARNING] No targets provided - skipping target encoding")
        
        # 5. Scale numerical features
        X = self.scale_numerical_features(X, fit=True)
        
        # 6. Log-transform formula features (to match log-transformed targets)
        X = self.log_transform_formula_features(X, fit=True)
        
        self.is_fitted = True
        print("[OK] Preprocessing complete")
        
        # Return only numeric feature columns (not original string columns)
        # Check if formula features exist in the data
        has_formula = 'formula_carbon_material' in X.columns
        feature_cols = self.get_feature_names(include_formula_features=has_formula)
        available_cols = [c for c in feature_cols if c in X.columns]
        
        # Store which features are actually available (for get_feature_names later)
        self._has_formula_features = has_formula
        
        return X[available_cols]
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Preprocessed features DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        print("Preprocessing data (transform)...")
        
        # 1. Create interaction features (before encoding, needs raw materials)
        X = self.create_interaction_features(X)
        
        # 2. Create missing indicators
        X = self.create_missing_indicators(X)
        
        # 3. Encode categorical features (label encoding)
        X = self.encode_categorical_features(X, fit=False)
        
        # 4. Target encode categories (using fitted maps)
        if self.target_encoding_maps:
            X = self.target_encode_categories(X, fit=False)
        
        # 5. Scale numerical features
        X = self.scale_numerical_features(X, fit=False)
        
        # 6. Log-transform formula features (using fitted mins)
        X = self.log_transform_formula_features(X, fit=False)
        
        print("[OK] Preprocessing complete")
        
        # Return only numeric feature columns (not original string columns)
        # Use the same formula feature setting as during fit
        has_formula = getattr(self, '_has_formula_features', False) and 'formula_carbon_material' in X.columns
        feature_cols = self.get_feature_names(include_formula_features=has_formula)
        available_cols = [c for c in feature_cols if c in X.columns]
        return X[available_cols]
    
    # Interaction feature names (must match create_interaction_features)
    INTERACTION_FEATURES = [
        'primary_material_pct',
        'material_diversity',
        'weight_x_primary_pct',
        'weight_x_total_material',
        'high_carbon_material_pct',
        'synthetic_material_pct',
        'natural_material_pct',
        'weight_x_high_carbon',
        'weight_x_synthetic',
        'weight_x_natural',
    ]
    
    # Target-encoded feature names (category × target)
    TARGET_ENCODED_FEATURES = [
        f'{cat}_target_{target.replace("_", "")}'
        for cat in CATEGORICAL_COLUMNS
        for target in ['carbon_material', 'carbon_transport', 'water_total']
    ]
    
    def get_feature_names(self, include_formula_features: bool = True) -> list:
        """
        Get list of all feature column names after preprocessing.
        
        Args:
            include_formula_features: Whether to include formula feature names
            
        Returns:
            List of feature column names
        """
        features = []
        
        # Encoded categorical features (label encoding)
        features.extend([f'{col}_encoded' for col in CATEGORICAL_COLUMNS])
        
        # Target-encoded categorical features (mean footprint per category)
        features.extend(self.TARGET_ENCODED_FEATURES)
        
        # Numerical features
        features.extend(NUMERICAL_COLUMNS)
        
        # Material features
        features.extend(MATERIAL_COLUMNS)
        
        # Interaction features (combined patterns)
        features.extend(self.INTERACTION_FEATURES)
        
        # Missing indicators
        features.extend([
            'weight_kg_missing', 'total_distance_km_missing', 'materials_missing',
            'category_missing', 'parent_category_missing', 'gender_missing'
        ])
        
        # Formula features (if included)
        if include_formula_features:
            features.extend([
                'formula_carbon_material', 'formula_carbon_transport', 'formula_water_total'
            ])
        
        return features
    
    def save(self, path: str):
        """Save preprocessor to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: str):
        """Load preprocessor from disk"""
        preprocessor = joblib.load(path)
        print(f"Preprocessor loaded from {path}")
        return preprocessor


if __name__ == '__main__':
    # Test preprocessor
    from .data_loader import load_data, get_material_dataset_path, MATERIAL_COLUMNS
    from .formula_features import add_formula_features
    
    print("Testing preprocessor...")
    
    # Load sample data
    X_train, y_train, X_val, y_val = load_data(sample_size=1000)
    
    # Add formula features
    X_train = add_formula_features(X_train, MATERIAL_COLUMNS, get_material_dataset_path())
    X_val = add_formula_features(X_val, MATERIAL_COLUMNS, get_material_dataset_path())
    
    # Preprocess (pass y_train for target encoding)
    preprocessor = FootprintPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)  # Pass targets for target encoding
    X_val_processed = preprocessor.transform(X_val)
    
    print(f"\nProcessed training shape: {X_train_processed.shape}")
    print(f"Processed validation shape: {X_val_processed.shape}")
    
    print(f"\nFeature count: {len(preprocessor.get_feature_names())}")
    print(f"Missing values after preprocessing: {X_train_processed.isna().sum().sum()}")
    
    print("\nSample processed features:")
    feature_cols = preprocessor.get_feature_names()
    print(X_train_processed[feature_cols].head(3))
    
    # Test save/load
    preprocessor.save('/tmp/test_preprocessor.pkl')
    loaded_preprocessor = FootprintPreprocessor.load('/tmp/test_preprocessor.pkl')
    print("\n[OK] Save/load test successful")
