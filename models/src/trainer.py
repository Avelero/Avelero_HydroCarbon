"""
Trainer Module

Multi-output XGBoost model with physics-constrained objective function.
Single model predicting all 4 targets with built-in physics constraint.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Dict, Tuple

from .utils import set_random_seed, setup_logger


class FootprintModelTrainer:
    """
    Multi-output XGBoost trainer with physics constraints.
    
    Trains a single model that outputs all 4 targets simultaneously:
    - carbon_material
    - carbon_transport
    - carbon_total
    - water_total
    
    Uses custom objective function to enforce physics constraint:
    carbon_total = carbon_material + carbon_transport
    """
    
    def __init__(
        self,
        lambda_weight: float = 0.1,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        tree_method: str = 'gpu_hist',  # Use GPU by default
        early_stopping_rounds: int = 50,
        random_state: int = 42
    ):
        """
        Initialize trainer with hyperparameters.
        
        Args:
            lambda_weight: Weight for physics constraint loss
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            tree_method: 'gpu_hist' for GPU, 'hist' for CPU
            early_stopping_rounds: Early stopping patience
            random_state: Random seed
        """
        self.lambda_weight = lambda_weight
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        
        self.model = None
        self.training_history = {}
        self.logger = setup_logger('trainer')
        
        set_random_seed(random_state)
    
    def physics_constrained_objective(
        self,
        preds: np.ndarray,
        dtrain: xgb.DMatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom XGBoost objective function with physics constraint.
        
        Loss = MSE_loss + lambda * |carbon_total - (carbon_material + carbon_transport)|²
        
        Args:
            preds: Predictions (n_samples * 4,) flattened
            dtrain: Training DMatrix
            
        Returns:
            grad, hess: Gradients and hessians for XGBoost
        """
        labels = dtrain.get_label()
        n_samples = len(labels) // 4
        
        # Reshape to (n_samples, 4)
        preds = preds.reshape(n_samples, 4)
        labels = labels.reshape(n_samples, 4)
        
        # MSE gradient and hessian for each target
        grad_mse = 2 * (preds - labels) / n_samples
        hess_mse = np.ones_like(preds) * 2 / n_samples
        
        # Physics constraint gradient
        # constraint_diff = carbon_total - (carbon_material + carbon_transport)
        constraint_diff = preds[:, 2] - (preds[:, 0] + preds[:, 1])
        
        # Add constraint gradients to relevant outputs
        grad_constraint = np.zeros_like(preds)
        grad_constraint[:, 0] = -2 * self.lambda_weight * constraint_diff / n_samples  # carbon_material
        grad_constraint[:, 1] = -2 * self.lambda_weight * constraint_diff / n_samples  # carbon_transport  
        grad_constraint[:, 2] = 2 * self.lambda_weight * constraint_diff / n_samples   # carbon_total
        # carbon_restraint[:, 3] = 0  # water_total (no constraint)
        
        # Combine gradients
        grad = grad_mse + grad_constraint
        hess = hess_mse
        
        # Return as (n_samples, 4) shape - required by XGBoost 2.1+
        return grad, hess
    
    def custom_eval_metric(
        self,
        preds: np.ndarray,
        dtrain: xgb.DMatrix
    ) -> Tuple[str, float]:
        """
        Custom evaluation metric: MAE averaged across all targets.
        """
        labels = dtrain.get_label()
        n_samples = len(labels) // 4
        
        preds = preds.reshape(n_samples, 4)
        labels = labels.reshape(n_samples, 4)
        
        # Calculate MAE for each target
        maes = []
        for i in range(4):
            mae = mean_absolute_error(labels[:, i], preds[:, i])
            maes.append(mae)
        
        # Average MAE
        avg_mae = np.mean(maes)
        
        return 'avg_mae', avg_mae
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        verbose: bool = True
    ):
        """
        Train multi-output XGBoost model with physics constraints.
        
        Args:
            X_train: Training features
            y_train: Training targets (4 columns)
            X_val: Validation features
            y_val: Validation targets (4 columns)
            verbose: Print training progress
        """
        self.logger.info("Starting multi-output XGBoost training with physics constraints...")
        self.logger.info(f"Training samples: {len(X_train):,}")
        self.logger.info(f"Validation samples: {len(X_val):,}")
        self.logger.info(f"Features: {X_train.shape[1]}")
        self.logger.info(f"Targets: {y_train.shape[1]}")
        self.logger.info(f"Tree method: {self.tree_method} (GPU={'enabled' if 'gpu' in self.tree_method else 'disabled'})")
        self.logger.info(f"Physics constraint weight: {self.lambda_weight}")
        
        # Prepare data for XGBoost
        # Flatten targets to (n_samples * 4,) for multi-output
        y_train_flat = y_train.values.flatten()
        y_val_flat = y_val.values.flatten()
        
        dtrain = xgb.DMatrix(X_train, label=y_train_flat)
        dval = xgb.DMatrix(X_val, label=y_val_flat)
        
        # XGBoost parameters
        params = {
            'max_depth': self.max_depth,
            'eta': self.learning_rate,  # learning_rate
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'alpha': self.reg_alpha,  # L1 regularization
            'lambda': self.reg_lambda,  # L2 regularization
            'tree_method': self.tree_method,
            'seed': self.random_state,
            'num_target': 4,  # 4 simultaneous outputs
            'disable_default_eval_metric': 1  # Use custom metric
        }
        
        # Training with custom objective
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}
        
        self.logger.info("Training in progress...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=self.physics_constrained_objective,
            custom_metric=self.custom_eval_metric,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=50 if verbose else False
        )
        
        self.training_history = evals_result
        
        self.logger.info(f"[OK] Training complete (best iteration: {self.model.best_iteration})")
        
        # Evaluate on validation set
        self.evaluate(X_val, y_val, dataset_name='Validation')
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for all 4 targets.
        
        Args:
            X: Features DataFrame
           
        Returns:
            DataFrame with 4 columns: carbon_material, carbon_transport, carbon_total, water_total
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        dmatrix = xgb.DMatrix(X)
        preds_flat = self.model.predict(dmatrix)
        
        # Reshape to (n_samples, 4)
        n_samples = len(X)
        preds = preds_flat.reshape(n_samples, 4)
        
        return pd.DataFrame(
            preds,
            columns=['carbon_material', 'carbon_transport', 'carbon_total', 'water_total'],
            index=X.index
        )
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        dataset_name: str = 'Test'
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on a dataset.
        
        Args:
            X: Features
            y: True targets
            dataset_name: Name for logging
            
        Returns:
            Dictionary of metrics for each target
        """
        preds = self.predict(X)
        
        metrics = {}
        target_names = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']
        
        self.logger.info(f"\n{dataset_name} Metrics:")
        self.logger.info("-" * 60)
        
        for target in target_names:
            y_true = y[target]
            y_pred = preds[target]
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            
            metrics[target] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }
            
            self.logger.info(f"{target:25s} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f} | R²: {r2:6.4f}")
        
        # Check physics constraint
        constraint_violation = np.abs(
            preds['carbon_total'] - (preds['carbon_material'] + preds['carbon_transport'])
        ).mean()
        self.logger.info(f"\nPhysics constraint violation (avg): {constraint_violation:.6f}")
        
        return metrics
    
    def save(self, path: str):
        """Save trained model"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = str(save_dir / 'xgb_model.json')
        self.model.save_model(model_path)
        
        # Save trainer config
        config = {
            'lambda_weight': self.lambda_weight,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'tree_method': self.tree_method,
            'random_state': self.random_state,
            'training_history': self.training_history
        }
        config_path = str(Path(path) / 'trainer_config.pkl')
        joblib.dump(config, config_path)
        
        self.logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load(path: str):
        """Load trained model"""
        # Load config
        config_path = str(Path(path) / 'trainer_config.pkl')
        config = joblib.load(config_path)
        
        # Create trainer
        trainer = FootprintModelTrainer(
            lambda_weight=config['lambda_weight'],
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            tree_method=config['tree_method'],
            random_state=config['random_state']
        )
        
        # Load XGBoost model
        model_path = str(Path(path) / 'xgb_model.json')
        trainer.model = xgb.Booster()
        trainer.model.load_model(model_path)
        trainer.training_history = config.get('training_history', {})
        
        trainer.logger.info(f"Model loaded from {path}")
        return trainer


if __name__ == '__main__':
    # Test trainer
    from .data_loader import load_data, get_material_dataset_path, MATERIAL_COLUMNS
    from .formula_features import add_formula_features
    from .preprocessor import FootprintPreprocessor
    
    print("Testing multi-output XGBoost trainer...")
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val = load_data(sample_size=5000)
    
    # Add formula features
    X_train = add_formula_features(X_train, MATERIAL_COLUMNS, get_material_dataset_path())
    X_val = add_formula_features(X_val, MATERIAL_COLUMNS, get_material_dataset_path())
    
    # Preprocess
    preprocessor = FootprintPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Get feature columns
    feature_cols = preprocessor.get_feature_names()
    X_train_final = X_train_processed[feature_cols]
    X_val_final = X_val_processed[feature_cols]
    
    # Train model
    trainer = FootprintModelTrainer(
        lambda_weight=0.1,
        n_estimators=100,  # Quick test
        tree_method='hist',  # CPU for test
        learning_rate=0.1
    )
    
    trainer.train(X_train_final, y_train, X_val_final, y_val)
    
    # Test save/load
    trainer.save('/tmp/test_model')
    loaded_trainer = FootprintModelTrainer.load('/tmp/test_model')
    
    print("\n[OK] Trainer test successful")
