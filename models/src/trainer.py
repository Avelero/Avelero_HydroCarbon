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
        tree_method: str = 'hist',      # Histogram-based (XGBoost 2.0+)
        device: str = 'cuda',           # 'cuda' for GPU, 'cpu' for CPU (XGBoost 2.0+)
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
            tree_method: 'hist' for histogram-based (recommended)
            device: 'cuda' for GPU, 'cpu' for CPU (XGBoost 2.0+ syntax)
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
        self.device = device
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        
        self.model = None
        self.training_history = {}
        self.logger = setup_logger('trainer')
        
        # Target scaling parameters (fitted during training)
        self.target_means = None
        self.target_stds = None
        self.target_columns = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']
        
        set_random_seed(random_state)

    def _resolve_device(self) -> str:
        """
        Validate requested device and fall back to CPU if GPU is unavailable.
        XGBoost 2.0+ uses 'device' parameter instead of 'gpu_hist'.
        """
        if self.device != 'cuda':
            return self.device

        # Run a tiny one-iteration check to verify GPU support
        try:
            test_X = np.array([[0.0], [1.0]], dtype=np.float32)
            test_y = np.array([0.0, 1.0], dtype=np.float32)
            test_dmatrix = xgb.DMatrix(test_X, label=test_y)
            xgb.train(
                {
                    'objective': 'reg:squarederror',
                    'tree_method': 'hist',
                    'device': 'cuda',
                    'max_depth': 1,
                    'eta': 1,
                    'verbosity': 0
                },
                test_dmatrix,
                num_boost_round=1
            )
            self.logger.info("GPU (CUDA) device validated successfully.")
            return 'cuda'
        except xgb.core.XGBoostError as exc:
            self.logger.warning(
                "CUDA device not available. Falling back to CPU. Error: %s",
                exc
            )
            return 'cpu'

    def _scale_targets(self, y: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Scale targets using log transformation + standardization.
        
        Log transformation compresses extreme values and handles skewed distributions.
        We use log1p (log(1+x)) to handle zero values safely.
        
        Args:
            y: Target DataFrame with 4 columns
            fit: If True, compute and store scaling parameters
            
        Returns:
            Scaled targets as numpy array
        """
        # Step 1: Handle negative values by shifting (add min + small epsilon)
        y_values = y.values.copy()
        
        if fit:
            # Store minimum values for each target to handle negatives
            self.target_mins = y_values.min(axis=0)
            self.logger.info(f"Target mins: {self.target_mins}")
        
        # Shift to make all values positive (min becomes ~1)
        y_shifted = y_values - self.target_mins + 1.0
        
        # Step 2: Apply log transformation
        y_log = np.log1p(y_shifted)  # log(1 + x) for numerical stability
        
        if fit:
            # Step 3: Standardize the log-transformed values
            self.target_log_means = y_log.mean(axis=0)
            self.target_log_stds = y_log.std(axis=0)
            # Prevent division by zero
            self.target_log_stds = np.where(self.target_log_stds == 0, 1.0, self.target_log_stds)
            
            self.logger.info(f"Log-transformed means: {self.target_log_means}")
            self.logger.info(f"Log-transformed stds: {self.target_log_stds}")
        
        # Standardize
        scaled = (y_log - self.target_log_means) / self.target_log_stds
        return scaled

    def _unscale_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Unscale targets back to original scale (reverse log transform).
        
        Args:
            y_scaled: Scaled predictions (n_samples, 4)
            
        Returns:
            Unscaled predictions in original scale
        """
        # Step 1: Reverse standardization
        y_log = y_scaled * self.target_log_stds + self.target_log_means
        
        # Step 2: Reverse log transformation
        y_shifted = np.expm1(y_log)  # exp(x) - 1, inverse of log1p
        
        # Step 3: Reverse the shift
        y_original = y_shifted + self.target_mins - 1.0
        
        return y_original
    
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
        Also stores per-target metrics for detailed logging.
        """
        labels = dtrain.get_label()
        n_samples = len(labels) // 4
        
        preds = preds.reshape(n_samples, 4)
        labels = labels.reshape(n_samples, 4)
        
        # Calculate MAE and R² for each target
        target_names = ['carbon_mat', 'carbon_trans', 'carbon_tot', 'water_tot']
        maes = []
        r2s = []
        
        for i in range(4):
            mae = mean_absolute_error(labels[:, i], preds[:, i])
            r2 = r2_score(labels[:, i], preds[:, i])
            maes.append(mae)
            r2s.append(r2)
        
        # Store for detailed logging callback
        self._last_eval_metrics = {
            'maes': maes,
            'r2s': r2s,
            'target_names': target_names
        }
        
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
        # Validate GPU availability and fall back to CPU if needed
        self.device = self._resolve_device()

        self.logger.info("Starting multi-output XGBoost training with physics constraints...")
        self.logger.info(f"Training samples: {len(X_train):,}")
        self.logger.info(f"Validation samples: {len(X_val):,}")
        self.logger.info(f"Features: {X_train.shape[1]}")
        self.logger.info(f"Targets: {y_train.shape[1]}")
        self.logger.info(f"Tree method: {self.tree_method}, Device: {self.device}")
        self.logger.info(f"Physics constraint weight: {self.lambda_weight}")
        
        # Scale targets to normalize gradients across different scales
        self.logger.info("Scaling targets for training...")
        y_train_scaled = self._scale_targets(y_train, fit=True)
        y_val_scaled = self._scale_targets(y_val, fit=False)
        
        # Flatten targets to (n_samples * 4,) for multi-output
        y_train_flat = y_train_scaled.flatten()
        y_val_flat = y_val_scaled.flatten()
        
        dtrain = xgb.DMatrix(X_train, label=y_train_flat)
        dval = xgb.DMatrix(X_val, label=y_val_flat)
        
        # XGBoost parameters (XGBoost 2.0+ syntax)
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
            'device': self.device,  # 'cuda' for GPU, 'cpu' for CPU
            'seed': self.random_state,
            'num_target': 4,  # 4 simultaneous outputs
            'disable_default_eval_metric': 1  # Use custom metric
        }
        
        # Training with custom objective (or built-in if lambda_weight=0)
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}
        
        # Use custom objective only if physics constraint is enabled
        use_custom_obj = self.lambda_weight > 0
        if use_custom_obj:
            self.logger.info("Using physics-constrained custom objective")
        else:
            self.logger.info("Using standard MSE objective (physics constraint disabled)")
            params['objective'] = 'reg:squarederror'
        
        # Initialize metrics storage for callback
        self._last_eval_metrics = None
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("TRAINING PROGRESS")
        self.logger.info("="*70)
        self.logger.info(f"{'Iter':<6} | {'Train MAE':>12} | {'Val MAE':>12}")
        self.logger.info("-"*70)
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=self.physics_constrained_objective if use_custom_obj else None,
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
        preds_scaled = preds_flat.reshape(n_samples, 4)
        
        # Unscale predictions back to original scale
        preds = self._unscale_targets(preds_scaled)
        
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
        
        # Save trainer config (including target scaling parameters)
        config = {
            'lambda_weight': self.lambda_weight,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'tree_method': self.tree_method,
            'random_state': self.random_state,
            'training_history': self.training_history,
            # Log-transform scaling parameters
            'target_mins': self.target_mins,
            'target_log_means': self.target_log_means,
            'target_log_stds': self.target_log_stds
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
        
        # Load log-transform scaling parameters
        trainer.target_mins = config.get('target_mins')
        trainer.target_log_means = config.get('target_log_means')
        trainer.target_log_stds = config.get('target_log_stds')
        
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
