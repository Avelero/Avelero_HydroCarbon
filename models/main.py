"""
Main Training Script

Full pipeline to train footprint prediction model on complete dataset.
Uses GPU acceleration and all 676K training samples.
"""

import argparse
from pathlib import Path
import sys

from src.data_loader import load_data, get_material_dataset_path, MATERIAL_COLUMNS
from src.formula_features import add_formula_features
from src.preprocessor import FootprintPreprocessor
from src.trainer import FootprintModelTrainer
from src.utils import set_random_seed, setup_logger


def main():
    parser = argparse.ArgumentParser(description='Train footprint prediction model')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample size for quick testing (default: use all data)')
    parser.add_argument('--save-dir', type=str, default='saved/xgboost',
                        help='Directory to save trained model')
    parser.add_argument('--lambda-weight', type=float, default=0.1,
                        help='Weight for physics constraint (default: 0.1)')
    parser.add_argument('--n-estimators', type=int, default=500,
                        help='Number of boosting rounds (default: 500)')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='Learning rate (default: 0.05)')
    parser.add_argument('--max-depth', type=int, default=8,
                        help='Maximum tree depth (default: 8)')
    parser.add_argument('--tree-method', type=str, default='hist',
                        choices=['hist', 'approx', 'exact'],
                        help='XGBoost tree method (hist recommended for GPU)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for training (cuda for GPU, cpu for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Setup
    set_random_seed(args.seed)
    logger = setup_logger('main', 'logs/training.log')
    
    logger.info("="*80)
    logger.info("FOOTPRINT PREDICTION MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Sample size: {'ALL DATA' if args.sample_size is None else args.sample_size}")
    logger.info(f"Tree method: {args.tree_method} (GPU={'enabled' if 'gpu' in args.tree_method else 'disabled'})")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info("")
    
    # Step 1: Load data
    logger.info("Step 1/5: Loading data...")
    X_train, y_train, X_val, y_val = load_data(sample_size=args.sample_size)
    
    # Step 2: Add formula features
    logger.info("Step 2/5: Adding formula-based features...")
    material_dataset = get_material_dataset_path()
    X_train = add_formula_features(X_train, MATERIAL_COLUMNS, material_dataset)
    X_val = add_formula_features(X_val, MATERIAL_COLUMNS, material_dataset)
    
    # Step 3: Preprocess
    logger.info("Step 3/5: Preprocessing features...")
    preprocessor = FootprintPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Get feature columns
    feature_cols = preprocessor.get_feature_names()
    X_train_final = X_train_processed[feature_cols]
    X_val_final = X_val_processed[feature_cols]
    
    logger.info(f"Final feature count: {len(feature_cols)}")
    logger.info(f"Training shape: {X_train_final.shape}")
    logger.info(f"Validation shape: {X_val_final.shape}")
    
    # Step 4: Train model
    logger.info("Step 4/5: Training model...")
    trainer = FootprintModelTrainer(
        lambda_weight=args.lambda_weight,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        tree_method=args.tree_method,
        device=args.device,
        random_state=args.seed
    )
    
    trainer.train(
        X_train_final, y_train,
        X_val_final, y_val,
        early_stopping_rounds=50,
        verbose=True
    )
    
    # Step 5: Save model and preprocessor
    logger.info("Step 5/5: Saving model and preprocessor...")
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save(str(save_path))
    preprocessor.save(str(save_path / 'preprocessor.pkl'))
    
    logger.info("")
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Model saved to: {save_path}")
    logger.info(f"Preprocessor saved to: {save_path / 'preprocessor.pkl'}")
    logger.info("")
    logger.info("To use the model for predictions, run:")
    logger.info(f"  python predict.py --model-dir {args.save_dir} --input <data.csv>")
    logger.info("")


if __name__ == '__main__':
    main()
