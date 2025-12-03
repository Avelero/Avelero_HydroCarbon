"""
Comprehensive Training Script

Multi-phase training strategy optimized for maximum accuracy:
1. Baseline training on complete data
2. Evaluation and robustness testing
3. Optional retraining with augmented data if needed
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from src.data_loader import load_data, get_material_dataset_path, MATERIAL_COLUMNS
from src.formula_features import add_formula_features
from src.preprocessor import FootprintPreprocessor
from src.trainer import FootprintModelTrainer
from src.evaluator import ModelEvaluator
from src.config import BASELINE_CONFIG, ROBUSTNESS_CONFIG, MISSING_AUGMENTATION, ACCURACY_TARGETS
from src.utils import set_random_seed, setup_logger


def train_baseline(args, logger):
    """Phase 1: Train baseline model on complete data for maximum accuracy"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: BASELINE TRAINING (Complete Data - Maximum Accuracy)")
    logger.info("="*80)
    
    # Step 1: Load data
    logger.info("\nStep 1/5: Loading data...")
    X_train, y_train, X_val, y_val = load_data(sample_size=args.sample_size)
    
    # Keep raw validation for robustness testing later
    X_val_raw = X_val.copy()
    
    # Step 2: Add formula features
    logger.info("Step 2/5: Adding formula-based features...")
    material_dataset = get_material_dataset_path()
    X_train = add_formula_features(X_train, MATERIAL_COLUMNS, material_dataset)
    X_val = add_formula_features(X_val, MATERIAL_COLUMNS, material_dataset)
    
    # Step 3: Preprocess (pass y_train for target encoding)
    logger.info("Step 3/5: Preprocessing features...")
    preprocessor = FootprintPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)  # Pass targets for target encoding
    X_val_processed = preprocessor.transform(X_val)
    
    feature_cols = preprocessor.get_feature_names()
    X_train_final = X_train_processed[feature_cols]
    X_val_final = X_val_processed[feature_cols]
    
    logger.info(f"Final feature count: {len(feature_cols)}")
    logger.info(f"Training shape: {X_train_final.shape}")
    logger.info(f"Validation shape: {X_val_final.shape}")
    
    # Step 4: Train model with maximum accuracy config
    logger.info("Step 4/5: Training model (MAXIMUM ACCURACY MODE)...")
    logger.info("Configuration:")
    for key, value in BASELINE_CONFIG.items():
        if key != 'random_state':
            logger.info(f"  {key}: {value}")
    
    trainer = FootprintModelTrainer(**BASELINE_CONFIG)
    trainer.train(
        X_train_final, y_train,
        X_val_final, y_val,
        verbose=True
    )
    
    # Step 5: Save
    logger.info("Step 5/5: Saving model and preprocessor...")
    save_path = Path(args.save_dir) / 'baseline'
    save_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save(str(save_path))
    preprocessor.save(str(save_path / 'preprocessor.pkl'))
    
    logger.info(f"\n[OK] Baseline model saved to: {save_path}")
    
    return trainer, preprocessor, X_val_raw, X_val_final, y_val


def evaluate_and_test(trainer, preprocessor, X_val_raw, X_val_final, y_val, args, logger):
    """Phase 2: Comprehensive evaluation and robustness testing"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: EVALUATION & ROBUSTNESS TESTING")
    logger.info("="*80)
    
    eval_dir = Path(args.save_dir) / 'baseline' / 'evaluation'
    evaluator = ModelEvaluator(save_dir=str(eval_dir))
    
    # Baseline evaluation
    baseline_metrics = evaluator.evaluate_baseline(trainer, X_val_final, y_val)
    
    # Robustness testing
    logger.info("\nStarting robustness testing...")
    robustness_results = evaluator.test_missing_value_robustness(
        trainer,
        preprocessor,
        X_val_raw,
        y_val,
        missing_levels=MISSING_AUGMENTATION['test_missing_levels'],
        n_trials=MISSING_AUGMENTATION.get('test_n_trials', 3)  # Faster iteration
    )
    
    # Generate report
    evaluator.generate_report()
    
    # Check if retraining needed
    needs_retraining = check_if_retraining_needed(baseline_metrics, robustness_results, logger)
    
    return needs_retraining, evaluator


def check_if_retraining_needed(baseline_metrics, robustness_results, logger):
    """Determine if we need to retrain with augmented data"""
    logger.info("\n" + "="*80)
    logger.info("CHECKING AGAINST ACCURACY TARGETS")
    logger.info("="*80)
    
    targets = ACCURACY_TARGETS
    
    # Check baseline
    baseline_ok = True
    logger.info("\nBaseline Targets:")
    carbon_r2 = baseline_metrics['carbon_total']['r2']
    carbon_mae = baseline_metrics['carbon_total']['mae']
    water_mae = baseline_metrics['water_total']['mae']
    
    logger.info(f"  R² target: {targets['baseline']['r2_min']} | Actual: {carbon_r2:.4f} | {'[OK]' if carbon_r2 >= targets['baseline']['r2_min'] else '[X]'}")
    logger.info(f"  Carbon MAE target: {targets['baseline']['mae_carbon_max']} | Actual: {carbon_mae:.4f} | {'[OK]' if carbon_mae <= targets['baseline']['mae_carbon_max'] else '[X]'}")
    logger.info(f"  Water MAE target: {targets['baseline']['mae_water_max']} | Actual: {water_mae:.1f} | {'[OK]' if water_mae <= targets['baseline']['mae_water_max'] else '[X]'}")
    
    if carbon_r2 < targets['baseline']['r2_min'] or carbon_mae > targets['baseline']['mae_carbon_max']:
        baseline_ok = False
    
    # Check robustness at 30% missing
    robustness_ok = True
    results_30 = [r for r in robustness_results if abs(r['missing_pct'] - 0.3) < 0.01]
    if results_30:
        r = results_30[0]
        r2_30 = r['carbon_total_r2']
        mae_30 = r['carbon_total_mae']
        degradation = (baseline_metrics['carbon_total']['r2'] - r2_30) / baseline_metrics['carbon_total']['r2']
        
        logger.info("\nRobustness Targets (30% Missing):")
        logger.info(f"  R² target: {targets['robustness_30pct_missing']['r2_min']} | Actual: {r2_30:.4f} | {'[OK]' if r2_30 >= targets['robustness_30pct_missing']['r2_min'] else '[X]'}")
        logger.info(f"  Degradation target: <{targets['robustness_30pct_missing']['degradation_max']*100}% | Actual: {degradation*100:.1f}% | {'[OK]' if degradation <= targets['robustness_30pct_missing']['degradation_max'] else '[X]'}")
        
        if r2_30 < targets['robustness_30pct_missing']['r2_min'] or degradation > targets['robustness_30pct_missing']['degradation_max']:
            robustness_ok = False
    
    if not baseline_ok:
        logger.info("\n[WARNING] BASELINE model did not meet accuracy targets")
        logger.info("    Recommendation: Tune hyperparameters or increase n_estimators")
        return True
    elif not robustness_ok:
        logger.info("\n[WARNING] ROBUSTNESS is below targets with missing values")
        logger.info("    Recommendation: Retrain with augmented data (artificial missing values)")
        return True
    else:
        logger.info("\n[OK] All accuracy targets met! No retraining needed.")
        return False


def train_with_augmentation(args, logger):
    """Phase 3: Retrain with artificial missing values for better robustness"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: RETRAINING WITH DATA AUGMENTATION (Missing Values)")
    logger.info("="*80)
    
    # Load data
    logger.info("\nLoading data...")
    X_train, y_train, X_val, y_val = load_data(sample_size=args.sample_size)
    X_val_raw = X_val.copy()
    
    # Augment training data with artificial missing values
    logger.info("\nAugmenting training data with artificial missing values...")
    logger.info(f"  Weight missing probability: {MISSING_AUGMENTATION['weight_missing_prob']*100}%")
    logger.info(f"  Distance missing probability: {MISSING_AUGMENTATION['distance_missing_prob']*100}%")
    logger.info(f"  Materials missing probability: {MISSING_AUGMENTATION['materials_missing_prob']*100}%")
    
    X_train_aug = X_train.copy()
    n_samples = len(X_train_aug)
    
    # Randomly introduce missing values
    for idx in range(n_samples):
        if np.random.random() < MISSING_AUGMENTATION['weight_missing_prob']:
            X_train_aug.iloc[idx, X_train_aug.columns.get_loc('weight_kg')] = np.nan
        if np.random.random() < MISSING_AUGMENTATION['distance_missing_prob']:
            X_train_aug.iloc[idx, X_train_aug.columns.get_loc('total_distance_km')] = np.nan
        if np.random.random() < MISSING_AUGMENTATION['materials_missing_prob']:
            X_train_aug.iloc[idx, X_train_aug.columns.isin(MATERIAL_COLUMNS)] = 0
    
    missing_counts = {
        'weight': X_train_aug['weight_kg'].isna().sum(),
        'distance': X_train_aug['total_distance_km'].isna().sum(),
        'materials': (X_train_aug[MATERIAL_COLUMNS].sum(axis=1) == 0).sum()
    }
    logger.info(f"\nAugmented training set:")
    logger.info(f"  Samples with missing weight: {missing_counts['weight']} ({missing_counts['weight']/n_samples*100:.1f}%)")
    logger.info(f"  Samples with missing distance: {missing_counts['distance']} ({missing_counts['distance']/n_samples*100:.1f}%)")
    logger.info(f"  Samples with missing materials: {missing_counts['materials']} ({missing_counts['materials']/n_samples*100:.1f}%)")
    
    # Add formula features
    logger.info("\nAdding formula features...")
    material_dataset = get_material_dataset_path()
    X_train_aug = add_formula_features(X_train_aug, MATERIAL_COLUMNS, material_dataset)
    X_val = add_formula_features(X_val, MATERIAL_COLUMNS, material_dataset)
    
    # Preprocess (pass y_train for target encoding)
    logger.info("Preprocessing...")
    preprocessor = FootprintPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_aug, y_train)  # Pass targets for target encoding
    X_val_processed = preprocessor.transform(X_val)
    
    feature_cols = preprocessor.get_feature_names()
    X_train_final = X_train_processed[feature_cols]
    X_val_final = X_val_processed[feature_cols]
    
    # Train with robustness config
    logger.info("\nTraining model with ROBUSTNESS CONFIG (Maximum Accuracy)...")
    logger.info("Configuration:")
    for key, value in ROBUSTNESS_CONFIG.items():
        if key != 'random_state':
            logger.info(f"  {key}: {value}")
    
    trainer = FootprintModelTrainer(**ROBUSTNESS_CONFIG)
    trainer.train(X_train_final, y_train, X_val_final, y_val, verbose=True)
    
    # Save
    save_path = Path(args.save_dir) / 'robustness'
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save(str(save_path))
    preprocessor.save(str(save_path / 'preprocessor.pkl'))
    
    logger.info(f"\n[OK] Robustness model saved to: {save_path}")
    
    # Evaluate
    eval_dir = save_path / 'evaluation'
    evaluator = ModelEvaluator(save_dir=str(eval_dir))
    evaluator.evaluate_baseline(trainer, X_val_final, y_val)
    evaluator.test_missing_value_robustness(
        trainer, preprocessor, X_val_raw, y_val,
        missing_levels=MISSING_AUGMENTATION['test_missing_levels'],
        n_trials=MISSING_AUGMENTATION.get('test_n_trials', 3)
    )
    evaluator.generate_report()
    
    return trainer, preprocessor


def main():
    parser = argparse.ArgumentParser(description='Train footprint prediction model - Maximum Accuracy')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample size for testing (default: all 676K)')
    parser.add_argument('--save-dir', type=str, default='saved/max_accuracy',
                        help='Base directory to save models')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline training (use existing model)')
    parser.add_argument('--force-augmentation', action='store_true',
                        help='Force retraining with augmentation even if targets met')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    logger = setup_logger('main', 'logs/training_max_accuracy.log')
    
    logger.info("="*80)
    logger.info("MAXIMUM ACCURACY TRAINING STRATEGY")
    logger.info("="*80)
    logger.info("Priority: Maximum accuracy (cost is not a constraint)")
    logger.info(f"Dataset: {'Full 676K samples' if args.sample_size is None else f'{args.sample_size} samples'}")
    logger.info("")
    
    # Phase 1: Baseline training
    if not args.skip_baseline:
        trainer, preprocessor, X_val_raw, X_val_final, y_val = train_baseline(args, logger)
        
        # Phase 2: Evaluation
        needs_retraining, evaluator = evaluate_and_test(
            trainer, preprocessor, X_val_raw, X_val_final, y_val, args, logger
        )
    else:
        logger.info("Skipping baseline training (using existing model)")
        needs_retraining = True
    
    # Phase 3: Retraining with augmentation (if needed)
    if needs_retraining or args.force_augmentation:
        logger.info("\nProceeding to Phase 3: Augmented retraining...")
        train_with_augmentation(args, logger)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nModels saved in: {args.save_dir}")
    logger.info("  - baseline/: Model trained on complete data")
    if needs_retraining or args.force_augmentation:
        logger.info("  - robustness/: Model trained with missing value augmentation")
    logger.info("\nEvaluation reports and plots available in evaluation/ subdirectories")


if __name__ == '__main__':
    main()
