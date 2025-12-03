"""
Evaluator Module

Comprehensive evaluation including missing value robustness testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from typing import Dict, List, Tuple
import json

from .utils import setup_logger


class ModelEvaluator:
    """
    Comprehensive model evaluation including robustness testing.
    """
    
    def __init__(self, save_dir: str = 'evaluation'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('evaluator')
        self.results = {}
    
    def evaluate_baseline(
        self,
        trainer,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame
    ) -> Dict:
        """
        Evaluate model on clean validation set (no missing values).
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("BASELINE EVALUATION (Complete Data)")
        self.logger.info("="*80)
        
        preds = trainer.predict(X_val)
        
        metrics = {}
        targets = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']
        
        for target in targets:
            y_true = y_val[target]
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
            
            self.logger.info(f"{target:25s} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f} | R²: {r2:6.4f} | MAPE: {mape:6.2f}%")
        
        # Physics constraint check
        constraint_violation = np.abs(
            preds['carbon_total'] - (preds['carbon_material'] + preds['carbon_transport'])
        ).mean()
        self.logger.info(f"\nPhysics constraint violation (avg): {constraint_violation:.6f} kg CO2e")
        
        metrics['constraint_violation'] = constraint_violation
        self.results['baseline'] = metrics
        
        # Save predictions for analysis
        comparison_df = pd.DataFrame({
            'actual_carbon_material': y_val['carbon_material'],
            'pred_carbon_material': preds['carbon_material'],
            'actual_carbon_transport': y_val['carbon_transport'],
            'pred_carbon_transport': preds['carbon_transport'],
            'actual_carbon_total': y_val['carbon_total'],
            'pred_carbon_total': preds['carbon_total'],
            'actual_water': y_val['water_total'],
            'pred_water': preds['water_total'],
        })
        comparison_df.to_csv(self.save_dir / 'baseline_predictions.csv', index=False)
        
        return metrics
    
    def test_missing_value_robustness(
        self,
        trainer,
        preprocessor,
        X_val_raw: pd.DataFrame,  # Before preprocessing
        y_val: pd.DataFrame,
        missing_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        n_trials: int = 5
    ) -> Dict:
        """
        Test model robustness by artificially introducing missing values.
        
        Args:
            trainer: Trained model
            preprocessor: Fitted preprocessor
            X_val_raw: Raw validation features (before preprocessing)
            y_val: Validation targets
            missing_levels: List of missing percentages to test
            n_trials: Number of random trials per missing level
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("MISSING VALUE ROBUSTNESS TESTING")
        self.logger.info("="*80)
        
        from .data_loader import MATERIAL_COLUMNS, get_material_dataset_path
        from .formula_features import add_formula_features
        
        results = []
        
        for missing_pct in missing_levels:
            self.logger.info(f"\nTesting with {missing_pct*100:.0f}% missing values...")
            
            trial_metrics = []
            
            for trial in range(n_trials):
                # Create copy and introduce missing values
                X_corrupted = X_val_raw.copy()
                
                if missing_pct > 0:
                    n_samples = len(X_corrupted)
                    n_corrupt = int(n_samples * missing_pct)
                    
                    # Randomly select samples to corrupt
                    corrupt_idx = np.random.choice(n_samples, n_corrupt, replace=False)
                    
                    # Randomly assign corruption type to each sample (VECTORIZED)
                    corruption_types = np.random.choice(['weight', 'distance', 'materials'], size=n_corrupt)
                    
                    # Get actual DataFrame indices
                    df_indices = X_corrupted.index[corrupt_idx]
                    
                    # Apply corruptions in bulk (MUCH faster than row-by-row)
                    weight_mask = corruption_types == 'weight'
                    distance_mask = corruption_types == 'distance'
                    materials_mask = corruption_types == 'materials'
                    
                    if weight_mask.any():
                        X_corrupted.loc[df_indices[weight_mask], 'weight_kg'] = np.nan
                    if distance_mask.any():
                        X_corrupted.loc[df_indices[distance_mask], 'total_distance_km'] = np.nan
                    if materials_mask.any():
                        X_corrupted.loc[df_indices[materials_mask], MATERIAL_COLUMNS] = 0
                
                # Add formula features (will be NaN where data missing)
                X_corrupted = add_formula_features(
                    X_corrupted,
                    MATERIAL_COLUMNS,
                    get_material_dataset_path()
                )
                
                # Preprocess - transform() returns only the available feature columns
                X_final = preprocessor.transform(X_corrupted)
                
                # Predict
                preds = trainer.predict(X_final)
                
                # Calculate metrics
                targets = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']
                trial_result = {'missing_pct': missing_pct, 'trial': trial}
                
                for target in targets:
                    mae = mean_absolute_error(y_val[target], preds[target])
                    r2 = r2_score(y_val[target], preds[target])
                    trial_result[f'{target}_mae'] = mae
                    trial_result[f'{target}_r2'] = r2
                
                trial_metrics.append(trial_result)
            
            # Average across trials
            avg_metrics = {}
            for target in targets:
                avg_mae = np.mean([m[f'{target}_mae'] for m in trial_metrics])
                avg_r2 = np.mean([m[f'{target}_r2'] for m in trial_metrics])
                avg_metrics[f'{target}_mae'] = avg_mae
                avg_metrics[f'{target}_r2'] = avg_r2
            
            results.append({
                'missing_pct': missing_pct,
                **avg_metrics
            })
            
            self.logger.info(f"  Avg MAE (carbon_total): {avg_metrics['carbon_total_mae']:.4f}")
            self.logger.info(f"  Avg R² (carbon_total): {avg_metrics['carbon_total_r2']:.4f}")
        
        # Save results
        robustness_df = pd.DataFrame(results)
        robustness_df.to_csv(self.save_dir / 'robustness_results.csv', index=False)
        
        self.results['robustness'] = results
        self.plot_robustness_curves(robustness_df)
        
        return results
    
    def plot_robustness_curves(self, robustness_df: pd.DataFrame):
        """Plot robustness curves showing performance degradation"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Missing Value Robustness Analysis', fontsize=16, fontweight='bold')
        
        targets = ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']
        
        for idx, target in enumerate(targets):
            ax = axes[idx // 2, idx % 2]
            
            # MAE plot
            ax.plot(robustness_df['missing_pct'] * 100,
                   robustness_df[f'{target}_mae'],
                   marker='o', linewidth=2, markersize=8, label='MAE')
            
            ax2 = ax.twinx()
            ax2.plot(robustness_df['missing_pct'] * 100,
                    robustness_df[f'{target}_r2'],
                    marker='s', linewidth=2, markersize=8, color='orange', label='R²')
            
            ax.set_xlabel('Missing Data (%)', fontsize=11)
            ax.set_ylabel('MAE', fontsize=11, color='blue')
            ax2.set_ylabel('R² Score', fontsize=11, color='orange')
            ax.set_title(target.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'robustness_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"\nRobustness plot saved to {self.save_dir / 'robustness_curves.png'}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report_path = self.save_dir / 'evaluation_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"\nEvaluation report saved to {report_path}")
        
        # Print summary
        if 'baseline' in self.results:
            self.logger.info("\n" + "="*80)
            self.logger.info("EVALUATION SUMMARY")
            self.logger.info("="*80)
            
            baseline = self.results['baseline']
            self.logger.info("\nBaseline Performance (Complete Data):")
            for target in ['carbon_material', 'carbon_transport', 'carbon_total', 'water_total']:
                if target in baseline:
                    self.logger.info(f"  {target}: R²={baseline[target]['r2']:.4f}, MAE={baseline[target]['mae']:.4f}")
            
            if 'robustness' in self.results:
                self.logger.info("\nRobustness (30% Missing Data):")
                results_30 = [r for r in self.results['robustness'] if abs(r['missing_pct'] - 0.3) < 0.01]
                if results_30:
                    r = results_30[0]
                    self.logger.info(f"  carbon_total: R²={r['carbon_total_r2']:.4f}, MAE={r['carbon_total_mae']:.4f}")


if __name__ == '__main__':
    print("Evaluator module ready. Use with trained model for comprehensive evaluation.")
