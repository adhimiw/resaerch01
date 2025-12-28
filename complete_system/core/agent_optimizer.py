"""
Agent-Lightning: Advanced Model Optimization & Hyperparameter Tuning
Phase 4 - LightGBM, XGBoost, and Automated Optimization

Features:
- Gradient boosting models (LightGBM, XGBoost)
- Automated hyperparameter optimization (Optuna)
- Ensemble methods
- Advanced feature engineering
- Model interpretation (SHAP)
- Multi-dataset optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
from datetime import datetime
import json

# Advanced ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Install with: pip install optuna")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

# Standard ML libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import VotingClassifier, VotingRegressor

# Import Phase 3 MLE-Agent
try:
    from .mle_agent import MLEAgent
    MLE_AGENT_AVAILABLE = True
except ImportError:
    try:
        from mle_agent import MLEAgent
        MLE_AGENT_AVAILABLE = True
    except ImportError:
        MLE_AGENT_AVAILABLE = False
        warnings.warn("MLE-Agent not available. Run Phase 3 first.")


class AgentLightning:
    """
    Advanced optimization system with gradient boosting and hyperparameter tuning.
    
    Features:
    1. LightGBM & XGBoost models
    2. Optuna hyperparameter optimization
    3. Ensemble methods (voting, stacking)
    4. SHAP-based feature importance
    5. Multi-dataset optimization
    """
    
    def __init__(
        self,
        enable_optimization: bool = True,
        optimization_trials: int = 50,
        verbose: bool = True
    ):
        """
        Initialize Agent-Lightning optimizer.
        
        Args:
            enable_optimization: Enable Optuna hyperparameter tuning
            optimization_trials: Number of Optuna trials
            verbose: Print progress messages
        """
        self.enable_optimization = enable_optimization and OPTUNA_AVAILABLE
        self.optimization_trials = optimization_trials
        self.verbose = verbose
        
        self.trained_models = {}
        self.optimization_results = {}
        self.feature_importance = {}
        self.ensemble_model = None
        
        if not LIGHTGBM_AVAILABLE and not XGBOOST_AVAILABLE:
            warnings.warn("Neither LightGBM nor XGBoost available. Install at least one.")
    
    def _log(self, message: str, emoji: str = "ℹ️"):
        """Print message if verbose enabled."""
        if self.verbose:
            print(f"{emoji} {message}")
    
    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = 'lightgbm',
        task_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Use Optuna to find optimal hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: 'lightgbm', 'xgboost', or 'both'
            task_type: 'classification', 'regression', or 'auto'
        
        Returns:
            Best hyperparameters and score
        """
        if not self.enable_optimization:
            self._log("Optuna not available, using default parameters", "⚠️")
            return {'params': {}, 'score': 0.0}
        
        # Auto-detect task type
        if task_type == 'auto':
            n_unique = y_train.nunique()
            task_type = 'classification' if n_unique <= 20 else 'regression'
        
        self._log(f"Starting hyperparameter optimization ({self.optimization_trials} trials)...", "🔍")
        
        def objective(trial):
            """Optuna objective function."""
            if model_type == 'lightgbm':
                params = {
                    'objective': 'binary' if task_type == 'classification' else 'regression',
                    'metric': 'binary_logloss' if task_type == 'classification' else 'rmse',
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 12)
                }
                
                if task_type == 'classification':
                    model = lgb.LGBMClassifier(**params, n_estimators=100, random_state=42)
                    scoring = 'accuracy'
                else:
                    model = lgb.LGBMRegressor(**params, n_estimators=100, random_state=42)
                    scoring = 'r2'
            
            elif model_type == 'xgboost':
                params = {
                    'objective': 'binary:logistic' if task_type == 'classification' else 'reg:squarederror',
                    'eval_metric': 'logloss' if task_type == 'classification' else 'rmse',
                    'verbosity': 0,
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
                
                if task_type == 'classification':
                    model = xgb.XGBClassifier(**params, n_estimators=100, random_state=42)
                    scoring = 'accuracy'
                else:
                    model = xgb.XGBRegressor(**params, n_estimators=100, random_state=42)
                    scoring = 'r2'
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task_type == 'classification' else KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.optimization_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_score = study.best_value
        
        self._log(f"Optimization complete! Best score: {best_score:.4f}", "✅")
        
        return {
            'params': best_params,
            'score': float(best_score),
            'trials': len(study.trials),
            'model_type': model_type,
            'task_type': task_type
        }
    
    def train_advanced_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        task_type: str = 'auto',
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Train advanced gradient boosting models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Optional test features
            task_type: 'classification', 'regression', or 'auto'
            optimize: Use Optuna for hyperparameter tuning
        
        Returns:
            Training results with models, scores, feature importance
        """
        self._log("Starting advanced model training...", "🚀")
        
        # Auto-detect task type
        if task_type == 'auto':
            n_unique = y_train.nunique()
            task_type = 'classification' if n_unique <= 20 else 'regression'
            self._log(f"Auto-detected task: {task_type}", "🔍")
        
        results = {}
        
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            self._log("Training LightGBM...", "⚙️")
            
            if optimize and self.enable_optimization:
                opt_result = self.optimize_hyperparameters(X_train, y_train, 'lightgbm', task_type)
                params = opt_result['params']
                self.optimization_results['lightgbm'] = opt_result
            else:
                params = {}
            
            if task_type == 'classification':
                lgb_model = lgb.LGBMClassifier(**params, n_estimators=200, random_state=42, verbose=-1)
            else:
                lgb_model = lgb.LGBMRegressor(**params, n_estimators=200, random_state=42, verbose=-1)
            
            lgb_model.fit(X_train, y_train)
            
            # Evaluate
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task_type == 'classification' else KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy' if task_type == 'classification' else 'r2'
            cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            
            results['lightgbm'] = {
                'model': lgb_model,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'feature_importance': dict(zip(X_train.columns, lgb_model.feature_importances_)),
                'optimized': optimize and self.enable_optimization
            }
            
            self._log(f"LightGBM: CV={cv_scores.mean():.4f} (±{cv_scores.std():.4f})", "✅")
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            self._log("Training XGBoost...", "⚙️")
            
            if optimize and self.enable_optimization:
                opt_result = self.optimize_hyperparameters(X_train, y_train, 'xgboost', task_type)
                params = opt_result['params']
                self.optimization_results['xgboost'] = opt_result
            else:
                params = {}
            
            if task_type == 'classification':
                xgb_model = xgb.XGBClassifier(**params, n_estimators=200, random_state=42, verbosity=0)
            else:
                xgb_model = xgb.XGBRegressor(**params, n_estimators=200, random_state=42, verbosity=0)
            
            xgb_model.fit(X_train, y_train)
            
            # Evaluate
            cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            
            results['xgboost'] = {
                'model': xgb_model,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'feature_importance': dict(zip(X_train.columns, xgb_model.feature_importances_)),
                'optimized': optimize and self.enable_optimization
            }
            
            self._log(f"XGBoost: CV={cv_scores.mean():.4f} (±{cv_scores.std():.4f})", "✅")
        
        self.trained_models = results
        
        # Find best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
            self._log(f"Best model: {best_model_name} ({results[best_model_name]['cv_mean']:.4f})", "🎉")
        
        return {
            'models': results,
            'best_model': best_model_name if results else None,
            'task_type': task_type,
            'optimization_results': self.optimization_results
        }
    
    def create_ensemble(
        self,
        task_type: str = 'classification'
    ) -> Any:
        """
        Create ensemble model from trained models.
        
        Args:
            task_type: 'classification' or 'regression'
        
        Returns:
            Ensemble model (VotingClassifier or VotingRegressor)
        """
        if not self.trained_models:
            raise ValueError("No trained models. Call train_advanced_models() first.")
        
        self._log("Creating ensemble model...", "🔗")
        
        estimators = [(name, info['model']) for name, info in self.trained_models.items()]
        
        if task_type == 'classification':
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
        else:
            ensemble = VotingRegressor(estimators=estimators)
        
        self.ensemble_model = ensemble
        self._log(f"Ensemble created with {len(estimators)} models", "✅")
        
        return ensemble
    
    def compute_shap_values(
        self,
        X_train: pd.DataFrame,
        model_name: str = 'lightgbm',
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for feature importance.
        
        Args:
            X_train: Training features
            model_name: Model to explain ('lightgbm', 'xgboost')
            max_samples: Maximum samples for SHAP computation
        
        Returns:
            SHAP values and feature importance
        """
        if not SHAP_AVAILABLE:
            self._log("SHAP not available", "⚠️")
            return {}
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        self._log(f"Computing SHAP values for {model_name}...", "📊")
        
        model = self.trained_models[model_name]['model']
        
        # Sample data for faster computation
        X_sample = X_train.sample(n=min(max_samples, len(X_train)), random_state=42)
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Get mean absolute SHAP values
        if isinstance(shap_values, list):  # Multi-class
            shap_values = np.abs(shap_values[1])  # Take positive class
        else:
            shap_values = np.abs(shap_values)
        
        mean_shap = np.mean(shap_values, axis=0)
        
        # Create feature importance dict
        feature_importance = dict(zip(X_train.columns, mean_shap))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        self.feature_importance[model_name] = feature_importance
        
        self._log("SHAP analysis complete", "✅")
        
        return {
            'feature_importance': feature_importance,
            'top_5_features': list(feature_importance.keys())[:5]
        }
    
    def generate_optimization_report(self) -> str:
        """
        Generate comprehensive optimization report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("   AGENT-LIGHTNING: ADVANCED OPTIMIZATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Model Performance
        if self.trained_models:
            report.append("🚀 GRADIENT BOOSTING MODELS")
            report.append("-" * 70)
            
            for model_name, model_info in self.trained_models.items():
                report.append(f"\n{model_name.upper()}:")
                report.append(f"   Cross-validation: {model_info['cv_mean']:.4f} (±{model_info['cv_std']:.4f})")
                report.append(f"   Optimized: {'Yes ✅' if model_info['optimized'] else 'No'}")
                
                # Top features
                feat_imp = model_info['feature_importance']
                top_5 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]
                report.append(f"   Top 5 features:")
                for i, (feat, imp) in enumerate(top_5, 1):
                    report.append(f"      {i}. {feat}: {imp:.4f}")
            
            report.append("")
        
        # Optimization Results
        if self.optimization_results:
            report.append("🔍 HYPERPARAMETER OPTIMIZATION")
            report.append("-" * 70)
            
            for model_name, opt_result in self.optimization_results.items():
                report.append(f"\n{model_name.upper()}:")
                report.append(f"   Best score: {opt_result['score']:.4f}")
                report.append(f"   Trials: {opt_result['trials']}")
                report.append(f"   Best parameters:")
                for param, value in opt_result['params'].items():
                    report.append(f"      {param}: {value}")
            
            report.append("")
        
        # SHAP Results
        if self.feature_importance:
            report.append("📊 SHAP FEATURE IMPORTANCE")
            report.append("-" * 70)
            
            for model_name, feat_imp in self.feature_importance.items():
                report.append(f"\n{model_name.upper()}:")
                top_5 = list(feat_imp.items())[:5]
                for i, (feat, imp) in enumerate(top_5, 1):
                    report.append(f"   {i}. {feat}: {imp:.6f}")
            
            report.append("")
        
        report.append("=" * 70)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_report(self, filepath: str):
        """Save optimization report to file."""
        report = self.generate_optimization_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        self._log(f"Report saved to {filepath}", "💾")


def optimize_with_lightning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    task_type: str = 'auto',
    enable_optimization: bool = True,
    optimization_trials: int = 50
) -> Dict[str, Any]:
    """
    Complete optimization workflow with Agent-Lightning.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Optional test features
        task_type: 'classification', 'regression', or 'auto'
        enable_optimization: Enable Optuna tuning
        optimization_trials: Number of optimization trials
    
    Returns:
        Complete optimization results
    """
    print("\n⚡ Agent-Lightning: Starting Advanced Optimization")
    print("=" * 70)
    
    # Initialize optimizer
    optimizer = AgentLightning(
        enable_optimization=enable_optimization,
        optimization_trials=optimization_trials,
        verbose=True
    )
    
    # Train advanced models
    training_results = optimizer.train_advanced_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        task_type=task_type,
        optimize=enable_optimization
    )
    
    # Create ensemble
    if len(training_results['models']) > 1:
        ensemble = optimizer.create_ensemble(task_type=training_results['task_type'])
    
    # Compute SHAP values
    if SHAP_AVAILABLE and training_results['best_model']:
        shap_results = optimizer.compute_shap_values(
            X_train=X_train,
            model_name=training_results['best_model']
        )
    
    # Generate report
    print("\n")
    print(optimizer.generate_optimization_report())
    
    return {
        'optimizer': optimizer,
        'training_results': training_results,
        'ensemble': optimizer.ensemble_model,
        'feature_importance': optimizer.feature_importance
    }


if __name__ == "__main__":
    print("Agent-Lightning: Advanced Model Optimization System")
    print("Phase 4 - LightGBM, XGBoost, Optuna, SHAP")
