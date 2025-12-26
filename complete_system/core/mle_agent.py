"""
MLE-Agent: Automated Machine Learning Engineering Agent
Phase 3 - Universal Dataset Analysis & Model Building

Features:
- Automated EDA (Exploratory Data Analysis)
- Feature engineering and selection
- Automatic preprocessing pipeline
- Model selection and training
- Hyperparameter optimization
- Performance evaluation
- Integration with data validation (Phase 2)

Works with ANY dataset (CSV, Excel, etc.), not just competitions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
from datetime import datetime
import json

# ML libraries
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. Install with: pip install scikit-learn")

# Import Phase 2 validation
try:
    from .data_validation import DataLeakageDetector
    VALIDATION_AVAILABLE = True
except ImportError:
    try:
        from data_validation import DataLeakageDetector
        VALIDATION_AVAILABLE = True
    except ImportError:
        VALIDATION_AVAILABLE = False
        warnings.warn("Data validation not available. Run Phase 2 first.")


class MLEAgent:
    """
    Machine Learning Engineering Agent for universal dataset analysis.
    
    Automates the complete ML pipeline:
    1. Data loading and validation
    2. Exploratory Data Analysis (EDA)
    3. Feature engineering
    4. Model selection and training
    5. Evaluation and reporting
    """
    
    def __init__(
        self,
        enable_validation: bool = True,
        auto_fix_leakage: bool = True,
        verbose: bool = True
    ):
        """
        Initialize MLE-Agent.
        
        Args:
            enable_validation: Enable Phase 2 data validation
            auto_fix_leakage: Automatically fix detected leakage
            verbose: Print progress messages
        """
        self.enable_validation = enable_validation and VALIDATION_AVAILABLE
        self.auto_fix_leakage = auto_fix_leakage
        self.verbose = verbose
        
        if self.enable_validation:
            self.validator = DataLeakageDetector(strict_mode=False)
        
        self.eda_results = {}
        self.preprocessing_pipeline = {}
        self.trained_models = {}
        self.evaluation_results = {}
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
    
    def _log(self, message: str, emoji: str = "ℹ️"):
        """Print message if verbose enabled."""
        if self.verbose:
            print(f"{emoji} {message}")
    
    def load_dataset(
        self,
        file_path: str,
        target_column: Optional[str] = None,
        test_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load dataset from file (CSV, Excel, etc.).
        
        Args:
            file_path: Path to training data
            target_column: Name of target column (if known)
            test_file_path: Optional path to test data
        
        Returns:
            Dict with 'train', 'test', 'target_column', 'metadata'
        """
        self._log(f"Loading dataset from {file_path}...", "📂")
        
        # Detect file type and load
        file_path = Path(file_path)
        if file_path.suffix == '.csv':
            train_df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            train_df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        self._log(f"Loaded training data: {train_df.shape}", "✅")
        
        # Load test data if provided
        test_df = None
        if test_file_path:
            test_path = Path(test_file_path)
            if test_path.suffix == '.csv':
                test_df = pd.read_csv(test_file_path)
            elif test_path.suffix in ['.xlsx', '.xls']:
                test_df = pd.read_excel(test_file_path)
            self._log(f"Loaded test data: {test_df.shape}", "✅")
        
        # Metadata
        metadata = {
            'train_shape': train_df.shape,
            'test_shape': test_df.shape if test_df is not None else None,
            'columns': list(train_df.columns),
            'dtypes': train_df.dtypes.to_dict(),
            'missing_values': train_df.isnull().sum().to_dict(),
            'loaded_at': datetime.now().isoformat()
        }
        
        return {
            'train': train_df,
            'test': test_df,
            'target_column': target_column,
            'metadata': metadata
        }
    
    def analyze_dataset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive Exploratory Data Analysis (EDA).
        
        Args:
            data: Output from load_dataset()
        
        Returns:
            EDA results with statistics, insights, recommendations
        """
        self._log("Starting Exploratory Data Analysis (EDA)...", "🔍")
        
        train_df = data['train']
        test_df = data['test']
        target_column = data['target_column']
        
        eda = {
            'timestamp': datetime.now().isoformat(),
            'basic_stats': {},
            'column_analysis': {},
            'insights': [],
            'recommendations': [],
            'validation_report': None
        }
        
        # Basic statistics
        eda['basic_stats'] = {
            'n_rows': len(train_df),
            'n_columns': len(train_df.columns),
            'n_numeric': len(train_df.select_dtypes(include=[np.number]).columns),
            'n_categorical': len(train_df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage_mb': train_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'total_missing': train_df.isnull().sum().sum(),
            'missing_percentage': (train_df.isnull().sum().sum() / (len(train_df) * len(train_df.columns))) * 100
        }
        
        # Column-level analysis
        for col in train_df.columns:
            col_info = {
                'dtype': str(train_df[col].dtype),
                'missing_count': int(train_df[col].isnull().sum()),
                'missing_pct': float((train_df[col].isnull().sum() / len(train_df)) * 100),
                'unique_count': int(train_df[col].nunique())
            }
            
            # Numeric columns
            if train_df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                col_info.update({
                    'mean': float(train_df[col].mean()) if not train_df[col].isnull().all() else None,
                    'median': float(train_df[col].median()) if not train_df[col].isnull().all() else None,
                    'std': float(train_df[col].std()) if not train_df[col].isnull().all() else None,
                    'min': float(train_df[col].min()) if not train_df[col].isnull().all() else None,
                    'max': float(train_df[col].max()) if not train_df[col].isnull().all() else None,
                    'skewness': float(train_df[col].skew()) if not train_df[col].isnull().all() else None
                })
            
            # Categorical columns
            elif train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
                top_values = train_df[col].value_counts().head(5).to_dict()
                col_info['top_5_values'] = {str(k): int(v) for k, v in top_values.items()}
            
            eda['column_analysis'][col] = col_info
        
        # Generate insights
        insights = []
        
        # Missing data insights
        high_missing = [col for col, info in eda['column_analysis'].items() 
                       if info['missing_pct'] > 50]
        if high_missing:
            insights.append(f"High missing data: {len(high_missing)} columns with >50% missing")
        
        # Cardinality insights
        high_cardinality = [col for col, info in eda['column_analysis'].items() 
                           if info['unique_count'] > len(train_df) * 0.5]
        if high_cardinality:
            insights.append(f"High cardinality: {len(high_cardinality)} columns (potential ID columns)")
        
        # Imbalance insights
        if target_column and target_column in train_df.columns:
            target_dist = train_df[target_column].value_counts()
            if len(target_dist) == 2:  # Binary classification
                ratio = target_dist.min() / target_dist.max()
                if ratio < 0.3:
                    insights.append(f"Imbalanced target: {ratio:.2%} minority class")
        
        eda['insights'] = insights
        
        # Generate recommendations
        recommendations = []
        
        if eda['basic_stats']['missing_percentage'] > 10:
            recommendations.append("Handle missing values (imputation or removal)")
        
        if eda['basic_stats']['n_categorical'] > 0:
            recommendations.append("Encode categorical variables (one-hot or label encoding)")
        
        if high_cardinality:
            recommendations.append(f"Consider removing high-cardinality columns: {high_cardinality[:3]}")
        
        # Numeric features with high skewness
        skewed_features = [col for col, info in eda['column_analysis'].items() 
                          if 'skewness' in info and info['skewness'] is not None 
                          and abs(info['skewness']) > 2]
        if skewed_features:
            recommendations.append(f"Apply log transformation to {len(skewed_features)} skewed features")
        
        eda['recommendations'] = recommendations
        
        # Phase 2 validation if enabled
        if self.enable_validation and test_df is not None:
            self._log("Running Phase 2 data validation...", "🔒")
            
            validation_report = self.validator.check(
                train_df=train_df,
                test_df=test_df,
                target_column=target_column
            )
            
            eda['validation_report'] = validation_report
            
            if validation_report['risk_score'] >= 70:
                self._log(f"⚠️  CRITICAL leakage risk: {validation_report['risk_score']}/100", "🔴")
                
                if self.auto_fix_leakage and validation_report['can_auto_fix']:
                    self._log("Applying auto-fix...", "🔧")
                    fixed_train, fixed_test = self.validator.auto_fix(
                        train_df=train_df,
                        test_df=test_df,
                        target_column=target_column,
                        remove_duplicates=True
                    )
                    data['train'] = fixed_train
                    data['test'] = fixed_test
                    self._log("Auto-fix complete", "✅")
        
        self.eda_results = eda
        self._log(f"EDA complete: {len(insights)} insights, {len(recommendations)} recommendations", "✅")
        
        return eda
    
    def preprocess_data(
        self,
        data: Dict[str, Any],
        handle_missing: str = 'median',
        encode_categorical: str = 'onehot',
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Preprocess data for model training.
        
        Args:
            data: Output from load_dataset()
            handle_missing: 'median', 'mean', 'mode', or 'drop'
            encode_categorical: 'onehot' or 'label'
            scale_features: Apply standard scaling
        
        Returns:
            Preprocessed data with X_train, y_train, X_test, feature_names
        """
        self._log("Starting data preprocessing...", "⚙️")
        
        train_df = data['train'].copy()
        test_df = data['test'].copy() if data['test'] is not None else None
        target_column = data['target_column']
        
        # Separate features and target
        if target_column and target_column in train_df.columns:
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
        else:
            X_train = train_df
            y_train = None
        
        X_test = test_df if test_df is not None else None
        
        # Store original columns
        original_columns = X_train.columns.tolist()
        
        # 1. Handle missing values
        self._log(f"Handling missing values ({handle_missing})...", "🔧")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        
        for col in numeric_cols:
            if X_train[col].isnull().any():
                if handle_missing == 'median':
                    fill_value = X_train[col].median()
                elif handle_missing == 'mean':
                    fill_value = X_train[col].mean()
                else:
                    fill_value = 0
                
                X_train[col].fillna(fill_value, inplace=True)
                if X_test is not None and col in X_test.columns:
                    X_test[col].fillna(fill_value, inplace=True)
        
        for col in categorical_cols:
            if X_train[col].isnull().any():
                if handle_missing == 'mode':
                    fill_value = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'Unknown'
                else:
                    fill_value = 'Unknown'
                
                X_train[col].fillna(fill_value, inplace=True)
                if X_test is not None and col in X_test.columns:
                    X_test[col].fillna(fill_value, inplace=True)
        
        # 2. Encode categorical variables
        if len(categorical_cols) > 0:
            self._log(f"Encoding {len(categorical_cols)} categorical columns ({encode_categorical})...", "🔧")
            
            if encode_categorical == 'label':
                # Label encoding
                for col in categorical_cols:
                    le = LabelEncoder()
                    X_train[col] = le.fit_transform(X_train[col].astype(str))
                    if X_test is not None and col in X_test.columns:
                        X_test[col] = le.transform(X_test[col].astype(str))
            
            elif encode_categorical == 'onehot':
                # One-hot encoding
                X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
                if X_test is not None:
                    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
                    
                    # Align columns
                    missing_cols = set(X_train.columns) - set(X_test.columns)
                    for col in missing_cols:
                        X_test[col] = 0
                    
                    X_test = X_test[X_train.columns]
        
        # 3. Scale features
        if scale_features:
            self._log("Scaling features (StandardScaler)...", "🔧")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            
            if X_test is not None:
                X_test_scaled = scaler.transform(X_test)
                X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        self.preprocessing_pipeline = {
            'handle_missing': handle_missing,
            'encode_categorical': encode_categorical,
            'scale_features': scale_features,
            'original_columns': original_columns,
            'final_columns': X_train.columns.tolist()
        }
        
        self._log(f"Preprocessing complete: {X_train.shape}", "✅")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'feature_names': X_train.columns.tolist(),
            'pipeline': self.preprocessing_pipeline
        }
    
    def train_models(
        self,
        processed_data: Dict[str, Any],
        task_type: str = 'auto',
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models and compare performance.
        
        Args:
            processed_data: Output from preprocess_data()
            task_type: 'classification', 'regression', or 'auto'
            models: List of model names to train (default: all)
        
        Returns:
            Training results with models, scores, best_model
        """
        self._log("Starting model training...", "🤖")
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        if y_train is None:
            raise ValueError("Target column required for training")
        
        # Auto-detect task type
        if task_type == 'auto':
            n_unique = y_train.nunique()
            if n_unique <= 20:
                task_type = 'classification'
            else:
                task_type = 'regression'
            self._log(f"Auto-detected task: {task_type} ({n_unique} unique targets)", "🔍")
        
        # Define models to train
        if task_type == 'classification':
            all_models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            }
            scoring = 'accuracy'
        else:
            all_models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'linear_regression': LinearRegression(n_jobs=-1)
            }
            scoring = 'r2'
        
        # Select models to train
        if models:
            models_to_train = {k: v for k, v in all_models.items() if k in models}
        else:
            models_to_train = all_models
        
        # Train and evaluate each model
        results = {}
        
        for model_name, model in models_to_train.items():
            self._log(f"Training {model_name}...", "⚙️")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
            
            # Train final model
            model.fit(X_train, y_train)
            
            # Training predictions
            train_pred = model.predict(X_train)
            
            # Calculate metrics
            if task_type == 'classification':
                metrics = {
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'train_accuracy': float(accuracy_score(y_train, train_pred)),
                    'train_precision': float(precision_score(y_train, train_pred, average='weighted', zero_division=0)),
                    'train_recall': float(recall_score(y_train, train_pred, average='weighted', zero_division=0)),
                    'train_f1': float(f1_score(y_train, train_pred, average='weighted', zero_division=0))
                }
            else:
                metrics = {
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'train_r2': float(r2_score(y_train, train_pred)),
                    'train_mse': float(mean_squared_error(y_train, train_pred)),
                    'train_mae': float(mean_absolute_error(y_train, train_pred)),
                    'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred)))
                }
            
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'task_type': task_type
            }
            
            self._log(f"{model_name}: CV={metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})", "✅")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['cv_mean'])
        
        self.trained_models = results
        
        self._log(f"Training complete. Best model: {best_model_name}", "🎉")
        
        return {
            'models': results,
            'best_model': best_model_name,
            'task_type': task_type,
            'n_features': X_train.shape[1]
        }
    
    def predict(
        self,
        processed_data: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions on test data.
        
        Args:
            processed_data: Output from preprocess_data()
            model_name: Model to use (default: best model)
        
        Returns:
            Predictions and probabilities (if classification)
        """
        X_test = processed_data['X_test']
        
        if X_test is None:
            raise ValueError("Test data required for predictions")
        
        if not self.trained_models:
            raise ValueError("No trained models. Call train_models() first.")
        
        # Select model
        if model_name is None:
            # Use best model
            model_name = max(self.trained_models.keys(), 
                           key=lambda k: self.trained_models[k]['metrics']['cv_mean'])
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        self._log(f"Generating predictions with {model_name}...", "🔮")
        
        model_info = self.trained_models[model_name]
        model = model_info['model']
        task_type = model_info['task_type']
        
        # Predictions
        predictions = model.predict(X_test)
        
        result = {
            'predictions': predictions,
            'model_name': model_name,
            'task_type': task_type,
            'n_predictions': len(predictions)
        }
        
        # Probabilities for classification
        if task_type == 'classification' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)
            result['probabilities'] = probabilities
        
        self._log(f"Generated {len(predictions)} predictions", "✅")
        
        return result
    
    def generate_report(self) -> str:
        """
        Generate comprehensive analysis report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("       MLE-AGENT: AUTOMATED ML ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # EDA Results
        if self.eda_results:
            report.append("📊 EXPLORATORY DATA ANALYSIS")
            report.append("-" * 70)
            
            stats = self.eda_results['basic_stats']
            report.append(f"Dataset: {stats['n_rows']:,} rows × {stats['n_columns']} columns")
            report.append(f"Features: {stats['n_numeric']} numeric, {stats['n_categorical']} categorical")
            report.append(f"Missing: {stats['missing_percentage']:.2f}%")
            report.append("")
            
            if self.eda_results['insights']:
                report.append("💡 Key Insights:")
                for i, insight in enumerate(self.eda_results['insights'], 1):
                    report.append(f"   {i}. {insight}")
                report.append("")
            
            if self.eda_results['recommendations']:
                report.append("📝 Recommendations:")
                for i, rec in enumerate(self.eda_results['recommendations'], 1):
                    report.append(f"   {i}. {rec}")
                report.append("")
        
        # Validation Results
        if self.eda_results and self.eda_results.get('validation_report'):
            val = self.eda_results['validation_report']
            report.append("🔒 DATA VALIDATION (Phase 2)")
            report.append("-" * 70)
            report.append(f"Risk Score: {val['risk_score']}/100 ({val['severity']})")
            report.append(f"Issues: {len(val['issues'])}")
            report.append("")
        
        # Training Results
        if self.trained_models:
            report.append("🤖 MODEL TRAINING RESULTS")
            report.append("-" * 70)
            
            for model_name, model_info in self.trained_models.items():
                metrics = model_info['metrics']
                report.append(f"\n{model_name.upper()}:")
                report.append(f"   Cross-validation: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
                
                if 'train_accuracy' in metrics:
                    report.append(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
                    report.append(f"   Train F1: {metrics['train_f1']:.4f}")
                else:
                    report.append(f"   Train R²: {metrics['train_r2']:.4f}")
                    report.append(f"   Train RMSE: {metrics['train_rmse']:.4f}")
            
            report.append("")
        
        report.append("=" * 70)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_report(self, filepath: str):
        """Save report to file."""
        report = self.generate_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        self._log(f"Report saved to {filepath}", "💾")


def analyze_any_dataset(
    train_path: str,
    target_column: str,
    test_path: Optional[str] = None,
    task_type: str = 'auto'
) -> Dict[str, Any]:
    """
    Complete end-to-end analysis workflow for any dataset.
    
    Args:
        train_path: Path to training data (CSV or Excel)
        target_column: Name of target column
        test_path: Optional path to test data
        task_type: 'classification', 'regression', or 'auto'
    
    Returns:
        Complete analysis results
    """
    print("\n🚀 MLE-Agent: Starting Universal Dataset Analysis")
    print("=" * 70)
    
    # Initialize agent
    agent = MLEAgent(
        enable_validation=True,
        auto_fix_leakage=True,
        verbose=True
    )
    
    # 1. Load dataset
    data = agent.load_dataset(
        file_path=train_path,
        target_column=target_column,
        test_file_path=test_path
    )
    
    # 2. Analyze (EDA + Validation)
    eda_results = agent.analyze_dataset(data)
    
    # 3. Preprocess
    processed_data = agent.preprocess_data(
        data=data,
        handle_missing='median',
        encode_categorical='onehot',
        scale_features=True
    )
    
    # 4. Train models
    training_results = agent.train_models(
        processed_data=processed_data,
        task_type=task_type
    )
    
    # 5. Generate predictions (if test data available)
    predictions = None
    if test_path:
        predictions = agent.predict(processed_data)
    
    # 6. Generate report
    print("\n")
    print(agent.generate_report())
    
    return {
        'agent': agent,
        'data': data,
        'eda': eda_results,
        'processed_data': processed_data,
        'training_results': training_results,
        'predictions': predictions
    }


if __name__ == "__main__":
    print("MLE-Agent: Universal Machine Learning Engineering Agent")
    print("Phase 3 - Ready for any dataset analysis!")
