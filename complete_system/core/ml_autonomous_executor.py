"""
ML Autonomous Executor Extension
=================================

Machine learning capabilities for the Mistral Autonomous Executor.
Enables automatic model generation, training, evaluation, and comparison.

Features:
- CLASSIFICATION: Generate classification models
- REGRESSION: Generate regression models  
- CLUSTERING: Generate clustering models
- MODEL COMPARISON: Compare multiple models side-by-side
- FEATURE ENGINEERING: Automatic feature processing
- HYPERPARAMETER TUNING: Automatic parameter optimization
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class MLTaskType(Enum):
    """Types of ML tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    RECOMMENDATION = "recommendation"


class ModelType(Enum):
    """Types of ML models"""
    # Classification
    RANDOM_FOREST_CLASSIFIER = "RandomForestClassifier"
    GRADIENT_BOOSTING_CLASSIFIER = "GradientBoostingClassifier"
    LOGISTIC_REGRESSION = "LogisticRegression"
    SVM = "SVC"
    KNN = "KNeighborsClassifier"
    DECISION_TREE = "DecisionTreeClassifier"
    NAIVE_BAYES = "GaussianNB"
    XGBOOST = "XGBClassifier"
    
    # Regression
    RANDOM_FOREST_REGRESSOR = "RandomForestRegressor"
    GRADIENT_BOOSTING_REGRESSOR = "GradientBoostingRegressor"
    LINEAR_REGRESSION = "LinearRegression"
    RIDGE_REGRESSION = "Ridge"
    LASSO_REGRESSION = "Lasso"
    SVR = "SVR"
    KNN_REGRESSOR = "KNeighborsRegressor"
    XGBOOST_REGRESSOR = "XGBRegressor"
    
    # Clustering
    KMEANS = "KMeans"
    DBSCAN = "DBSCAN"
    HIERARCHICAL = "AgglomerativeClustering"


class EvaluationMetric(Enum):
    """ML evaluation metrics"""
    # Classification
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    ROC_AUC = "roc_auc"
    CONFUSION_MATRIX = "confusion_matrix"
    
    # Regression
    MSE = "mean_squared_error"
    RMSE = "rmse"
    MAE = "mean_absolute_error"
    R2_SCORE = "r2"
    
    # Clustering
    SILHOUETTE = "silhouette_score"
    CALINSKI = "calinski_harabasz_score"


@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    path: str
    shape: Tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    target_column: str = ""
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    missing_values: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.numeric_columns:
            self.numeric_columns = [c for c in self.columns if self.dtypes.get(c, '') in ['int64', 'float64']]
        if not self.categorical_columns:
            self.categorical_columns = [c for c in self.columns if self.dtypes.get(c, '') == 'object']


@dataclass
class ModelResult:
    """Result from training a model"""
    model_name: str
    model_type: ModelType
    task_type: MLTaskType
    metrics: Dict[str, float]
    training_time_ms: float
    prediction_time_ms: float
    model_path: str = ""
    feature_importance: Dict = field(default_factory=dict)
    confusion_matrix: Any = None
    cross_validation_scores: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def primary_metric(self) -> float:
        """Get primary metric based on task type"""
        if self.task_type == MLTaskType.CLASSIFICATION:
            return self.metrics.get('accuracy', 0.0)
        elif self.task_type == MLTaskType.REGRESSION:
            return self.metrics.get('r2', 0.0)
        elif self.task_type == MLTaskType.CLUSTERING:
            return self.metrics.get('silhouette_score', 0.0)
        return 0.0


@dataclass
class ComparisonResult:
    """Result from comparing multiple models"""
    dataset_name: str
    task_type: MLTaskType
    models: List[ModelResult]
    best_model: ModelResult
    ranking: List[Tuple[str, float]]
    summary: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class MLCodeGenerator:
    """
    Generates ML code for various tasks
    """
    
    # Templates for different model types
    TEMPLATES = {
        ModelType.RANDOM_FOREST_CLASSIFIER: '''
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import joblib
import json

# Load dataset
df = pd.read_csv('{data_path}')
y = df['{target}']
X = df.drop('{target}', axis=1)

# Handle target encoding
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Select only numeric columns for features
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols]

# Handle missing values
X = X.fillna(X.median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
start_time = time.time()
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
training_time = (time.time() - start_time) * 1000

# Evaluate
start_pred = time.time()
y_pred = model.predict(X_test)
prediction_time = (time.time() - start_pred) * 1000

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Feature importance
feature_importance = dict(zip(X.columns, model.feature_importances_))

# Save results
results = {{
    'model_name': 'RandomForestClassifier',
    'accuracy': accuracy,
    'classification_report': report,
    'confusion_matrix': cm.tolist(),
    'training_time_ms': training_time,
    'prediction_time_ms': prediction_time,
    'cv_scores': cv_scores.tolist(),
    'feature_importance': feature_importance
}}

print(json.dumps(results, indent=2))

# Save model
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved to random_forest_model.pkl")
''',

        ModelType.GRADIENT_BOOSTING_CLASSIFIER: '''
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import joblib
import json

df = pd.read_csv('{data_path}')
y = df['{target}']
X = df.drop('{target}', axis=1)

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Select only numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols]
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start_time = time.time()
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
training_time = (time.time() - start_time) * 1000

start_pred = time.time()
y_pred = model.predict(X_test)
prediction_time = (time.time() - start_pred) * 1000

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)
feature_importance = dict(zip(X.columns, model.feature_importances_))

results = {{
    'model_name': 'GradientBoostingClassifier',
    'accuracy': accuracy,
    'classification_report': report,
    'confusion_matrix': cm.tolist(),
    'training_time_ms': training_time,
    'prediction_time_ms': prediction_time,
    'cv_scores': cv_scores.tolist(),
    'feature_importance': feature_importance
}}

print(json.dumps(results, indent=2))
joblib.dump(model, 'gradient_boosting_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved to gradient_boosting_model.pkl")
''',

        ModelType.RANDOM_FOREST_REGRESSOR: '''
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import json

df = pd.read_csv('{data_path}')
y = df['{target}']
X = df.drop('{target}', axis=1)

# Select only numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols]
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start_time = time.time()
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
training_time = (time.time() - start_time) * 1000

start_pred = time.time()
y_pred = model.predict(X_test)
prediction_time = (time.time() - start_pred) * 1000

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
feature_importance = dict(zip(X.columns, model.feature_importances_))

results = {{
    'model_name': 'RandomForestRegressor',
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'r2': r2,
    'training_time_ms': training_time,
    'prediction_time_ms': prediction_time,
    'cv_scores': cv_scores.tolist(),
    'feature_importance': feature_importance
}}

print(json.dumps(results, indent=2))
joblib.dump(model, 'random_forest_regressor.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved to random_forest_regressor.pkl")
''',

        ModelType.LOGISTIC_REGRESSION: '''
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import joblib
import json

df = pd.read_csv('{data_path}')
y = df['{target}']
X = df.drop('{target}', axis=1)

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Select only numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols]
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start_time = time.time()
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
training_time = (time.time() - start_time) * 1000

start_pred = time.time()
y_pred = model.predict(X_test)
prediction_time = (time.time() - start_pred) * 1000

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

results = {{
    'model_name': 'LogisticRegression',
    'accuracy': accuracy,
    'classification_report': report,
    'confusion_matrix': cm.tolist(),
    'training_time_ms': training_time,
    'prediction_time_ms': prediction_time,
    'cv_scores': cv_scores.tolist()
}}

print(json.dumps(results, indent=2))
joblib.dump(model, 'logistic_regression.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved to logistic_regression.pkl")
''',

        ModelType.KMEANS: '''
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import json

df = pd.read_csv('{data_path}')
X = df.select_dtypes(include=[np.number]).fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

start_time = time.time()
model = KMeans(n_clusters={n_clusters}, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)
training_time = (time.time() - start_time) * 1000

silhouette = silhouette_score(X_scaled, clusters)
inertia = model.inertia_

results = {{
    'model_name': 'KMeans',
    'n_clusters': {n_clusters},
    'silhouette_score': silhouette,
    'inertia': inertia,
    'training_time_ms': training_time,
    'cluster_distribution': dict(zip(*np.unique(clusters, return_counts=True)))
}}

print(json.dumps(results, indent=2))
import joblib
joblib.dump(model, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved to kmeans_model.pkl")
'''
    }
    
    @classmethod
    def generate_code(
        cls,
        model_type: ModelType,
        data_path: str,
        target: str = "target",
        n_clusters: int = 3
    ) -> str:
        """Generate ML code for a specific model type"""
        template = cls.TEMPLATES.get(model_type, "# Model template not found")
        
        code = template.format(
            data_path=data_path,
            target=target,
            n_clusters=n_clusters
        )
        
        return code


class MLDatasetAnalyzer:
    """
    Analyzes datasets for ML suitability
    """
    
    @staticmethod
    def analyze(data_path: str) -> DatasetInfo:
        """
        Analyze a dataset and return information
        
        Args:
            data_path: Path to CSV file
            
        Returns:
            DatasetInfo with analysis results
        """
        import pandas as pd
        
        df = pd.read_csv(data_path)
        
        # Basic info
        info = DatasetInfo(
            name=Path(data_path).stem,
            path=data_path,
            shape=df.shape,
            columns=list(df.columns),
            dtypes=df.dtypes.astype(str).to_dict()
        )
        
        # Identify target (heuristic: last column or named 'target')
        if 'target' in df.columns:
            info.target_column = 'target'
        elif 'label' in df.columns:
            info.target_column = 'label'
        elif 'class' in df.columns:
            info.target_column = 'class'
        else:
            info.target_column = df.columns[-1]
        
        # Categorize columns
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                info.numeric_columns.append(col)
                # Check missing values
                missing = df[col].isnull().sum()
                if missing > 0:
                    info.missing_values[col] = int(missing)
            elif df[col].dtype == 'object':
                info.categorical_columns.append(col)
        
        return info
    
    @staticmethod
    def detect_task_type(info: DatasetInfo) -> MLTaskType:
        """Detect the type of ML task based on dataset"""
        if info.target_column in info.categorical_columns:
            # Check number of unique values
            import pandas as pd
            df = pd.read_csv(info.path)
            n_unique = df[info.target_column].nunique()
            
            if n_unique <= 10:
                return MLTaskType.CLASSIFICATION
            else:
                return MLTaskType.CLUSTERING
        else:
            return MLTaskType.REGRESSION


class MLModelTrainer:
    """
    Trains and evaluates ML models
    """
    
    def __init__(self, execution_engine):
        """
        Initialize trainer
        
        Args:
            execution_engine: ExecutionEngine instance for running code
        """
        self.execution_engine = execution_engine
        self.code_generator = MLCodeGenerator()
    
    def train_model(
        self,
        model_type: ModelType,
        data_path: str,
        target: str = "target"
    ) -> ModelResult:
        """
        Train a single model and return results
        
        Args:
            model_type: Type of model to train
            data_path: Path to dataset
            target: Target column name
            
        Returns:
            ModelResult with metrics and information
        """
        # Generate code
        code = self.code_generator.generate_code(
            model_type=model_type,
            data_path=data_path,
            target=target
        )
        
        # Execute code
        output, success, errors = self.execution_engine.execute(code, timeout=120.0)
        
        if success:
            # Parse results
            try:
                # Extract JSON from output
                json_start = output.find('{')
                json_end = output.rfind('}') + 1
                json_str = output[json_start:json_end]
                results = json.loads(json_str)
                
                return ModelResult(
                    model_name=results.get('model_name', model_type.value),
                    model_type=model_type,
                    task_type=MLDatasetAnalyzer.detect_task_type(
                        MLDatasetAnalyzer.analyze(data_path)
                    ),
                    metrics=results,
                    training_time_ms=results.get('training_time_ms', 0),
                    prediction_time_ms=results.get('prediction_time_ms', 0),
                    feature_importance=results.get('feature_importance', {}),
                    cross_validation_scores=results.get('cv_scores', [])
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse results: {e}")
                return ModelResult(
                    model_name=model_type.value,
                    model_type=model_type,
                    task_type=MLDatasetAnalyzer.detect_task_type(
                        MLDatasetAnalyzer.analyze(data_path)
                    ),
                    metrics={},
                    training_time_ms=0,
                    prediction_time_ms=0,
                    errors=[f"JSON parse error: {str(e)}"]
                )
        
        # Execution failed
        return ModelResult(
            model_name=model_type.value,
            model_type=model_type,
            task_type=MLDatasetAnalyzer.detect_task_type(
                MLDatasetAnalyzer.analyze(data_path)
            ),
            metrics={},
            training_time_ms=0,
            prediction_time_ms=0,
            errors=errors
        )
    
    def compare_models(
        self,
        data_path: str,
        target: str = "target",
        model_types: List[ModelType] = None
    ) -> ComparisonResult:
        """
        Compare multiple models on the same dataset
        
        Args:
            data_path: Path to dataset
            target: Target column name
            model_types: List of models to compare
            
        Returns:
            ComparisonResult with ranking and summary
        """
        if model_types is None:
            # Default models based on task type
            task_type = MLDatasetAnalyzer.detect_task_type(
                MLDatasetAnalyzer.analyze(data_path)
            )
            
            if task_type == MLTaskType.CLASSIFICATION:
                model_types = [
                    ModelType.RANDOM_FOREST_CLASSIFIER,
                    ModelType.GRADIENT_BOOSTING_CLASSIFIER,
                    ModelType.LOGISTIC_REGRESSION
                ]
            elif task_type == MLTaskType.REGRESSION:
                model_types = [
                    ModelType.RANDOM_FOREST_REGRESSOR,
                    ModelType.GRADIENT_BOOSTING_REGRESSOR
                ]
            else:
                model_types = [ModelType.KMEANS]
        
        # Analyze dataset
        info = MLDatasetAnalyzer.analyze(data_path)
        task_type = MLDatasetAnalyzer.detect_task_type(info)
        
        logger.info(f"Comparing {len(model_types)} models on {info.name}")
        logger.info(f"Task type: {task_type.value}")
        
        # Train all models
        results = []
        for model_type in model_types:
            logger.info(f"Training {model_type.value}...")
            
            result = self.train_model(model_type, data_path, target)
            results.append(result)
            
            logger.info(f"  → Accuracy/R2: {result.primary_metric:.4f}")
        
        # Find best model
        best_model = max(results, key=lambda r: r.primary_metric)
        
        # Create ranking
        ranking = [(r.model_name, r.primary_metric) for r in results]
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        # Summary
        summary = {
            'dataset': info.name,
            'task_type': task_type.value,
            'total_models': len(results),
            'best_model': best_model.model_name,
            'best_score': best_model.primary_metric,
            'avg_training_time_ms': sum(r.training_time_ms for r in results) / len(results)
        }
        
        return ComparisonResult(
            dataset_name=info.name,
            task_type=task_type,
            models=results,
            best_model=best_model,
            ranking=ranking,
            summary=summary
        )


class MLAutonomousExecutor:
    """
    Main ML autonomous executor combining all components
    """
    
    def __init__(self, execution_engine):
        """
        Initialize ML executor
        
        Args:
            execution_engine: ExecutionEngine instance
        """
        self.execution_engine = execution_engine
        self.trainer = MLModelTrainer(execution_engine)
        self.analyzer = MLDatasetAnalyzer()
    
    def analyze_dataset(self, data_path: str) -> DatasetInfo:
        """Analyze a dataset and return information"""
        return self.analyzer.analyze(data_path)
    
    def train_single_model(
        self,
        model_type: ModelType,
        data_path: str,
        target: str = "target"
    ) -> ModelResult:
        """Train a single model"""
        return self.trainer.train_model(model_type, data_path, target)
    
    def compare_models(
        self,
        data_path: str,
        target: str = "target",
        model_types: List[ModelType] = None
    ) -> ComparisonResult:
        """Compare multiple models"""
        return self.trainer.compare_models(data_path, target, model_types)
    
    def auto_train(
        self,
        data_path: str,
        target: str = "target"
    ) -> ComparisonResult:
        """
        Automatically train and compare models
        
        Args:
            data_path: Path to dataset
            target: Target column name
            
        Returns:
            ComparisonResult with best model and ranking
        """
        # Analyze dataset
        info = self.analyze_dataset(data_path)
        task_type = self.analyzer.detect_task_type(info)
        
        logger.info(f"Auto-training on {info.name}")
        logger.info(f"Shape: {info.shape}")
        logger.info(f"Target: {target}")
        logger.info(f"Task: {task_type.value}")
        
        # Compare models
        return self.compare_models(data_path, target)


def print_comparison_result(result: ComparisonResult):
    """Pretty print comparison result"""
    print("\n" + "=" * 80)
    print(f"📊 MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n📁 Dataset: {result.dataset_name}")
    print(f"🎯 Task Type: {result.task_type.value}")
    print(f"📈 Models Compared: {len(result.models)}")
    
    print(f"\n🏆 RANKING:")
    print("-" * 60)
    for i, (name, score) in enumerate(result.ranking, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f" {medal} {i}. {name}: {score:.4f}")
    
    print(f"\n✅ BEST MODEL: {result.best_model.model_name}")
    print(f"   Primary Metric: {result.best_model.primary_metric:.4f}")
    print(f"   Training Time: {result.best_model.training_time_ms:.1f}ms")
    
    if result.best_model.cross_validation_scores:
        cv_mean = sum(result.best_model.cross_validation_scores) / len(result.best_model.cross_validation_scores)
        print(f"   CV Score Mean: {cv_mean:.4f}")
    
    if result.best_model.feature_importance:
        print(f"\n📊 Top Features:")
        sorted_features = sorted(
            result.best_model.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for feat, imp in sorted_features:
            print(f"   • {feat}: {imp:.4f}")
    
    print("\n" + "=" * 80)


# Demo and testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from core.mistral_autonomous_executor import ExecutionEngine
    
    print("=" * 80)
    print("🤖 ML AUTONOMOUS EXECUTOR - DEMO")
    print("=" * 80)
    
    # Initialize
    executor = MLAutonomousExecutor(ExecutionEngine())
    
    # Check for demo datasets
    data_dir = Path(__file__).parent.parent / "tests" / "test_datasets"
    datasets = list(data_dir.glob("*.csv"))
    
    if datasets:
        print(f"\n📁 Found {len(datasets)} datasets for testing")
        
        for dataset in datasets[:2]:  # Test first 2
            print(f"\n{'='*60}")
            print(f"🧪 Testing with: {dataset.name}")
            print(f"{'='*60}")
            
            # Analyze
            info = executor.analyze_dataset(dataset)
            print(f"   Shape: {info.shape}")
            print(f"   Target: {info.target_column}")
            print(f"   Numeric: {len(info.numeric_columns)} columns")
            
            # Auto-train
            result = executor.auto_train(str(dataset), info.target_column)
            print_comparison_result(result)
    else:
        print("\n⚠️ No test datasets found")
        print("   Add CSV files to: complete_system/tests/test_datasets/")
    
    print("\n✅ ML Autonomous Executor ready!")
