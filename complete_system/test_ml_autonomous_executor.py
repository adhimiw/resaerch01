"""
ML Autonomous Executor - Comprehensive Test Suite
===================================================

This test suite verifies all ML capabilities:
- Dataset analysis and preprocessing
- Model generation for classification, regression, clustering
- Model training and evaluation
- Model comparison and selection

Usage:
    cd /workspace/resaerch01/complete_system
    python test_ml_autonomous_executor.py
"""

import os
import sys
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_test(name: str, success: bool, details: str = ""):
    """Print test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} | {name}")
    if details and not success:
        print(f"       └── {details}")
    return success


def test_ml_imports():
    """Test ML module imports"""
    print_header("TEST 1: ML Module Imports")
    
    all_passed = True
    
    try:
        from core.ml_autonomous_executor import (
            MLAutonomousExecutor,
            MLCodeGenerator,
            MLDatasetAnalyzer,
            MLModelTrainer,
            MLTaskType,
            ModelType,
            EvaluationMetric,
            DatasetInfo,
            ModelResult,
            ComparisonResult,
            print_comparison_result
        )
        print_test("Import all ML classes", True)
        print(f"       └── MLTaskType: {len(MLTaskType)} types")
        print(f"       └── ModelType: {len(ModelType)} models")
        print(f"       └── EvaluationMetric: {len(EvaluationMetric)} metrics")
    except ImportError as e:
        print_test("Import all ML classes", False, str(e))
        all_passed = False
    
    # Test core imports
    try:
        from core import (
            MLAutonomousExecutor,
            MLTaskType,
            ModelType,
            DatasetInfo
        )
        print_test("Import from core package", True)
    except ImportError as e:
        print_test("Import from core package", False, str(e))
        all_passed = False
    
    return all_passed


def test_dataset_analysis():
    """Test dataset analysis functionality"""
    print_header("TEST 2: Dataset Analysis")
    
    try:
        from core.ml_autonomous_executor import MLDatasetAnalyzer
        from core.mistral_autonomous_executor import ExecutionEngine
        
        analyzer = MLDatasetAnalyzer()
        engine = ExecutionEngine()
        print_test("Initialize analyzer", True)
        
        # Find test datasets
        data_dir = PROJECT_ROOT / "tests" / "test_datasets"
        datasets = list(data_dir.glob("*.csv"))
        
        if not datasets:
            print_test("Find test datasets", False, "No datasets found")
            return False
        
        print_test(f"Found {len(datasets)} datasets", True)
        
        # Analyze first dataset
        dataset_path = str(datasets[0])
        print(f"\n  📊 Analyzing: {datasets[0].name}")
        
        info = analyzer.analyze(dataset_path)
        print_test("Analyze dataset", True)
        
        # Verify info structure
        if hasattr(info, 'shape') and info.shape:
            print_test(f"Shape detected: {info.shape}", True)
        else:
            print_test("Shape detected", False)
            return False
        
        if info.columns:
            print_test(f"Columns detected: {len(info.columns)}", True)
        else:
            print_test("Columns detected", False)
            return False
        
        if info.target_column:
            print_test(f"Target column: {info.target_column}", True)
        else:
            print_test("Target column detected", False)
            return False
        
        if info.numeric_columns:
            print_test(f"Numeric columns: {len(info.numeric_columns)}", True)
        
        # Detect task type
        task_type = analyzer.detect_task_type(info)
        print_test(f"Task type: {task_type.value}", True)
        
        return True
        
    except Exception as e:
        print_test("Dataset analysis test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_ml_code_generation():
    """Test ML code generation"""
    print_header("TEST 3: ML Code Generation")
    
    try:
        from core.ml_autonomous_executor import (
            MLCodeGenerator,
            ModelType
        )
        
        generator = MLCodeGenerator()
        print_test("Initialize code generator", True)
        
        # Test different model types
        models_to_test = [
            ModelType.RANDOM_FOREST_CLASSIFIER,
            ModelType.GRADIENT_BOOSTING_CLASSIFIER,
            ModelType.RANDOM_FOREST_REGRESSOR,
            ModelType.LOGISTIC_REGRESSION
        ]
        
        for model in models_to_test:
            code = generator.generate_code(
                model_type=model,
                data_path="test_data.csv",
                target="target"
            )
            
            if code and len(code) > 100:
                print_test(f"Generate {model.value}", True)
            else:
                print_test(f"Generate {model.value}", False, "Code too short")
                return False
        
        # Verify code quality
        if "from sklearn" in code:
            print_test("Code contains sklearn imports", True)
        
        if "model.fit" in code or "model = " in code:
            print_test("Code contains model training", True)
        
        return True
        
    except Exception as e:
        print_test("Code generation test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_single_model_training():
    """Test training a single model"""
    print_header("TEST 4: Single Model Training")
    
    try:
        from core.ml_autonomous_executor import (
            MLAutonomousExecutor,
            ModelType
        )
        from core.mistral_autonomous_executor import ExecutionEngine
        
        # Initialize
        engine = ExecutionEngine()
        ml_executor = MLAutonomousExecutor(engine)
        print_test("Initialize ML executor", True)
        
        # Find dataset
        data_dir = PROJECT_ROOT / "tests" / "test_datasets"
        datasets = list(data_dir.glob("*.csv"))
        
        if not datasets:
            print_test("Find test dataset", False, "No datasets found")
            return False
        
        dataset_path = str(datasets[0])
        dataset_name = datasets[0].name
        print_test(f"Using dataset: {dataset_name}", True)
        
        # Analyze dataset
        info = ml_executor.analyze_dataset(dataset_path)
        print_test(f"Analyze dataset: {info.shape}", True)
        
        # Train single model
        print(f"\n  🎯 Training RandomForestClassifier...")
        start_time = time.time()
        result = ml_executor.train_single_model(
            ModelType.RANDOM_FOREST_CLASSIFIER,
            dataset_path,
            info.target_column
        )
        duration = (time.time() - start_time) * 1000
        
        print_test(f"Training completed", len(result.metrics) > 0, f"{duration:.0f}ms")
        
        if len(result.metrics) > 0:
            print(f"       └── Accuracy: {result.primary_metric:.4f}")
            print(f"       └── Training time: {result.training_time_ms:.1f}ms")
            print(f"       └── CV scores: {len(result.cross_validation_scores)} folds")
        else:
            print(f"       └── Errors: {result.errors if hasattr(result, 'errors') else 'Unknown'}")
        
        return len(result.metrics) > 0
        
    except Exception as e:
        print_test("Single model training test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_model_comparison():
    """Test comparing multiple models"""
    print_header("TEST 5: Model Comparison")
    
    try:
        from core.ml_autonomous_executor import (
            MLAutonomousExecutor,
            ModelType
        )
        from core.mistral_autonomous_executor import ExecutionEngine
        
        # Initialize
        engine = ExecutionEngine()
        ml_executor = MLAutonomousExecutor(engine)
        print_test("Initialize ML executor", True)
        
        # Find dataset
        data_dir = PROJECT_ROOT / "tests" / "test_datasets"
        datasets = list(data_dir.glob("*.csv"))
        
        if not datasets:
            print_test("Find test datasets", False, "No datasets found")
            return False
        
        dataset_path = str(datasets[0])
        dataset_name = datasets[0].name
        print_test(f"Dataset: {dataset_name}", True)
        
        # Analyze
        info = ml_executor.analyze_dataset(dataset_path)
        print_test(f"Analyze: {info.shape[0]} rows, {info.shape[1]} columns", True)
        
        # Compare models
        print(f"\n  🔄 Comparing models...")
        start_time = time.time()
        result = ml_executor.compare_models(
            dataset_path,
            info.target_column,
            model_types=[
                ModelType.RANDOM_FOREST_CLASSIFIER,
                ModelType.LOGISTIC_REGRESSION
            ]
        )
        duration = (time.time() - start_time) * 1000
        
        print_test(f"Comparison completed", True, f"{duration:.0f}ms")
        
        # Print results
        from core.ml_autonomous_executor import print_comparison_result
        print("\n")
        print_comparison_result(result)
        
        # Verify comparison
        if len(result.models) >= 2:
            print_test(f"Models compared: {len(result.models)}", True)
        else:
            print_test("Multiple models compared", False)
            return False
        
        if result.best_model:
            print_test(f"Best model: {result.best_model.model_name}", True)
        
        if result.ranking:
            print_test(f"Ranking created: {len(result.ranking)} entries", True)
        
        return True
        
    except Exception as e:
        print_test("Model comparison test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_auto_train():
    """Test automatic training with dataset detection"""
    print_header("TEST 6: Auto-Train Feature")
    
    try:
        from core.ml_autonomous_executor import (
            MLAutonomousExecutor,
            MLTaskType
        )
        from core.mistral_autonomous_executor import ExecutionEngine
        
        # Initialize
        engine = ExecutionEngine()
        ml_executor = MLAutonomousExecutor(engine)
        print_test("Initialize ML executor", True)
        
        # Find dataset
        data_dir = PROJECT_ROOT / "tests" / "test_datasets"
        datasets = list(data_dir.glob("*.csv"))
        
        if not datasets:
            print_test("Find datasets", False, "No datasets found")
            return False
        
        dataset_path = str(datasets[0])
        print_test(f"Dataset: {datasets[0].name}", True)
        
        # Auto-train
        print(f"\n  🤖 Auto-training...")
        start_time = time.time()
        result = ml_executor.auto_train(dataset_path)
        duration = (time.time() - start_time) * 1000
        
        print_test(f"Auto-train completed", True, f"{duration:.0f}ms")
        
        # Verify
        if result.task_type in [MLTaskType.CLASSIFICATION, MLTaskType.REGRESSION]:
            print_test(f"Task detected: {result.task_type.value}", True)
        else:
            print_test("Task type detected", False)
            return False
        
        if result.best_model:
            print_test(f"Best: {result.best_model.model_name} ({result.best_model.primary_metric:.4f})", True)
        
        return True
        
    except Exception as e:
        print_test("Auto-train test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_multiple_datasets():
    """Test with multiple datasets"""
    print_header("TEST 7: Multiple Datasets")
    
    try:
        from core.ml_autonomous_executor import (
            MLAutonomousExecutor,
            ModelType
        )
        from core.mistral_autonomous_executor import ExecutionEngine
        
        # Initialize
        engine = ExecutionEngine()
        ml_executor = MLAutonomousExecutor(engine)
        
        # Find datasets
        data_dir = PROJECT_ROOT / "tests" / "test_datasets"
        datasets = list(data_dir.glob("*.csv"))
        
        if len(datasets) < 2:
            print_test("Multiple datasets", False, f"Only {len(datasets)} found")
            return False
        
        print_test(f"Found {len(datasets)} datasets", True)
        
        results_summary = []
        
        for i, dataset in enumerate(datasets[:3], 1):  # Test first 3
            print(f"\n  📊 Dataset {i}: {dataset.name}")
            
            info = ml_executor.analyze_dataset(str(dataset))
            
            result = ml_executor.compare_models(
                str(dataset),
                info.target_column,
                model_types=[ModelType.RANDOM_FOREST_CLASSIFIER]
            )
            
            results_summary.append({
                'dataset': dataset.name,
                'task': result.task_type.value,
                'best_model': result.best_model.model_name,
                'score': result.best_model.primary_metric
            })
            
            print(f"       └── {result.best_model.model_name}: {result.best_model.primary_metric:.4f}")
        
        print_test(f"Processed {len(results_summary)} datasets", True)
        
        return True
        
    except Exception as e:
        print_test("Multiple datasets test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  🤖 ML AUTONOMOUS EXECUTOR - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"\n  Python: {sys.version.split()[0]}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run all tests
    results.append(("ML Module Imports", test_ml_imports()))
    results.append(("Dataset Analysis", test_dataset_analysis()))
    results.append(("ML Code Generation", test_ml_code_generation()))
    results.append(("Single Model Training", test_single_model_training()))
    results.append(("Model Comparison", test_model_comparison()))
    results.append(("Auto-Train Feature", test_auto_train()))
    results.append(("Multiple Datasets", test_multiple_datasets()))
    
    # Summary
    print_header("📊 FINAL TEST SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} | {name}")
    
    print("\n" + "=" * 80)
    if passed == total:
        print(f"  🎉 ALL {total} TESTS PASSED!")
        print("\n  ✅ Your ML Autonomous Executor is production-ready!")
        print("  ✅ It can CODE, TRAIN, and COMPARE ML models autonomously!")
        print("\n  Usage Example:")
        print("    from core import MLAutonomousExecutor, ModelType")
        print("    from core.mistral_autonomous_executor import ExecutionEngine")
        print("    ")
        print("    executor = MLAutonomousExecutor(ExecutionEngine())")
        print("    result = executor.compare_models('data.csv', 'target')")
        print("    print(result.best_model)")
    else:
        print(f"  ⚠️  {total - passed} TESTS FAILED")
        print("  Please review the errors above.")
    print("=" * 80 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
