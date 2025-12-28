"""
Test Suite for Universal Data Science Orchestrator (Phase 5)
Tests end-to-end integration of all 4 backend phases
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from complete_system.orchestrator import UniversalDataScienceOrchestrator, quick_analyze


def create_test_dataset(task_type='classification', n_samples=500, n_features=10):
    """Create synthetic test dataset."""
    if task_type == 'classification':
        # Ensure informative + redundant < n_features
        n_informative = min(max(n_features - 2, 2), n_features)
        n_redundant = min(1, n_features - n_informative - 1)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=2,
            random_state=42,
            flip_y=0.1
        )
    else:  # regression
        n_informative = min(max(n_features - 1, 2), n_features)
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=10.0,
            random_state=42
        )
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def test_orchestrator_initialization():
    """Test 1: Orchestrator initialization with all components."""
    print("\n" + "="*70)
    print("Test 1: Orchestrator Initialization")
    print("="*70)
    
    try:
        # Test with all phases enabled
        orchestrator = UniversalDataScienceOrchestrator(
            enable_rag=True,
            enable_validation=True,
            enable_mle=True,
            enable_optimization=True,
            optimization_trials=10,
            verbose=False
        )
        
        print("✅ All phases initialized successfully")
        
        # Test selective initialization
        orchestrator_minimal = UniversalDataScienceOrchestrator(
            enable_rag=False,
            enable_validation=False,
            enable_mle=True,
            enable_optimization=False,
            verbose=False
        )
        
        print("✅ Selective initialization working")
        
        return True
    
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_classification_workflow():
    """Test 2: Complete workflow with classification dataset."""
    print("\n" + "="*70)
    print("Test 2: Classification Workflow")
    print("="*70)
    
    try:
        # Create test dataset
        df = create_test_dataset('classification', n_samples=500, n_features=8)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            df.to_csv(f, index=False)
        
        try:
            # Initialize orchestrator
            orchestrator = UniversalDataScienceOrchestrator(
                enable_rag=False,  # Disable RAG for faster testing
                enable_validation=True,
                enable_mle=True,
                enable_optimization=True,
                optimization_trials=5,  # Reduced for faster testing
                verbose=True
            )
            
            # Run complete analysis
            results = orchestrator.analyze_dataset(
                file_path=temp_file,
                target_column='target',
                test_size=0.2,
                random_state=42,
                run_validation=True,
                run_optimization=True,
                store_results=False  # Skip ChromaDB storage
            )
            
            # Verify results
            assert 'workflow_id' in results
            assert 'phases' in results
            assert 'mle_agent' in results['phases']
            assert 'optimization' in results['phases']
            
            # Check MLE phase
            mle = results['phases']['mle_agent']
            assert mle['task_type'] == 'classification'
            assert mle['best_score'] > 0.5  # Should beat random
            
            # Check optimization phase
            opt = results['phases']['optimization']
            assert 'lightgbm' in opt['models_trained']
            assert 'xgboost' in opt['models_trained']
            assert opt['ensemble_test_score'] > 0.5
            
            print(f"\n✅ Classification Analysis Complete")
            print(f"   Task Type: {mle['task_type']}")
            print(f"   Best Model (Phase 3): {mle['best_model']} - {mle['best_score']:.4f}")
            print(f"   Best Model (Phase 4): {opt['best_model']} - {opt['best_cv_score']:.4f}")
            print(f"   Ensemble Test Score: {opt['ensemble_test_score']:.4f}")
            print(f"   Top Features: {opt['top_features'][:3]}")
            
            return True
        
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    except Exception as e:
        print(f"❌ Classification workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_regression_workflow():
    """Test 3: Complete workflow with regression dataset."""
    print("\n" + "="*70)
    print("Test 3: Regression Workflow")
    print("="*70)
    
    try:
        # Create test dataset
        df = create_test_dataset('regression', n_samples=500, n_features=8)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            df.to_csv(f, index=False)
        
        try:
            # Initialize orchestrator
            orchestrator = UniversalDataScienceOrchestrator(
                enable_rag=False,
                enable_validation=True,
                enable_mle=True,
                enable_optimization=True,
                optimization_trials=5,
                verbose=True
            )
            
            # Run complete analysis
            results = orchestrator.analyze_dataset(
                file_path=temp_file,
                target_column='target',
                test_size=0.2,
                random_state=42,
                run_validation=True,
                run_optimization=True,
                store_results=False
            )
            
            # Verify results
            assert results['phases']['mle_agent']['task_type'] == 'regression'
            assert results['phases']['mle_agent']['best_score'] > 0  # R² score
            
            opt = results['phases']['optimization']
            assert opt['ensemble_test_score'] > 0
            
            print(f"\n✅ Regression Analysis Complete")
            print(f"   Task Type: {results['phases']['mle_agent']['task_type']}")
            print(f"   Best R² (Phase 3): {results['phases']['mle_agent']['best_score']:.4f}")
            print(f"   Best R² (Phase 4): {opt['best_cv_score']:.4f}")
            print(f"   Ensemble Test R²: {opt['ensemble_test_score']:.4f}")
            
            return True
        
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    except Exception as e:
        print(f"❌ Regression workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_integration():
    """Test 4: Data validation integration (Phase 2)."""
    print("\n" + "="*70)
    print("Test 4: Data Validation Integration")
    print("="*70)
    
    try:
        # Create dataset with potential leakage
        df = create_test_dataset('classification', n_samples=300, n_features=6)
        
        # Add a nearly perfect predictor (data leakage)
        df['leaked_feature'] = df['target'] + np.random.normal(0, 0.1, len(df))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            df.to_csv(f, index=False)
        
        try:
            orchestrator = UniversalDataScienceOrchestrator(
                enable_rag=False,
                enable_validation=True,
                enable_mle=True,
                enable_optimization=False,  # Skip optimization for speed
                verbose=True
            )
            
            results = orchestrator.analyze_dataset(
                file_path=temp_file,
                target_column='target',
                run_validation=True,
                run_optimization=False,
                store_results=False
            )
            
            # Verify validation ran
            assert 'validation' in results['phases']
            val = results['phases']['validation']
            
            print(f"\n✅ Validation Integration Working")
            print(f"   Risk Score: {val['risk_score']}/100")
            print(f"   Auto-fix Applied: {val['auto_fix_applied']}")
            
            return True
        
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    except Exception as e:
        print(f"❌ Validation integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_functionality():
    """Test 5: Prediction with trained models."""
    print("\n" + "="*70)
    print("Test 5: Prediction Functionality")
    print("="*70)
    
    try:
        # Create training dataset
        df_train = create_test_dataset('classification', n_samples=400, n_features=6)
        
        # Create test dataset (same features)
        df_test = create_test_dataset('classification', n_samples=100, n_features=6)
        X_test = df_test.drop('target', axis=1)
        
        # Save training data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            df_train.to_csv(f, index=False)
        
        try:
            orchestrator = UniversalDataScienceOrchestrator(
                enable_rag=False,
                enable_validation=False,
                enable_mle=True,
                enable_optimization=True,
                optimization_trials=5,
                verbose=False
            )
            
            # Train models
            results = orchestrator.analyze_dataset(
                file_path=temp_file,
                target_column='target',
                run_validation=False,
                run_optimization=True,
                store_results=False
            )
            
            # Make predictions
            predictions = orchestrator.predict(X_test)
            
            # Verify predictions
            assert len(predictions) == len(X_test)
            assert predictions.dtype in [np.int32, np.int64, np.float64]
            
            print(f"\n✅ Prediction Functionality Working")
            print(f"   Predictions shape: {predictions.shape}")
            print(f"   Unique values: {len(np.unique(predictions))}")
            print(f"   Sample predictions: {predictions[:5]}")
            
            return True
        
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generation():
    """Test 6: Report generation."""
    print("\n" + "="*70)
    print("Test 6: Report Generation")
    print("="*70)
    
    try:
        # Create test dataset
        df = create_test_dataset('classification', n_samples=300, n_features=6)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            df.to_csv(f, index=False)
        
        try:
            orchestrator = UniversalDataScienceOrchestrator(
                enable_rag=False,
                enable_validation=True,
                enable_mle=True,
                enable_optimization=True,
                optimization_trials=5,
                verbose=False
            )
            
            # Run analysis
            results = orchestrator.analyze_dataset(
                file_path=temp_file,
                target_column='target',
                store_results=False
            )
            
            # Generate report
            report = orchestrator.generate_report()
            
            # Verify report contents
            assert 'UNIVERSAL DATA SCIENCE ANALYSIS REPORT' in report
            assert 'PHASE 2: DATA VALIDATION' in report
            assert 'PHASE 3: UNIVERSAL ML ANALYSIS' in report
            assert 'PHASE 4: ADVANCED OPTIMIZATION' in report
            
            # Test saving to file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                report_file = f.name
            
            try:
                orchestrator.generate_report(output_file=report_file)
                assert os.path.exists(report_file)
                
                with open(report_file, 'r', encoding='utf-8') as f:
                    saved_report = f.read()
                
                assert saved_report == report
                
                print(f"\n✅ Report Generation Working")
                print(f"   Report length: {len(report)} characters")
                print(f"   Report saved to: {report_file}")
                
            finally:
                if os.path.exists(report_file):
                    os.remove(report_file)
            
            return True
        
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_analyze_function():
    """Test 7: Quick analyze convenience function."""
    print("\n" + "="*70)
    print("Test 7: Quick Analyze Function")
    print("="*70)
    
    try:
        # Create test dataset
        df = create_test_dataset('classification', n_samples=300, n_features=6)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            df.to_csv(f, index=False)
        
        try:
            # Use quick_analyze function
            print("\n🚀 Running quick_analyze()...")
            results = quick_analyze(
                file_path=temp_file,
                target_column='target',
                test_size=0.2,
                random_state=42,
                optimization_trials=5
            )
            
            # Verify results
            assert 'workflow_id' in results
            assert 'phases' in results
            
            print(f"\n✅ Quick Analyze Function Working")
            print(f"   Workflow ID: {results['workflow_id']}")
            print(f"   Phases completed: {list(results['phases'].keys())}")
            
            return True
        
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    except Exception as e:
        print(f"❌ Quick analyze failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_history():
    """Test 8: Workflow history tracking."""
    print("\n" + "="*70)
    print("Test 8: Workflow History Tracking")
    print("="*70)
    
    try:
        orchestrator = UniversalDataScienceOrchestrator(
            enable_rag=False,
            enable_validation=False,
            enable_mle=True,
            enable_optimization=False,
            verbose=False
        )
        
        # Run multiple workflows
        for i in range(3):
            df = create_test_dataset('classification', n_samples=200, n_features=5)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_file = f.name
                df.to_csv(f, index=False)
            
            try:
                orchestrator.analyze_dataset(
                    file_path=temp_file,
                    target_column='target',
                    run_validation=False,
                    run_optimization=False,
                    store_results=False
                )
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        # Check history
        summary = orchestrator.get_workflow_summary()
        
        assert summary['total_workflows'] == 3
        assert len(summary['workflows']) == 3
        
        print(f"\n✅ Workflow History Working")
        print(f"   Total workflows: {summary['total_workflows']}")
        print(f"   Workflow IDs: {[w['workflow_id'] for w in summary['workflows']]}")
        
        return True
    
    except Exception as e:
        print(f"❌ Workflow history failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all orchestrator tests."""
    print("\n" + "="*70)
    print("UNIVERSAL DATA SCIENCE ORCHESTRATOR - TEST SUITE")
    print("Phase 5: Full Integration Testing")
    print("="*70)
    
    tests = [
        ("Initialization", test_orchestrator_initialization),
        ("Classification Workflow", test_classification_workflow),
        ("Regression Workflow", test_regression_workflow),
        ("Validation Integration", test_validation_integration),
        ("Prediction Functionality", test_prediction_functionality),
        ("Report Generation", test_report_generation),
        ("Quick Analyze Function", test_quick_analyze_function),
        ("Workflow History", test_workflow_history)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 5 Integration Complete!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
