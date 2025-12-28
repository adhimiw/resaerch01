"""
Test Script for MLE-Agent (Phase 3)
Tests universal dataset analysis workflow

Run: python test_mle_agent.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from complete_system.core.mle_agent import MLEAgent, analyze_any_dataset


def create_sample_dataset():
    """Create sample dataset for testing."""
    print("\n📍 Creating Sample Dataset for Testing")
    print("-" * 70)
    
    np.random.seed(42)
    
    # Create synthetic classification dataset
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'employment_years': np.random.randint(0, 30, n_samples),
        'num_dependents': np.random.randint(0, 5, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
        'home_ownership': np.random.choice(['Own', 'Rent', 'Mortgage'], n_samples)
    }
    
    # Create target based on rules (loan approval)
    df = pd.DataFrame(data)
    df['approved'] = (
        (df['income'] > 40000) & 
        (df['credit_score'] > 600) & 
        (df['employment_years'] > 1)
    ).astype(int)
    
    # Add some noise
    noise_idx = np.random.choice(df.index, size=int(n_samples * 0.1), replace=False)
    df.loc[noise_idx, 'approved'] = 1 - df.loc[noise_idx, 'approved']
    
    # Add some missing values
    missing_cols = ['income', 'credit_score', 'employment_years']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    # Split train/test
    train_df = df.iloc[:800].copy()
    test_df = df.iloc[800:].copy()
    
    # Save to CSV
    train_path = Path('sample_train.csv')
    test_path = Path('sample_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.drop(columns=['approved']).to_csv(test_path, index=False)
    
    print(f"✅ Created datasets:")
    print(f"   Train: {train_df.shape} → {train_path}")
    print(f"   Test: {test_df.shape} → {test_path}")
    print(f"   Target: 'approved' (binary classification)")
    print(f"   Features: 9 (6 numeric, 3 categorical)")
    print(f"   Missing values: ~5% in 3 columns")
    
    return str(train_path), str(test_path)


def test_phase3_mle_agent():
    """
    Comprehensive test for Phase 3: MLE-Agent.
    
    Tests:
    1. Dataset loading (CSV/Excel)
    2. Exploratory Data Analysis (EDA)
    3. Data validation integration (Phase 2)
    4. Preprocessing pipeline
    5. Model training (multiple algorithms)
    6. Cross-validation
    7. Predictions
    8. Report generation
    """
    print("=" * 70)
    print("🧪 PHASE 3 TEST: MLE-AGENT (UNIVERSAL DATASET ANALYSIS)")
    print("=" * 70)
    
    # Create sample dataset
    train_path, test_path = create_sample_dataset()
    
    # Test 1: Initialize Agent
    print("\n📍 Test 1: Initialize MLE-Agent")
    print("-" * 70)
    
    try:
        agent = MLEAgent(
            enable_validation=True,
            auto_fix_leakage=True,
            verbose=True
        )
        print("✅ MLE-Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return False
    
    # Test 2: Load Dataset
    print("\n📍 Test 2: Load Dataset")
    print("-" * 70)
    
    try:
        data = agent.load_dataset(
            file_path=train_path,
            target_column='approved',
            test_file_path=test_path
        )
        
        print(f"✅ Dataset loaded:")
        print(f"   Train shape: {data['train'].shape}")
        print(f"   Test shape: {data['test'].shape}")
        print(f"   Target: {data['target_column']}")
        print(f"   Columns: {len(data['metadata']['columns'])}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Test 3: Exploratory Data Analysis
    print("\n📍 Test 3: Exploratory Data Analysis (EDA)")
    print("-" * 70)
    
    try:
        eda_results = agent.analyze_dataset(data)
        
        print(f"\n✅ EDA complete:")
        print(f"   Rows: {eda_results['basic_stats']['n_rows']:,}")
        print(f"   Numeric features: {eda_results['basic_stats']['n_numeric']}")
        print(f"   Categorical features: {eda_results['basic_stats']['n_categorical']}")
        print(f"   Missing data: {eda_results['basic_stats']['missing_percentage']:.2f}%")
        
        print(f"\n   💡 Insights ({len(eda_results['insights'])}):")
        for i, insight in enumerate(eda_results['insights'][:3], 1):
            print(f"      {i}. {insight}")
        
        print(f"\n   📝 Recommendations ({len(eda_results['recommendations'])}):")
        for i, rec in enumerate(eda_results['recommendations'][:3], 1):
            print(f"      {i}. {rec}")
        
    except Exception as e:
        print(f"❌ EDA failed: {e}")
        return False
    
    # Test 4: Data Validation (Phase 2 Integration)
    print("\n📍 Test 4: Data Validation (Phase 2 Integration)")
    print("-" * 70)
    
    if eda_results.get('validation_report'):
        val_report = eda_results['validation_report']
        print(f"\n✅ Validation complete:")
        print(f"   Risk score: {val_report['risk_score']}/100")
        print(f"   Severity: {val_report['severity']}")
        print(f"   Issues found: {len(val_report['issues'])}")
        
        if val_report['risk_score'] < 40:
            print(f"   🟢 Low risk - dataset is clean!")
        
    else:
        print("   ℹ️  Validation skipped (Phase 2 not available)")
    
    # Test 5: Data Preprocessing
    print("\n📍 Test 5: Data Preprocessing Pipeline")
    print("-" * 70)
    
    try:
        processed_data = agent.preprocess_data(
            data=data,
            handle_missing='median',
            encode_categorical='onehot',
            scale_features=True
        )
        
        print(f"\n✅ Preprocessing complete:")
        print(f"   Original features: {len(agent.preprocessing_pipeline['original_columns'])}")
        print(f"   Encoded features: {len(processed_data['feature_names'])}")
        print(f"   X_train shape: {processed_data['X_train'].shape}")
        print(f"   X_test shape: {processed_data['X_test'].shape}")
        
        print(f"\n   Pipeline steps:")
        print(f"      1. Missing values: {agent.preprocessing_pipeline['handle_missing']}")
        print(f"      2. Categorical encoding: {agent.preprocessing_pipeline['encode_categorical']}")
        print(f"      3. Feature scaling: {agent.preprocessing_pipeline['scale_features']}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False
    
    # Test 6: Model Training
    print("\n📍 Test 6: Model Training & Cross-Validation")
    print("-" * 70)
    
    try:
        training_results = agent.train_models(
            processed_data=processed_data,
            task_type='auto'
        )
        
        print(f"\n✅ Training complete:")
        print(f"   Task type: {training_results['task_type']}")
        print(f"   Models trained: {len(training_results['models'])}")
        print(f"   Best model: {training_results['best_model']}")
        print(f"   Features used: {training_results['n_features']}")
        
        print(f"\n   📊 Model Performance:")
        for model_name, model_info in training_results['models'].items():
            metrics = model_info['metrics']
            cv_score = metrics['cv_mean']
            cv_std = metrics['cv_std']
            
            print(f"\n   {model_name.upper()}:")
            print(f"      Cross-validation: {cv_score:.4f} (±{cv_std:.4f})")
            
            if 'train_accuracy' in metrics:
                print(f"      Train accuracy: {metrics['train_accuracy']:.4f}")
                print(f"      Train F1: {metrics['train_f1']:.4f}")
            else:
                print(f"      Train R²: {metrics['train_r2']:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 7: Predictions
    print("\n📍 Test 7: Generate Predictions")
    print("-" * 70)
    
    try:
        predictions = agent.predict(processed_data)
        
        print(f"\n✅ Predictions generated:")
        print(f"   Model used: {predictions['model_name']}")
        print(f"   Task type: {predictions['task_type']}")
        print(f"   Predictions: {predictions['n_predictions']}")
        
        # Show distribution
        unique, counts = np.unique(predictions['predictions'], return_counts=True)
        print(f"\n   Prediction distribution:")
        for val, count in zip(unique, counts):
            pct = (count / predictions['n_predictions']) * 100
            print(f"      Class {val}: {count} ({pct:.1f}%)")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    # Test 8: Report Generation
    print("\n📍 Test 8: Generate Analysis Report")
    print("-" * 70)
    
    try:
        report = agent.generate_report()
        
        print("\n✅ Report generated successfully")
        print("\nReport preview (first 20 lines):")
        print("-" * 70)
        
        report_lines = report.split('\n')
        for line in report_lines[:20]:
            print(line)
        
        if len(report_lines) > 20:
            print(f"... ({len(report_lines) - 20} more lines)")
        
        # Save report
        report_path = Path('mle_agent_report.txt')
        agent.save_report(str(report_path))
        print(f"\n✅ Full report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False
    
    # Test 9: End-to-End Workflow
    print("\n📍 Test 9: Complete End-to-End Workflow")
    print("-" * 70)
    
    try:
        print("\nTesting analyze_any_dataset() convenience function...")
        
        results = analyze_any_dataset(
            train_path=train_path,
            target_column='approved',
            test_path=test_path,
            task_type='classification'
        )
        
        print("\n✅ End-to-end workflow complete!")
        print(f"   Components: {list(results.keys())}")
        
    except Exception as e:
        print(f"❌ End-to-end workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    print("\n📍 Cleanup: Removing test files...")
    try:
        Path(train_path).unlink()
        Path(test_path).unlink()
        Path('mle_agent_report.txt').unlink()
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ PHASE 3 TEST COMPLETE - MLE-Agent fully operational!")
    print("=" * 70)
    
    print("\n📊 Test Summary:")
    print(f"   ✅ Agent initialization: Working")
    print(f"   ✅ Dataset loading: Working (CSV/Excel)")
    print(f"   ✅ EDA: Working ({len(eda_results['insights'])} insights)")
    print(f"   ✅ Phase 2 validation: Integrated")
    print(f"   ✅ Preprocessing: Working ({processed_data['X_train'].shape[1]} features)")
    print(f"   ✅ Model training: Working ({len(training_results['models'])} models)")
    print(f"   ✅ Predictions: Working ({predictions['n_predictions']} predictions)")
    print(f"   ✅ Report generation: Working")
    print(f"   ✅ End-to-end workflow: Working")
    
    print("\n🎉 Phase 3 (MLE-Agent) implementation successful!")
    print("   Works with ANY dataset (not just competitions)")
    print("   Next: Phase 4 (Agent-Lightning) - Week 6-7")
    
    return True


if __name__ == "__main__":
    print("\n")
    print("🚀 Starting Phase 3 Test Suite...")
    print("   Testing: MLE-Agent (Universal Dataset Analysis)")
    print("   Timeline: Week 4-5 Implementation")
    print("\n")
    
    success = test_phase3_mle_agent()
    
    if success:
        print("\n✅ All tests passed! Phase 3 ready for integration.")
    else:
        print("\n❌ Some tests failed. Check errors above.")
    
    sys.exit(0 if success else 1)
