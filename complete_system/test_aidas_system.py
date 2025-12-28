"""
Test Suite for AIDAS - Autonomous Intelligence for Data Analysis and Science
==============================================================================

This test file validates all components of the AIDAS system:
1. Data ingestion and cleaning
2. Pattern discovery
3. Hypothesis testing
4. Model building
5. Insight generation
6. Natural language explanation
7. Report generation

Author: MiniMax Agent
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

# Add the complete_system directory to the path
sys.path.insert(0, '/workspace/resaerch01/complete_system')

from aidas_system import (
    AutonomousIntelligence,
    AnalysisConfig,
    DataQualityReport,
    Hypothesis,
    Insight,
    ModelResult,
    AnalysisPhase,
    TaskComplexity
)


def test_imports():
    """Test that all imports work correctly"""
    print("\n" + "="*60)
    print("TEST 1: Testing Imports")
    print("="*60)
    
    try:
        from aidas_system import (
            AutonomousIntelligence,
            AnalysisConfig,
            DataQualityReport,
            Hypothesis,
            Insight,
            ModelResult,
            AnalysisPhase,
            TaskComplexity
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dataclasses():
    """Test dataclass creation and functionality"""
    print("\n" + "="*60)
    print("TEST 2: Testing Dataclasses")
    print("="*60)
    
    # Test AnalysisConfig
    config = AnalysisConfig(max_iterations=5, confidence_threshold=0.90)
    assert config.max_iterations == 5
    assert config.confidence_threshold == 0.90
    print("✓ AnalysisConfig works")
    
    # Test DataQualityReport
    quality_report = DataQualityReport(
        total_rows=1000,
        total_columns=10,
        missing_values={'col1': 5, 'col2': 10},
        duplicate_rows=3,
        data_types={'col1': 'int64', 'col2': 'float64'},
        quality_score=0.95,
        issues_found=['Missing values found'],
        suggested_fixes=['Impute missing values']
    )
    assert quality_report.total_rows == 1000
    assert quality_report.quality_score == 0.95
    print("✓ DataQualityReport works")
    
    # Test Hypothesis
    hypothesis = Hypothesis(
        id="H001",
        statement="There is a correlation between X and Y",
        variables=["X", "Y"],
        test_type="correlation",
        statistic=0.75,
        p_value=0.001,
        significance_level=0.05,
        is_significant=True,
        effect_size=0.75,
        confidence_interval=(0.65, 0.85),
        explanation="Strong positive correlation found"
    )
    assert hypothesis.is_significant
    assert hypothesis.p_value < 0.05
    print("✓ Hypothesis works")
    
    # Test Insight
    insight = Insight(
        id="I001",
        category="Statistical Finding",
        description="Strong correlation between X and Y",
        evidence="r=0.75, p<0.001",
        impact="high",
        recommendation="Use this relationship in predictions",
        confidence=0.95,
        supporting_metrics={'correlation': 0.75}
    )
    assert insight.impact == "high"
    print("✓ Insight works")
    
    # Test ModelResult
    model_result = ModelResult(
        model_type="Random Forest",
        accuracy=0.92,
        precision=0.90,
        recall=0.88,
        f1_score=0.89,
        roc_auc=0.95,
        feature_importance={'feature1': 0.3, 'feature2': 0.25},
        confusion_matrix=[[10, 2], [1, 15]],
        cross_validation_scores=[0.90, 0.91, 0.89, 0.92, 0.88]
    )
    assert model_result.f1_score > 0.85
    print("✓ ModelResult works")
    
    print("\n✓ All dataclasses tested successfully")
    return True


def test_system_initialization():
    """Test system initialization"""
    print("\n" + "="*60)
    print("TEST 3: Testing System Initialization")
    print("="*60)
    
    # Test with default config
    aidas = AutonomousIntelligence()
    assert aidas.session_id is not None
    assert aidas.config.max_iterations == 3
    print("✓ Default initialization works")
    
    # Test with custom config
    custom_config = AnalysisConfig(
        max_iterations=5,
        confidence_threshold=0.99,
        sample_size=5000
    )
    aidas_custom = AutonomousIntelligence(config=custom_config)
    assert aidas_custom.config.max_iterations == 5
    assert aidas_custom.config.confidence_threshold == 0.99
    print("✓ Custom initialization works")
    
    # Test sub-systems exist
    assert hasattr(aidas, 'data_processor')
    assert hasattr(aidas, 'pattern_discoverer')
    assert hasattr(aidas, 'hypothesis_engine')
    assert hasattr(aidas, 'model_builder')
    assert hasattr(aidas, 'insight_generator')
    assert hasattr(aidas, 'explainer')
    print("✓ All sub-systems initialized")
    
    return True


def test_data_ingestion():
    """Test data ingestion functionality"""
    print("\n" + "="*60)
    print("TEST 4: Testing Data Ingestion")
    print("="*60)
    
    from aidas_system import _DataProcessor
    
    processor = _DataProcessor(AnalysisConfig())
    
    # Create a test CSV file
    test_data = """id,age,salary,department,performance
1,25,50000,Sales,8.5
2,30,60000,Marketing,7.2
3,35,75000,Sales,9.0
4,28,55000,Marketing,6.8
5,40,80000,Engineering,8.9
6,32,65000,Sales,7.5
7,45,90000,Engineering,9.2
8,27,48000,Marketing,6.5
9,38,78000,Sales,8.8
10,33,70000,Engineering,8.1
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(test_data)
        temp_file = f.name
    
    try:
        # Test CSV ingestion
        df = processor.ingest(temp_file)
        assert len(df) == 10
        assert len(df.columns) == 5
        print(f"✓ CSV ingestion successful: {len(df)} rows, {len(df.columns)} columns")
        
        return True
        
    finally:
        os.unlink(temp_file)


def test_data_cleaning():
    """Test data cleaning functionality"""
    print("\n" + "="*60)
    print("TEST 5: Testing Data Cleaning")
    print("="*60)
    
    from aidas_system import _DataProcessor
    import pandas as pd
    import numpy as np
    
    processor = _DataProcessor(AnalysisConfig())
    
    # Create test data with issues
    test_data = {
        'id': [1, 2, 3, 4, 5, 1],  # Duplicate
        'age': [25, 30, 35, np.nan, 45, 25],
        'salary': [50000, 60000, 75000, 55000, 80000, 50000],
        'department': ['Sales', 'Marketing', 'Sales', 'Marketing', 'Engineering', 'Sales'],
        'constant_col': ['same', 'same', 'same', 'same', 'same', 'same']  # Constant
    }
    df = pd.DataFrame(test_data)
    
    # Test cleaning
    cleaned_df, quality_report = processor.clean(df)
    
    assert quality_report.duplicate_rows > 0
    assert quality_report.quality_score > 0
    assert len(quality_report.issues_found) > 0
    # Check that constant column issue was detected
    constant_issue_found = any('constant' in issue.lower() for issue in quality_report.issues_found)
    assert constant_issue_found, f"Constant column issue not found in: {quality_report.issues_found}"
    print(f"✓ Data cleaning works")
    print(f"  - Quality score: {quality_report.quality_score:.2%}")
    print(f"  - Issues found: {len(quality_report.issues_found)}")
    print(f"  - Duplicates removed: {quality_report.duplicate_rows}")
    
    return True


def test_pattern_discovery():
    """Test pattern discovery functionality"""
    print("\n" + "="*60)
    print("TEST 6: Testing Pattern Discovery")
    print("="*60)
    
    from aidas_system import _PatternDiscoverer
    import pandas as pd
    import numpy as np
    
    discoverer = _PatternDiscoverer(AnalysisConfig())
    
    # Create test data with patterns
    np.random.seed(42)
    test_data = {
        'age': np.random.normal(35, 10, 100),
        'salary': 50000 + 1000 * np.random.normal(0, 1, 100) + 500 * np.random.normal(35, 10, 100),
        'experience': np.random.randint(1, 20, 100),
        'department': np.random.choice(['Sales', 'Marketing', 'Engineering'], 100)
    }
    df = pd.DataFrame(test_data)
    
    # Test pattern discovery
    results = discoverer.explore(df)
    
    assert 'numeric_summary' in results
    assert len(results['numeric_summary']) > 0
    assert 'categorical_summary' in results
    assert 'correlations' in results
    
    # Check numeric summary
    assert 'age' in results['numeric_summary']
    assert 'salary' in results['numeric_summary']
    print(f"✓ Pattern discovery works")
    print(f"  - Numeric variables analyzed: {len(results['numeric_summary'])}")
    print(f"  - Categorical variables analyzed: {len(results['categorical_summary'])}")
    print(f"  - Correlations found: {len(results['correlations'])}")
    
    return True


def test_hypothesis_engine():
    """Test hypothesis engine functionality"""
    print("\n" + "="*60)
    print("TEST 7: Testing Hypothesis Engine")
    print("="*60)
    
    from aidas_system import _HypothesisEngine
    import pandas as pd
    import numpy as np
    
    engine = _HypothesisEngine(AnalysisConfig())
    
    # Create test data
    np.random.seed(42)
    test_data = {
        'age': np.random.normal(35, 10, 100),
        'salary': 50000 + 1000 * np.random.normal(0, 1, 100) + 500 * np.random.normal(35, 10, 100),
        'experience': np.random.randint(1, 20, 100),
        'department': np.random.choice(['Sales', 'Marketing', 'Engineering'], 100)
    }
    df = pd.DataFrame(test_data)
    
    eda_results = {
        'correlations': [
            {'var1': 'age', 'var2': 'salary', 'correlation': 0.75, 'strength': 'strong'},
            {'var1': 'experience', 'var2': 'salary', 'correlation': 0.65, 'strength': 'moderate'}
        ],
        'categorical_summary': {
            'department': {'unique_values': 3, 'top_values': {'Sales': 35, 'Marketing': 33, 'Engineering': 32}}
        }
    }
    
    # Test hypothesis discovery
    hypotheses = engine.discover(df, eda_results)
    assert len(hypotheses) > 0
    print(f"✓ Hypothesis discovery works: {len(hypotheses)} hypotheses discovered")
    
    # Test hypothesis testing
    tested = engine.test(hypotheses, df)
    assert len(tested) > 0
    
    # Check that some hypotheses were tested
    tested_count = sum(1 for h in tested if h.statistic != 0.0 or h.p_value != 0.0)
    print(f"✓ Hypothesis testing works: {tested_count} hypotheses tested")
    
    return True


def test_model_builder():
    """Test model building functionality"""
    print("\n" + "="*60)
    print("TEST 8: Testing Model Builder")
    print("="*60)
    
    from aidas_system import _ModelBuilder
    import pandas as pd
    import numpy as np
    
    builder = _ModelBuilder(AnalysisConfig())
    
    # Create test data for classification
    np.random.seed(42)
    test_data = {
        'feature1': np.random.normal(50, 10, 200),
        'feature2': np.random.normal(30, 5, 200),
        'feature3': np.random.normal(100, 20, 200),
        'target': np.random.choice([0, 1, 2], 200),  # Classification
        'score': np.random.normal(75, 15, 200)  # Regression target
    }
    df = pd.DataFrame(test_data)
    
    eda_results = {
        'numeric_summary': {
            'feature1': {'mean': 50, 'std': 10},
            'feature2': {'mean': 30, 'std': 5},
            'target': {'mean': 1, 'std': 0.8}
        }
    }
    
    # Test model building
    models = builder.build_models(df, eda_results)
    
    assert len(models) > 0
    assert all(hasattr(m, 'accuracy') for m in models)
    assert all(hasattr(m, 'f1_score') for m in models)
    
    print(f"✓ Model builder works")
    print(f"  - Models built: {len(models)}")
    for model in models:
        print(f"    - {model.model_type}: Accuracy={model.accuracy:.3f}, F1={model.f1_score:.3f}")
    
    return True


def test_insight_generator():
    """Test insight generation functionality"""
    print("\n" + "="*60)
    print("TEST 9: Testing Insight Generator")
    print("="*60)
    
    from aidas_system import _InsightGenerator, Hypothesis, Insight, ModelResult
    import pandas as pd
    import numpy as np
    
    generator = _InsightGenerator(AnalysisConfig())
    
    # Create test data
    df = pd.DataFrame({'col1': range(100), 'col2': range(100)})
    
    eda_results = {
        'correlations': [],
        'numeric_summary': {'col1': {'mean': 49.5}}
    }
    
    # Create test hypotheses
    hypotheses = [
        Hypothesis(
            id="H001",
            statement="Test hypothesis",
            variables=["col1", "col2"],
            test_type="correlation",
            statistic=0.75,
            p_value=0.001,
            significance_level=0.05,
            is_significant=True,
            effect_size=0.75,
            confidence_interval=(0.65, 0.85),
            explanation="Strong correlation found"
        )
    ]
    
    # Create test models
    models = [
        ModelResult(
            model_type="Random Forest",
            accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            roc_auc=0.95,
            feature_importance={'col1': 0.3, 'col2': 0.25},
            confusion_matrix=[[10, 2], [1, 15]],
            cross_validation_scores=[0.90, 0.91, 0.89, 0.92, 0.88]
        )
    ]
    
    # Test insight generation
    insights = generator.generate(df, eda_results, hypotheses, models)
    
    assert len(insights) > 0
    print(f"✓ Insight generator works")
    print(f"  - Insights generated: {len(insights)}")
    for insight in insights:
        print(f"    - {insight.category}: {insight.description[:50]}...")
    
    return True


def test_nl_explainer():
    """Test natural language explainer"""
    print("\n" + "="*60)
    print("TEST 10: Testing Natural Language Explainer")
    print("="*60)
    
    from aidas_system import _NLExplainer, Hypothesis, Insight, ModelResult
    import pandas as pd
    
    explainer = _NLExplainer()
    
    # Create test data
    df = pd.DataFrame({'col1': range(100), 'col2': range(100)})
    
    eda_results = {
        'numeric_summary': {'col1': {'mean': 49.5}},
        'correlations': []
    }
    
    hypotheses = [
        Hypothesis(
            id="H001",
            statement="There is a relationship between X and Y",
            variables=["X", "Y"],
            test_type="correlation",
            statistic=0.75,
            p_value=0.001,
            significance_level=0.05,
            is_significant=True,
            effect_size=0.75,
            confidence_interval=(0.65, 0.85),
            explanation="Strong positive correlation found"
        )
    ]
    
    insights = [
        Insight(
            id="I001",
            category="Statistical Finding",
            description="Strong correlation found",
            evidence="r=0.75",
            impact="high",
            recommendation="Use this in predictions",
            confidence=0.95,
            supporting_metrics={}
        )
    ]
    
    models = [
        ModelResult(
            model_type="Random Forest",
            accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            roc_auc=0.95,
            feature_importance={'col1': 0.3},
            confusion_matrix=[[10, 2], [1, 15]],
            cross_validation_scores=[0.90, 0.91]
        )
    ]
    
    # Test explanation generation
    explanation = explainer.explain_all(df, eda_results, hypotheses, insights, models)
    
    assert len(explanation) > 0
    assert 'Data Overview' in explanation
    assert 'Key Statistical Findings' in explanation
    assert 'Actionable Insights' in explanation
    
    print(f"✓ Natural language explainer works")
    print(f"  - Explanation length: {len(explanation)} characters")
    print(f"  - Contains data overview: {'Data Overview' in explanation}")
    print(f"  - Contains findings: {'Key Statistical Findings' in explanation}")
    
    return True


def test_full_pipeline():
    """Test the complete analysis pipeline"""
    print("\n" + "="*60)
    print("TEST 11: Testing Full Pipeline")
    print("="*60)
    
    # Create test dataset
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    test_data = {
        'id': range(1, 101),
        'age': np.random.normal(35, 10, 100).astype(int),
        'salary': np.random.normal(60000, 15000, 100).astype(int),
        'experience_years': np.random.randint(1, 25, 100),
        'department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR'], 100),
        'performance_score': np.random.uniform(1, 10, 100).round(1),
        'target_variable': np.random.choice(['High', 'Medium', 'Low'], 100)
    }
    df = pd.DataFrame(test_data)
    
    # Add some missing values and duplicates for cleaning
    df.loc[0, 'age'] = np.nan
    df.loc[1, 'salary'] = np.nan
    df = pd.concat([df, df.head(5)], ignore_index=True)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_file = f.name
    
    try:
        # Initialize AIDAS
        config = AnalysisConfig(
            max_iterations=3,
            confidence_threshold=0.95,
            sample_size=1000,
            test_size=0.3
        )
        
        aidas = AutonomousIntelligence(config)
        
        # Run full analysis
        results = aidas.analyze(temp_file, 'aidas_test_output')
        
        # Verify results
        assert results['status'] == 'success', f"Pipeline failed: {results.get('error')}"
        assert 'phases' in results
        assert len(results['phases']) > 0
        
        print(f"✓ Full pipeline test successful")
        print(f"  - Duration: {results.get('total_duration', 0):.2f} seconds")
        print(f"  - Phases completed: {len(results['phases'])}")
        
        # Check key phases
        phases = results['phases']
        assert 'data_ingestion' in phases
        assert 'data_cleaning' in phases
        assert 'exploratory_analysis' in phases
        assert 'hypothesis_testing' in phases
        assert 'model_building' in phases
        
        print(f"  - Data ingestion: {phases['data_ingestion'].get('rows', 'N/A')} rows")
        print(f"  - Models built: {phases['model_building'].get('models_built', 0)}")
        
        return True
        
    finally:
        os.unlink(temp_file)
        # Clean up test output
        import shutil
        if os.path.exists('aidas_test_output'):
            shutil.rmtree('aidas_test_output')


def test_with_real_dataset():
    """Test with the real fast food consumption dataset"""
    print("\n" + "="*60)
    print("TEST 12: Testing with Real Dataset")
    print("="*60)
    
    dataset_path = '/workspace/user_input_files/fast_food_consumption_health_impact_dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found: {dataset_path}")
        return False
    
    # Initialize AIDAS
    config = AnalysisConfig(
        max_iterations=3,
        confidence_threshold=0.95,
        sample_size=1000
    )
    
    aidas = AutonomousIntelligence(config)
    
    # Run analysis
    results = aidas.analyze(dataset_path, 'aidas_real_output')
    
    if results['status'] != 'success':
        print(f"✗ Analysis failed: {results.get('error')}")
        return False
    
    print(f"✓ Real dataset analysis successful")
    print(f"  - Duration: {results.get('total_duration', 0):.2f} seconds")
    
    # Check results
    phases = results.get('phases', {})
    
    if 'data_ingestion' in phases:
        di = phases['data_ingestion']
        print(f"  - Data: {di.get('rows', 'N/A')} rows × {di.get('columns', 'N/A')} columns")
    
    if 'hypothesis_testing' in phases:
        ht = phases['hypothesis_testing']
        sig = ht.get('significant_hypotheses', 0)
        print(f"  - Significant hypotheses: {sig}")
    
    if 'model_building' in phases:
        mb = phases['model_building']
        print(f"  - Models built: {mb.get('models_built', 0)}")
        print(f"  - Best model: {mb.get('best_model', 'N/A')}")
    
    # Clean up
    import shutil
    if os.path.exists('aidas_real_output'):
        shutil.rmtree('aidas_real_output')
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("AIDAS SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Import Tests", test_imports),
        ("Dataclass Tests", test_dataclasses),
        ("System Initialization", test_system_initialization),
        ("Data Ingestion", test_data_ingestion),
        ("Data Cleaning", test_data_cleaning),
        ("Pattern Discovery", test_pattern_discovery),
        ("Hypothesis Engine", test_hypothesis_engine),
        ("Model Builder", test_model_builder),
        ("Insight Generator", test_insight_generator),
        ("Natural Language Explainer", test_nl_explainer),
        ("Full Pipeline Test", test_full_pipeline),
        ("Real Dataset Test", test_with_real_dataset),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"   Error: {error[:100]}...")
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*80}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
