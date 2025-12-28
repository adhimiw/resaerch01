"""
Example: Autonomous MCP Data Science System
Demonstrates end-to-end autonomous operation like GitHub Copilot
"""

import asyncio
import json
import tempfile
from pathlib import Path
import pandas as pd


async def example_1_basic_autonomous_analysis():
    """Example 1: Basic autonomous dataset analysis"""
    print("\n" + "="*70)
    print("📊 EXAMPLE 1: Basic Autonomous Analysis")
    print("="*70)
    
    # Import autonomous analysis
    from complete_system.core.autonomous_mcp_agent import autonomous_analysis
    
    # Create sample data
    data = {
        'id': range(1, 101),
        'product': [f'Product_{i}' for i in range(1, 101)],
        'sales': [i * 10 for i in range(1, 101)],
        'region': ['North' if i % 2 == 0 else 'South' for i in range(1, 101)],
        'profit': [i * 2.5 for i in range(1, 101)]
    }
    df = pd.DataFrame(data)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        # Run autonomous analysis - NO MANUAL INTERVENTION NEEDED
        results = await autonomous_analysis(
            dataset_path=temp_path,
            analysis_goal="analyze sales performance by region"
        )
        
        print(f"✅ Analysis Complete!")
        print(f"📊 Success Rate: {results['success_rate']:.1%}")
        print(f"⏱️ Duration: {results['total_duration']:.2f}s")
        
        if 'dataset_info' in results:
            info = results['dataset_info']
            print(f"📏 Dataset Shape: {info.get('shape', 'N/A')}")
        
        return results
        
    finally:
        Path(temp_path).unlink()


async def example_2_custom_workflow():
    """Example 2: Custom autonomous workflow with dependencies"""
    print("\n" + "="*70)
    print("🔗 EXAMPLE 2: Custom Workflow with Dependencies")
    print("="*70)
    
    from complete_system.core.mcp_integration_autonomous import get_autonomous_mcp
    
    # Create sample data
    data = {
        'date': pd.date_range('2024-01-01', periods=100),
        'temperature': [20 + i + (i % 7) for i in range(100)],
        'humidity': [50 + i % 10 for i in range(100)],
        'pressure': [1013 + i % 5 for i in range(100)]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        integration = await get_autonomous_mcp()
        
        # Define custom workflow
        tasks = [
            {
                "id": "load_weather_data",
                "description": "Load weather dataset",
                "tool": "pandas.read_csv",
                "parameters": {
                    "filepath": temp_path,
                    "name": "weather"
                }
            },
            {
                "id": "explore_weather",
                "description": "Get weather data info",
                "tool": "pandas.info",
                "parameters": {"name": "weather"},
                "dependencies": ["load_weather_data"]
            },
            {
                "id": "weather_stats",
                "description": "Get statistical summary",
                "tool": "pandas.describe",
                "parameters": {"name": "weather"},
                "dependencies": ["explore_weather"]
            },
            {
                "id": "correlation",
                "description": "Find correlations between weather variables",
                "tool": "pandas.correlation",
                "parameters": {"name": "weather"},
                "dependencies": ["weather_stats"]
            }
        ]
        
        print("🚀 Starting autonomous workflow execution...")
        results = await integration.execute_autonomous_workflow(tasks)
        
        print(f"\n📊 Workflow Results:")
        print(f"   Total Tasks: {results['total_tasks']}")
        print(f"   ✅ Completed: {results['completed']}")
        print(f"   ❌ Failed: {results['failed']}")
        
        return results
        
    finally:
        Path(temp_path).unlink()


async def example_3_self_healing_execution():
    """Example 3: Self-healing function execution"""
    print("\n" + "="*70)
    print("🩹 EXAMPLE 3: Self-Healing Function Execution")
    print("="*70)
    
    from complete_system.core.self_healing_executor import (
        SelfHealingExecutor,
        RecoveryConfig,
        RecoveryStrategy,
        with_self_healing
    )
    
    # Simulate an unreliable function
    call_count = [0]
    
    def unreliable_function(data):
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError(f"Network error on attempt {call_count[0]}")
        return f"Processed {len(data)} items successfully"
    
    # Use self-healing executor
    executor = SelfHealingExecutor(RecoveryConfig(
        max_attempts=5,
        initial_delay=0.5,
        strategy=RecoveryStrategy.RETRY
    ))
    
    test_data = [1, 2, 3, 4, 5]
    result = executor.execute(unreliable_function, test_data)
    
    print(f"✅ Final Result: {result.result}")
    print(f"📊 Attempts Required: {result.attempt_number}")
    print(f"⏱️ Duration: {result.duration_ms:.1f}ms")
    
    # Use decorator
    print("\n🔧 Using decorator...")
    
    @with_self_healing(max_attempts=3, strategy=RecoveryStrategy.RETRY)
    def another_unreliable():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise TimeoutError("Request timeout")
        return "Success!"
    
    try:
        output = another_unreliable()
        print(f"✅ Decorator Result: {output}")
    except RuntimeError as e:
        print(f"❌ Failed after retries: {e}")
    
    return result


async def example_4_environment_aware():
    """Example 4: Environment-aware configuration"""
    print("\n" + "="*70)
    print("⚙️ EXAMPLE 4: Environment-Aware Configuration")
    print("="*70)
    
    from complete_system.core.environment_config import get_config, get_env_manager
    
    # Get configuration (loads from .env automatically)
    config = get_config()
    
    print(f"📁 Workspace Directory: {config.workspace_dir}")
    print(f"📁 Data Directory: {config.data_dir}")
    print(f"📁 Results Directory: {config.results_dir}")
    
    # Check API keys
    has_mistral = bool(config.mistral_api_key)
    has_langfuse = bool(config.langfuse_public_key)
    
    print(f"\n🔑 API Configuration:")
    print(f"   Mistral API: {'✅ Configured' if has_mistral else '⚠️ Not set'}")
    print(f"   Langfuse: {'✅ Configured' if has_langfuse else '⚠️ Not set'}")
    
    print(f"\n⚙️ Execution Settings:")
    print(f"   Max Retries: {config.max_retries}")
    print(f"   Retry Delay: {config.retry_delay}s")
    
    # Check if environment is ready
    env_manager = get_env_manager()
    if env_manager.is_configured():
        print("\n✅ Environment is ready for autonomous operation!")
    else:
        print("\n⚠️ Environment not fully configured. Create .env file.")
    
    return config


async def example_5_end_to_end_pipeline():
    """Example 5: Complete end-to-end autonomous pipeline"""
    print("\n" + "="*70)
    print("🚀 EXAMPLE 5: End-to-End Autonomous Pipeline")
    print("="*70)
    
    from complete_system.core.mcp_integration_autonomous import get_autonomous_mcp
    
    # Create comprehensive sample dataset
    import numpy as np
    
    n_samples = 500
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(30000, 150000, n_samples),
        'spending_score': np.random.randint(1, 100, n_samples),
        'purchase_frequency': np.random.randint(1, 30, n_samples),
        'satisfaction': np.random.uniform(1, 5, n_samples),
        'churned': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        integration = await get_autonomous_mcp()
        
        # Define comprehensive analysis pipeline
        pipeline = [
            {
                "id": "load_customer_data",
                "description": "Load customer churn dataset",
                "tool": "pandas.read_csv",
                "parameters": {"filepath": temp_path, "name": "customers"}
            },
            {
                "id": "data_overview",
                "description": "Get dataset overview",
                "tool": "pandas.info",
                "parameters": {"name": "customers"}
            },
            {
                "id": "statistics",
                "description": "Get statistical summary",
                "tool": "pandas.describe",
                "parameters": {"name": "customers"}
            },
            {
                "id": "correlations",
                "description": "Find correlations with churn",
                "tool": "pandas.correlation",
                "parameters": {"name": "customers"}
            },
            {
                "id": "outliers_age",
                "description": "Detect age outliers",
                "tool": "pandas.find_outliers",
                "parameters": {"name": "customers", "column": "age", "method": "iqr"},
                "dependencies": ["data_overview"]
            }
        ]
        
        print("🔄 Running autonomous pipeline...")
        results = await integration.execute_autonomous_workflow(pipeline)
        
        print(f"\n📊 Pipeline Summary:")
        print(f"   Steps: {results['total_tasks']}")
        print(f"   ✅ Success: {results['completed']}")
        print(f"   ❌ Failed: {results['failed']}")
        
        # Show key results
        if 'correlations' in results.get('results', {}):
            corr_result = results['results']['correlations']
            if isinstance(corr_result, dict):
                print(f"\n📈 Key Correlations Found:")
                print(f"   {corr_result}")
        
        return results
        
    finally:
        Path(temp_path).unlink()


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("🤖 AUTONOMOUS MCP SYSTEM - EXAMPLE SHOWCASE")
    print("="*70)
    
    examples = [
        ("Basic Autonomous Analysis", example_1_basic_autonomous_analysis),
        ("Custom Workflow", example_2_custom_workflow),
        ("Self-Healing Execution", example_3_self_healing_execution),
        ("Environment Awareness", example_4_environment_aware),
        ("End-to-End Pipeline", example_5_end_to_end_pipeline),
    ]
    
    results = {}
    
    for name, func in examples:
        try:
            results[name] = await func()
            print(f"\n✅ {name} - Complete")
        except Exception as e:
            print(f"\n❌ {name} - Failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = str(e)
    
    # Summary
    print("\n" + "="*70)
    print("📋 FINAL SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if isinstance(v, dict) and v.get('completed', 0) > 0)
    total = len(results)
    
    print(f"\n✅ Examples Passed: {passed}/{total}")
    
    print("\n🎯 Key Features Demonstrated:")
    print("   ✅ Autonomous dataset analysis")
    print("   ✅ Custom workflow execution")
    print("   ✅ Self-healing with automatic retry")
    print("   ✅ Environment-aware configuration")
    print("   ✅ End-to-end autonomous pipeline")
    
    print("\n🚀 Your MCP tools are now GitHub Copilot-style autonomous!")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
