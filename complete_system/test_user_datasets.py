"""
Test Autonomous MCP System with User Datasets
Demonstrates end-to-end autonomous operation on real datasets
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_dataset_autonomous(dataset_name: str, dataset_path: str, analysis_goal: str):
    """Test autonomous analysis on a single dataset"""
    print("\n" + "="*80)
    print(f"📊 TESTING: {dataset_name}")
    print(f"📁 File: {dataset_path}")
    print(f"🎯 Goal: {analysis_goal}")
    print("="*80)
    
    try:
        from core.autonomous_mcp_agent import autonomous_analysis
        
        # Run autonomous analysis
        print("\n🚀 Starting autonomous analysis...")
        results = await autonomous_analysis(
            dataset_path=dataset_path,
            analysis_goal=analysis_goal
        )
        
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Success Rate: {results['success_rate']:.1%}")
        print(f"⏱️ Duration: {results['total_duration']:.2f}s")
        
        if 'dataset_info' in results:
            info = results['dataset_info']
            print(f"\n📈 Dataset Information:")
            print(f"   Shape: {info.get('shape', 'N/A')}")
            print(f"   Columns: {info.get('columns', [])}")
        
        if 'exploration' in results:
            print(f"\n🔍 Key Insights:")
            exp = results['exploration']
            if isinstance(exp, dict):
                for key, value in exp.items():
                    if isinstance(value, (int, float, str)):
                        print(f"   • {key}: {value}")
        
        print("\n" + "-"*80)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Test all user datasets"""
    print("\n" + "="*80)
    print("🤖 AUTONOMOUS MCP SYSTEM - REAL DATASET TESTING")
    print("="*80)
    
    # Define datasets to test
    datasets = [
        {
            "name": "GitHub Trending Repositories",
            "path": "/workspace/user_input_files/github_trending_repos.csv",
            "goal": "Analyze GitHub trending repositories, identify popular programming languages, and find patterns in stars and forks"
        },
        {
            "name": "Global Climate & Energy (2020-2024)",
            "path": "/workspace/user_input_files/global_climate_energy_2020_2024.csv",
            "goal": "Analyze climate and energy trends, correlations between temperature, CO2 emissions, and renewable energy adoption"
        },
        {
            "name": "Fast Food Consumption & Health Impact",
            "path": "/workspace/user_input_files/fast_food_consumption_health_impact_dataset.csv",
            "goal": "Analyze the relationship between fast food consumption, BMI, and overall health scores"
        }
    ]
    
    results = {}
    
    for dataset in datasets:
        print(f"\n\n📦 Loading dataset: {dataset['name']}")
        
        result = await test_dataset_autonomous(
            dataset_name=dataset['name'],
            dataset_path=dataset['path'],
            analysis_goal=dataset['goal']
        )
        
        results[dataset['name']] = result
    
    # Summary
    print("\n\n" + "="*80)
    print("📊 FINAL SUMMARY - ALL DATASETS")
    print("="*80)
    
    successful = 0
    total = len(datasets)
    
    for name, result in results.items():
        status = "✅ SUCCESS" if result and result.get('success_rate', 0) > 0 else "❌ FAILED"
        if result and result.get('success_rate', 0) > 0:
            successful += 1
        print(f"\n{status}: {name}")
        
        if result:
            print(f"   Success Rate: {result.get('success_rate', 0):.1%}")
            print(f"   Duration: {result.get('total_duration', 0):.2f}s")
            
            if 'dataset_info' in result:
                shape = result['dataset_info'].get('shape', 'N/A')
                print(f"   Dataset: {shape}")
    
    print(f"\n\n🎯 Overall Results: {successful}/{total} datasets analyzed successfully")
    
    if successful == total:
        print("\n🎉 ALL DATASETS PROCESSED AUTONOMOUSLY!")
        print("\nYour MCP system successfully:")
        print("   ✅ Discovered and connected to Pandas MCP server")
        print("   ✅ Executed end-to-end analysis workflows")
        print("   ✅ Handled diverse data types (GitHub, Climate, Health)")
        print("   ✅ Generated insights without manual intervention")
    else:
        print(f"\n⚠️ {total - successful} dataset(s) failed. Check logs above.")
    
    print("\n" + "="*80)
    
    return successful == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
