"""
Test 1: DSPy System with Spotify Data (Proven Baseline)
Tests the universal agent with the working Spotify dataset
"""

import os
import sys
import json
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from dspy_universal_agent import UniversalAgenticDataScience


def main():
    print("="*80)
    print("TEST 1: DSPY UNIVERSAL AGENT WITH SPOTIFY DATA")
    print("="*80)
    
    # API Keys
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "6IOUctuofzEsOgw0SHi17BfmjoieITTQ")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-53f3176f-72f7-4183-9cdc-e589f62ab968")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-65bf0f45-143e-4a6c-883f-769cd8da4444")
    
    # Initialize agent
    print("\n🚀 Initializing Universal Agent...")
    agent = UniversalAgenticDataScience(
        mistral_api_key=MISTRAL_API_KEY,
        langfuse_public_key=LANGFUSE_PUBLIC_KEY,
        langfuse_secret_key=LANGFUSE_SECRET_KEY
    )
    
    # Test with both Spotify datasets
    datasets = [
        ("../../spotify_data-clean.csv", "Spotify Clean Dataset"),
        ("../../track_data_final.csv", "Spotify Track Data")
    ]
    
    results = {}
    
    for dataset_path, dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {dataset_name}")
        print(f"{'='*80}")
        
        try:
            # Check if file exists
            full_path = Path(__file__).parent / dataset_path
            if not full_path.exists():
                print(f"⚠️  File not found: {full_path}")
                print(f"   Looking for alternate locations...")
                
                # Try root folder
                alt_path = Path(__file__).parent.parent.parent / Path(dataset_path).name
                if alt_path.exists():
                    full_path = alt_path
                    print(f"✓ Found at: {full_path}")
                else:
                    print(f"❌ Could not find dataset")
                    continue
            
            # Run analysis
            result = agent.analyze(str(full_path))
            results[dataset_name] = result
            
            # Show key insights
            print(f"\n💡 KEY INSIGHTS:")
            print("-"*80)
            print(result['insights']['key_insights'][:500] + "...")
            
            print(f"\n📊 RECOMMENDATIONS:")
            print("-"*80)
            print(result['insights']['recommendations'][:500] + "...")
            
        except Exception as e:
            print(f"\n❌ Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'spotify_dspy_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ TEST COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Datasets analyzed: {len(results)}")
    print(f"📈 View Langfuse dashboard: https://cloud.langfuse.com")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    results = main()
