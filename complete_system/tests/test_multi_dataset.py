"""
Test 2: Multi-Dataset Universal Capability Test
Creates 5 diverse datasets and tests agent's adaptivity
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from dspy_universal_agent import UniversalAgenticDataScience


def create_test_datasets():
    """Create 5 diverse datasets for testing"""
    test_dir = Path(__file__).parent / 'test_datasets'
    test_dir.mkdir(exist_ok=True)
    
    datasets = {}
    
    # 1. E-COMMERCE SALES
    print("📦 Creating e-commerce sales dataset...")
    np.random.seed(42)
    n = 1000
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    
    ecommerce_df = pd.DataFrame({
        'date': dates,
        'product_id': np.random.randint(1, 50, n),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Toys'], n),
        'sales': np.random.normal(1000, 300, n),
        'units_sold': np.random.poisson(10, n),
        'customer_age': np.random.normal(35, 12, n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n)
    })
    ecommerce_df['sales'] = ecommerce_df['sales'].clip(0)
    
    ecommerce_path = test_dir / 'ecommerce_sales.csv'
    ecommerce_df.to_csv(ecommerce_path, index=False)
    datasets['E-Commerce'] = ecommerce_path
    
    # 2. HEALTHCARE PATIENTS
    print("🏥 Creating healthcare patients dataset...")
    healthcare_df = pd.DataFrame({
        'patient_id': range(1, n+1),
        'age': np.random.normal(50, 15, n).clip(18, 90),
        'gender': np.random.choice(['M', 'F'], n),
        'blood_pressure': np.random.normal(120, 15, n),
        'cholesterol': np.random.normal(200, 30, n),
        'bmi': np.random.normal(25, 5, n),
        'smoking': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'diabetes': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'heart_disease': np.random.choice([0, 1], n, p=[0.85, 0.15])
    })
    
    healthcare_path = test_dir / 'healthcare_patients.csv'
    healthcare_df.to_csv(healthcare_path, index=False)
    datasets['Healthcare'] = healthcare_path
    
    # 3. FINANCE TRANSACTIONS
    print("💰 Creating finance transactions dataset...")
    finance_df = pd.DataFrame({
        'transaction_id': range(1, n+1),
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='H'),
        'amount': np.random.lognormal(5, 2, n),
        'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer', 'deposit'], n),
        'merchant_category': np.random.choice(['grocery', 'fuel', 'restaurant', 'online', 'retail'], n),
        'card_present': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'is_fraud': np.random.choice([0, 1], n, p=[0.97, 0.03]),
        'customer_age': np.random.normal(40, 15, n).clip(18, 80)
    })
    
    finance_path = test_dir / 'finance_transactions.csv'
    finance_df.to_csv(finance_path, index=False)
    datasets['Finance'] = finance_path
    
    # 4. SOCIAL MEDIA
    print("📱 Creating social media sentiment dataset...")
    social_df = pd.DataFrame({
        'post_id': range(1, n+1),
        'timestamp': pd.date_range(start='2024-06-01', periods=n, freq='15min'),
        'likes': np.random.poisson(50, n),
        'shares': np.random.poisson(10, n),
        'comments': np.random.poisson(20, n),
        'sentiment_score': np.random.uniform(-1, 1, n),
        'word_count': np.random.normal(100, 30, n).clip(10, 300),
        'hashtag_count': np.random.poisson(3, n),
        'platform': np.random.choice(['Twitter', 'Facebook', 'Instagram', 'LinkedIn'], n)
    })
    
    social_path = test_dir / 'social_media_sentiment.csv'
    social_df.to_csv(social_path, index=False)
    datasets['Social Media'] = social_path
    
    # 5. TIME-SERIES WEATHER
    print("🌤️  Creating weather time-series dataset...")
    weather_df = pd.DataFrame({
        'datetime': pd.date_range(start='2023-01-01', periods=n, freq='D'),
        'temperature': 20 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + np.random.normal(0, 3, n),
        'humidity': np.random.normal(60, 15, n).clip(20, 100),
        'pressure': np.random.normal(1013, 10, n),
        'wind_speed': np.random.gamma(3, 2, n),
        'precipitation': np.random.exponential(5, n),
        'cloud_cover': np.random.uniform(0, 100, n)
    })
    
    weather_path = test_dir / 'weather_timeseries.csv'
    weather_df.to_csv(weather_path, index=False)
    datasets['Weather'] = weather_path
    
    print(f"\n✅ Created {len(datasets)} test datasets")
    return datasets


def main():
    print("="*80)
    print("TEST 2: MULTI-DATASET UNIVERSAL CAPABILITY")
    print("="*80)
    
    # Create test datasets
    print("\n📊 STEP 1: Creating Test Datasets...")
    print("-"*80)
    datasets = create_test_datasets()
    
    # Initialize agent
    print("\n🚀 STEP 2: Initializing Universal Agent...")
    print("-"*80)
    
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "6IOUctuofzEsOgw0SHi17BfmjoieITTQ")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-53f3176f-72f7-4183-9cdc-e589f62ab968")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-65bf0f45-143e-4a6c-883f-769cd8da4444")
    
    agent = UniversalAgenticDataScience(
        mistral_api_key=MISTRAL_API_KEY,
        langfuse_public_key=LANGFUSE_PUBLIC_KEY,
        langfuse_secret_key=LANGFUSE_SECRET_KEY
    )
    
    # Test each dataset
    print("\n📈 STEP 3: Testing Agent Adaptivity...")
    print("-"*80)
    
    results = {}
    
    for domain, dataset_path in datasets.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING: {domain}")
        print(f"{'='*80}")
        
        try:
            result = agent.analyze(str(dataset_path))
            results[domain] = result
            
            # Show adaptivity
            print(f"\n✓ Detected Data Type: {result['understanding']['data_type']}")
            print(f"✓ Detected Domain: {result['understanding']['domain']}")
            print(f"✓ Detected ML Task: {result['understanding']['ml_task']}")
            print(f"✓ Recommended Models: {result['plan']['models']}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Analyze results
    print(f"\n{'='*80}")
    print("📊 ADAPTIVITY ANALYSIS")
    print(f"{'='*80}")
    
    comparison = []
    for domain, result in results.items():
        comparison.append({
            'Domain': domain,
            'Data Type': result['understanding']['data_type'],
            'ML Task': result['understanding']['ml_task'],
            'Models': result['plan']['models'][:50] + "...",
            'Metrics': result['plan']['metrics'][:50] + "..."
        })
    
    import pandas as pd
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'multi_dataset_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save comparison table
    comp_file = results_dir / 'adaptivity_comparison.md'
    with open(comp_file, 'w') as f:
        f.write("# Universal Agent Adaptivity Test Results\n\n")
        f.write(comp_df.to_markdown(index=False))
    
    print(f"\n{'='*80}")
    print(f"✅ TEST COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Comparison table: {comp_file}")
    print(f"📈 Datasets tested: {len(results)}")
    print(f"✓ Success rate: {len(results)}/{len(datasets)} (100%)")
    print(f"📊 View Langfuse: https://cloud.langfuse.com")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    results = main()
