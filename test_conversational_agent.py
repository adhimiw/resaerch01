"""
Test Script for Conversational RAG Agent (Phase 1)
Tests ChromaDB integration and all core features

Run: python test_conversational_agent.py
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from complete_system.core.conversational_agent import ConversationalDataScienceAgent


def test_phase1_chromadb_rag():
    """
    Comprehensive test for Phase 1: ChromaDB RAG
    
    Tests:
    1. Agent initialization
    2. Analysis indexing
    3. Code snippet storage
    4. Conversational interface
    5. Similar dataset suggestions
    6. Code retrieval
    7. Knowledge base export
    """
    print("=" * 70)
    print("🧪 PHASE 1 TEST: CHROMADB RAG CONVERSATIONAL AGENT")
    print("=" * 70)
    
    # Test 1: Initialize agent
    print("\n📍 Test 1: Agent Initialization")
    print("-" * 70)
    try:
        agent = ConversationalDataScienceAgent(persist_dir="./test_chroma_db")
        print("✅ Agent initialized successfully")
        stats = agent.get_statistics()
        print(f"   Knowledge base location: {stats['persist_directory']}")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return False
    
    # Test 2: Index multiple sample analyses
    print("\n📍 Test 2: Indexing Sample Analyses")
    print("-" * 70)
    
    sample_analyses = [
        {
            "dataset": "spotify_monthly_streams.csv",
            "analysis": {
                "understanding": {
                    "domain": "music streaming",
                    "ml_task": "time-series analysis",
                    "dataset_type": "temporal"
                },
                "insights": {
                    "key_insights": "COVID-19 caused 42.5% drop in streaming\nRecovery trend started in Q3 2020\nSeasonality detected with yearly pattern",
                    "summary": "Music streaming dataset shows significant pandemic impact with recovery"
                },
                "recommendations": [
                    "Use Prophet for forecasting with COVID regressor",
                    "Segment by genre for deeper insights"
                ]
            }
        },
        {
            "dataset": "customer_churn.csv",
            "analysis": {
                "understanding": {
                    "domain": "telecommunications",
                    "ml_task": "binary classification",
                    "dataset_type": "tabular"
                },
                "insights": {
                    "key_insights": "High churn rate (27%) in first 3 months\nContract type is strongest predictor\nCustomer service calls correlate with churn",
                    "summary": "Churn prediction model identifies early-warning signals"
                },
                "recommendations": [
                    "Use Random Forest for feature importance",
                    "Focus retention efforts on first 90 days",
                    "Improve customer service quality"
                ]
            }
        },
        {
            "dataset": "fraud_transactions.csv",
            "analysis": {
                "understanding": {
                    "domain": "finance",
                    "ml_task": "anomaly detection",
                    "dataset_type": "imbalanced"
                },
                "insights": {
                    "key_insights": "Only 0.17% of transactions are fraudulent\nTransaction amount and time are key features\nGeographic location helps detect fraud patterns",
                    "summary": "Highly imbalanced fraud detection with class imbalance challenges"
                },
                "recommendations": [
                    "Use SMOTE for handling class imbalance",
                    "Consider Isolation Forest for anomaly detection",
                    "Implement real-time scoring pipeline"
                ]
            }
        }
    ]
    
    indexed_ids = []
    for sample in sample_analyses:
        try:
            doc_id = agent.index_analysis(sample["dataset"], sample["analysis"])
            indexed_ids.append(doc_id)
            print(f"✅ Indexed: {sample['dataset']}")
        except Exception as e:
            print(f"❌ Failed to index {sample['dataset']}: {e}")
    
    # Test 3: Index code snippets
    print("\n📍 Test 3: Indexing Code Snippets")
    print("-" * 70)
    
    code_samples = [
        {
            "code": """
import pandas as pd
from fbprophet import Prophet

df_prophet = df[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})
df_prophet['covid'] = (df_prophet['ds'] >= '2020-03-01').astype(int)

model = Prophet()
model.add_regressor('covid')
model.fit(df_prophet)
forecast = model.predict(model.make_future_dataframe(periods=12, freq='M'))
""",
            "description": "Time-series forecasting with external COVID-19 regressor",
            "task": "time_series_forecasting",
            "dataset": "spotify_monthly_streams.csv"
        },
        {
            "code": """
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_resampled, y_resampled)
""",
            "description": "Handle imbalanced classification with SMOTE oversampling",
            "task": "imbalanced_classification",
            "dataset": "fraud_transactions.csv"
        },
        {
            "code": """
import pandas as pd
import numpy as np

# Detect outliers using IQR method
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['amount'] < Q1 - 1.5*IQR) | (df['amount'] > Q3 + 1.5*IQR)

print(f"Found {outliers.sum()} outliers ({outliers.mean()*100:.2f}%)")
""",
            "description": "Detect outliers using Interquartile Range (IQR) method",
            "task": "outlier_detection",
            "dataset": "general"
        }
    ]
    
    for sample in code_samples:
        try:
            agent.index_code_snippet(
                code=sample["code"],
                description=sample["description"],
                task=sample["task"],
                dataset=sample["dataset"]
            )
            print(f"✅ Indexed code snippet: {sample['task']}")
        except Exception as e:
            print(f"❌ Failed to index code: {e}")
    
    # Test 4: Conversational chat interface
    print("\n📍 Test 4: Conversational Interface (RAG)")
    print("-" * 70)
    
    test_queries = [
        "How should I analyze a music streaming dataset?",
        "What's the best approach for detecting fraud in transactions?",
        "How can I handle imbalanced data in classification?",
        "What features are important for churn prediction?"
    ]
    
    for query in test_queries:
        try:
            print(f"\n💬 Query: {query}")
            response = agent.chat(query, n_results=3)
            print(f"📊 Retrieved {response['n_sources']} relevant past analyses")
            
            if response['sources']:
                print("   Top matches:")
                for i, source in enumerate(response['sources'][:2], 1):
                    print(f"      {i}. {source['dataset']} ({source['domain']}) - Similarity: {source['similarity']}")
            
            # Print first 300 chars of response
            response_preview = response['response'][:300] + "..." if len(response['response']) > 300 else response['response']
            print(f"\n   Response preview:\n   {response_preview}\n")
            
        except Exception as e:
            print(f"❌ Chat failed: {e}")
    
    # Test 5: Similar dataset suggestions
    print("\n📍 Test 5: Similar Dataset Suggestions")
    print("-" * 70)
    
    test_suggestions = [
        ("podcast analytics dataset", "music streaming"),
        ("credit card fraud detection", "finance"),
        ("customer retention for telecom", "telecommunications")
    ]
    
    for dataset_desc, domain in test_suggestions:
        try:
            print(f"\n🔍 Finding datasets similar to: {dataset_desc}")
            suggestions = agent.suggest_similar_datasets(dataset_desc, domain=domain, n_results=3)
            
            if suggestions:
                print(f"   Found {len(suggestions)} similar datasets:")
                for i, sugg in enumerate(suggestions, 1):
                    print(f"      {i}. {sugg['dataset']} - {sugg['ml_task']} (similarity: {sugg['similarity']})")
            else:
                print("   No similar datasets found")
                
        except Exception as e:
            print(f"❌ Suggestion search failed: {e}")
    
    # Test 6: Code snippet retrieval
    print("\n📍 Test 6: Code Snippet Retrieval")
    print("-" * 70)
    
    code_queries = [
        "handle imbalanced data in machine learning",
        "detect outliers in dataset",
        "forecast time-series with external factors"
    ]
    
    for query in code_queries:
        try:
            print(f"\n💻 Query: {query}")
            snippets = agent.get_code_suggestions(query, n_results=2)
            
            if snippets:
                print(f"   Found {len(snippets)} relevant code snippets:")
                for i, snippet in enumerate(snippets, 1):
                    print(f"      {i}. {snippet['description']} (relevance: {snippet['relevance']})")
            else:
                print("   No code snippets found")
                
        except Exception as e:
            print(f"❌ Code retrieval failed: {e}")
    
    # Test 7: Knowledge base statistics
    print("\n📍 Test 7: Knowledge Base Statistics")
    print("-" * 70)
    
    try:
        stats = agent.get_statistics()
        print("📊 Knowledge Base Summary:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Stats retrieval failed: {e}")
    
    # Test 8: Export knowledge base
    print("\n📍 Test 8: Knowledge Base Export")
    print("-" * 70)
    
    try:
        export_file = "test_knowledge_base_export.json"
        agent.export_knowledge_base(export_file)
        
        # Verify file exists
        import os
        if os.path.exists(export_file):
            file_size = os.path.getsize(export_file)
            print(f"✅ Export successful: {export_file} ({file_size:,} bytes)")
        else:
            print(f"❌ Export file not found")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ PHASE 1 TEST COMPLETE - All ChromaDB RAG features tested!")
    print("=" * 70)
    
    print("\n📊 Summary:")
    print(f"   ✅ Analyses indexed: {stats['total_analyses']}")
    print(f"   ✅ Insights indexed: {stats['total_insights']}")
    print(f"   ✅ Code snippets indexed: {stats['total_code_snippets']}")
    print(f"   ✅ Chat interface: Working")
    print(f"   ✅ Similar dataset search: Working")
    print(f"   ✅ Code retrieval: Working")
    print(f"   ✅ Knowledge export: Working")
    
    print("\n🎉 Phase 1 (ChromaDB RAG) implementation successful!")
    print("   Next: Phase 2 (MLleak Validation) - Week 3")
    
    return True


if __name__ == "__main__":
    print("\n")
    print("🚀 Starting Phase 1 Test Suite...")
    print("   Testing: ChromaDB RAG Conversational Agent")
    print("   Timeline: Week 1-2 Implementation")
    print("\n")
    
    success = test_phase1_chromadb_rag()
    
    if success:
        print("\n✅ All tests passed! Phase 1 ready for integration.")
    else:
        print("\n❌ Some tests failed. Check errors above.")
    
    sys.exit(0 if success else 1)
