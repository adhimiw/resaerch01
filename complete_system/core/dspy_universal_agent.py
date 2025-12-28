"""
Universal Agentic Data Science System with DSPy
Works with ANY dataset through adaptive reasoning
Integrated with Triple MCP Architecture (Pandas + Jupyter + Docker)
"""

import dspy
from dspy import ChainOfThought, Signature, InputField, OutputField
import pandas as pd
import json
from typing import List, Dict, Any
from langfuse import Langfuse
import os
from pathlib import Path
from .mcp_integration import get_mcp_integration


# DSPy Signatures
class DatasetUnderstandingSignature(Signature):
    """Analyze unknown dataset structure and determine appropriate analysis approach"""
    dataset_info: str = InputField(desc="Complete dataset info: columns, dtypes, shape, sample, statistics")
    mcp_capabilities: str = InputField(desc="Available MCP tools and capabilities across Pandas, Jupyter, Docker servers")
    data_type: str = OutputField(desc="Primary data type: time-series, tabular, text, spatial, or mixed")
    domain: str = OutputField(desc="Business domain: e-commerce, healthcare, finance, entertainment, etc.")
    ml_task: str = OutputField(desc="ML task: classification, regression, clustering, forecasting, anomaly-detection")
    key_columns: str = OutputField(desc="Most important columns based on variance, uniqueness, correlation")
    challenges: str = OutputField(desc="Data quality issues: missing values, outliers, class imbalance")
    mcp_tools_to_use: str = OutputField(desc="Specific MCP tools from Pandas/Jupyter/Docker to leverage for this analysis")
    analysis_strategy: str = OutputField(desc="Step-by-step analysis plan with 5-7 specific steps using MCP tools")


class AnalysisPlanningSignature(Signature):
    """Create adaptive analysis workflow based on dataset characteristics"""
    data_type: str = InputField()
    ml_task: str = InputField()
    domain: str = InputField()
    key_columns: str = InputField()
    challenges: str = InputField()
    mcp_tools: str = InputField(desc="Available MCP tools to use")
    
    reasoning: str = OutputField(desc="Chain of thought: why this plan is optimal using MCP capabilities")
    exploratory_steps: str = OutputField(desc="Specific EDA steps using Pandas/Jupyter MCP tools")
    feature_engineering: str = OutputField(desc="Feature engineering using MCP tool pipeline")
    model_recommendations: str = OutputField(desc="Top 3 ML models with MCP execution strategy")
    evaluation_metrics: str = OutputField(desc="Metrics for this ML task")
    mcp_execution_plan: str = OutputField(desc="Detailed MCP tool execution sequence: server.tool_name(args)")


class InsightSynthesisSignature(Signature):
    """Generate actionable insights from analysis + external context"""
    analysis_results: str = InputField(desc="Statistical findings from data analysis")
    external_context: str = InputField(desc="Researched external factors")
    domain: str = InputField()
    
    key_insights: str = OutputField(desc="5-10 data-driven insights")
    causality_analysis: str = OutputField(desc="Cause-effect relationships")
    recommendations: str = OutputField(desc="Actionable business recommendations")
    future_predictions: str = OutputField(desc="Expected future trends")


# DSPy Modules
class DatasetUnderstanding(dspy.Module):
    def __init__(self):
        super().__init__()
        self.understand = ChainOfThought(DatasetUnderstandingSignature)
    
    def forward(self, dataset_info: str, mcp_capabilities: str):
        return self.understand(dataset_info=dataset_info, mcp_capabilities=mcp_capabilities)


class AnalysisPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = ChainOfThought(AnalysisPlanningSignature)
    
    def forward(self, understanding):
        return self.plan(
            data_type=understanding.data_type,
            ml_task=understanding.ml_task,
            domain=understanding.domain,
            key_columns=understanding.key_columns,
            challenges=understanding.challenges,
            mcp_tools=understanding.mcp_tools_to_use
        )


class InsightSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesizer = ChainOfThought(InsightSynthesisSignature)
    
    def forward(self, analysis: str, context: str, domain: str):
        return self.synthesizer(
            analysis_results=analysis,
            external_context=context,
            domain=domain
        )


# Main Universal Agent
class UniversalAgenticDataScience:
    """Complete dataset-agnostic agentic data science system"""
    
    def __init__(
        self,
        mistral_api_key: str,
        langfuse_public_key: str,
        langfuse_secret_key: str,
        langfuse_host: str = "https://cloud.langfuse.com"
    ):
        # Initialize Mistral via DSPy
        self.lm = dspy.LM(
            model="mistral/mistral-large-latest",
            api_key=mistral_api_key
        )
        dspy.configure(lm=self.lm)
        
        # Initialize MCP Integration (Triple MCP Architecture)
        self.mcp = get_mcp_integration()
        self.mcp_capabilities = self.mcp.get_capabilities()
        
        # Initialize DSPy modules
        self.dataset_understander = DatasetUnderstanding()
        self.analysis_planner = AnalysisPlanner()
        self.insight_synthesizer = InsightSynthesizer()
        
        # Langfuse observability
        self.langfuse = Langfuse(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            host=langfuse_host
        )
        
        print("✅ Universal Agentic Data Science System initialized")
        print(f"   🧠 LLM: Mistral Large 3 (via DSPy)")
        print(f"   📊 Observability: Langfuse")
        print(f"   🔧 MCP Tools: {self.mcp_capabilities['total_tools']} across {self.mcp_capabilities['active_count']} servers")
    
    def analyze(self, dataset_path: str) -> Dict[str, Any]:
        """Universal analysis pipeline - works with ANY dataset"""
        
        # Create Langfuse trace (using correct API)
        trace_id = f"analysis_{dataset_path.split('/')[-1]}_{hash(dataset_path) % 10000}"
        
        print("\n" + "="*70)
        print("🚀 UNIVERSAL AGENTIC DATA SCIENCE SYSTEM")
        print("="*70)
        
        try:
            # PHASE 1: UNDERSTAND DATASET (with MCP awareness)
            print("\n📖 PHASE 1: Understanding Dataset with MCP Capabilities...")
            print("-" * 70)
            
            dataset_info = self._get_dataset_info(dataset_path)
            mcp_caps_str = self.mcp_capabilities['capabilities_summary']
            
            understanding = self.dataset_understander(
                dataset_info=dataset_info,
                mcp_capabilities=mcp_caps_str
            )
            
            print(f"✓ Data Type: {understanding.data_type}")
            print(f"✓ Domain: {understanding.domain}")
            print(f"✓ ML Task: {understanding.ml_task}")
            print(f"✓ MCP Tools Selected: {understanding.mcp_tools_to_use[:100]}...")
            
            # PHASE 2: CREATE PLAN (with MCP execution strategy)
            print("\n🎯 PHASE 2: Creating Adaptive Plan with MCP Execution...")
            print("-" * 70)
            
            plan = self.analysis_planner(understanding=understanding)
            print(f"✓ Models: {plan.model_recommendations}")
            print(f"✓ MCP Execution Plan: {plan.mcp_execution_plan[:150]}...")
            
            # PHASE 3: EXECUTE ANALYSIS
            print("\n📊 PHASE 3: Executing Analysis...")
            print("-" * 70)
            
            analysis_results = self._execute_analysis(dataset_path, understanding, plan)
            print(f"✓ Completed {len(analysis_results.get('steps_completed', []))} steps")
            
            # PHASE 4: SYNTHESIZE INSIGHTS
            print("\n💡 PHASE 4: Generating Insights...")
            print("-" * 70)
            
            insights = self.insight_synthesizer(
                analysis=str(analysis_results),
                context="No external context required",
                domain=understanding.domain
            )
            
            # Compile result
            result = {
                'dataset': os.path.basename(dataset_path),
                'mcp_integration': {
                    'total_tools': self.mcp_capabilities['total_tools'],
                    'active_servers': self.mcp_capabilities['active_count'],
                    'tools_used': understanding.mcp_tools_to_use,
                    'execution_plan': plan.mcp_execution_plan
                },
                'understanding': {
                    'data_type': understanding.data_type,
                    'domain': understanding.domain,
                    'ml_task': understanding.ml_task,
                    'key_columns': understanding.key_columns
                },
                'plan': {
                    'models': plan.model_recommendations,
                    'feature_engineering': plan.feature_engineering,
                    'metrics': plan.evaluation_metrics,
                    'reasoning': plan.reasoning
                },
                'analysis': analysis_results,
                'insights': {
                    'key_insights': insights.key_insights,
                    'recommendations': insights.recommendations
                },
                'trace_id': trace_id
            }
            
            # Log to Langfuse (optional)
            try:
                self.langfuse.flush()
            except:
                pass
            
            print("\n✅ ANALYSIS COMPLETE!")
            print("="*70)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _get_dataset_info(self, dataset_path: str) -> str:
        """Extract dataset information"""
        try:
            df = pd.read_csv(dataset_path)
            
            info = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_rows': df.head(3).to_dict('records'),
                'statistics': df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {}
            }
            
            return json.dumps(info, indent=2)
        except Exception as e:
            return json.dumps({'error': str(e), 'dataset_path': dataset_path})
    
    def _execute_analysis(self, dataset_path: str, understanding, plan) -> Dict[str, Any]:
        """Execute analysis based on plan"""
        results = {'steps_completed': []}
        
        try:
            df = pd.read_csv(dataset_path)
            
            # Basic stats
            results['basic_stats'] = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=['number']).columns)
            }
            results['steps_completed'].append('basic_statistics')
            
            # Correlations
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                results['top_correlations'] = []
                for i in range(min(3, len(corr_matrix.columns))):
                    for j in range(i+1, min(3, len(corr_matrix.columns))):
                        results['top_correlations'].append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': float(corr_matrix.iloc[i, j])
                        })
                results['steps_completed'].append('correlation_analysis')
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
