"""
Universal Data Science Orchestrator - Phase 5 Integration
Combines all 4 backend phases into a unified workflow:
- Phase 1: ChromaDB RAG (Conversational Interface)
- Phase 2: Data Validation (MLleak)
- Phase 3: Universal ML Agent (MLE-Agent)
- Phase 4: Advanced Optimization (Agent-Lightning)
"""

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import all phase components
try:
    from complete_system.core.conversational_agent import ConversationalDataScienceAgent
    from complete_system.core.data_validation import DataLeakageDetector
    from complete_system.core.mle_agent import MLEAgent
    from complete_system.core.agent_optimizer import AgentLightning
except ModuleNotFoundError:
    # If running from complete_system directory
    from core.conversational_agent import ConversationalDataScienceAgent
    from core.data_validation import DataLeakageDetector
    from core.mle_agent import MLEAgent
    from core.agent_optimizer import AgentLightning


class UniversalDataScienceOrchestrator:
    """
    Main orchestrator that coordinates all 4 backend phases for
    automated data science workflows.
    
    Workflow:
    1. Load dataset
    2. Validate data (Phase 2)
    3. Run ML analysis (Phase 3)
    4. Optimize models (Phase 4)
    5. Store results (Phase 1)
    6. Provide conversational interface (Phase 1)
    """
    
    def __init__(
        self,
        enable_rag: bool = True,
        enable_validation: bool = True,
        enable_mle: bool = True,
        enable_optimization: bool = True,
        optimization_trials: int = 20,
        chroma_persist_dir: str = "./data/chroma_results",
        verbose: bool = True
    ):
        """
        Initialize the orchestrator with all phase components.
        
        Args:
            enable_rag: Enable Phase 1 (RAG) for conversational interface
            enable_validation: Enable Phase 2 (Data Validation)
            enable_mle: Enable Phase 3 (MLE-Agent)
            enable_optimization: Enable Phase 4 (Agent-Lightning)
            optimization_trials: Number of Optuna trials for hyperparameter tuning
            chroma_persist_dir: Directory for ChromaDB persistence
            verbose: Print detailed progress information
        """
        self.enable_rag = enable_rag
        self.enable_validation = enable_validation
        self.enable_mle = enable_mle
        self.enable_optimization = enable_optimization
        self.optimization_trials = optimization_trials
        self.chroma_persist_dir = chroma_persist_dir
        self.verbose = verbose
        
        # Initialize phase components
        self._initialize_components()
        
        # Workflow state
        self.current_dataset = None
        self.current_results = {}
        self.workflow_history = []
        
        if self.verbose:
            print("🚀 Universal Data Science Orchestrator Initialized")
            print(f"   Phase 1 (RAG): {'✅ Enabled' if enable_rag else '❌ Disabled'}")
            print(f"   Phase 2 (Validation): {'✅ Enabled' if enable_validation else '❌ Disabled'}")
            print(f"   Phase 3 (MLE-Agent): {'✅ Enabled' if enable_mle else '❌ Disabled'}")
            print(f"   Phase 4 (Optimization): {'✅ Enabled' if enable_optimization else '❌ Disabled'}")
            print()
    
    def _initialize_components(self):
        """Initialize all phase components."""
        # Phase 1: ChromaDB RAG (Conversational Interface)
        if self.enable_rag:
            try:
                # Create persist directory if it doesn't exist
                os.makedirs(self.chroma_persist_dir, exist_ok=True)
                
                # Initialize MCP integration first
                from core.mcp_integration import MCPIntegration
                self.mcp = MCPIntegration()
                
                self.rag_agent = ConversationalDataScienceAgent(
                    mcp_integration=self.mcp,
                    persist_dir=self.chroma_persist_dir
                )
                if self.verbose:
                    print("✅ Phase 1 (RAG) initialized")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Phase 1 (RAG) initialization warning: {e}")
                self.rag_agent = None
        else:
            self.rag_agent = None
        
        # Phase 2: Data Validation (MLleak)
        if self.enable_validation:
            try:
                self.validator = DataLeakageDetector()
                if self.verbose:
                    print("✅ Phase 2 (Validation) initialized")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Phase 2 (Validation) initialization warning: {e}")
                self.validator = None
        else:
            self.validator = None
        
        # Phase 3: Universal ML Agent (MLE-Agent)
        if self.enable_mle:
            try:
                self.mle_agent = MLEAgent()
                if self.verbose:
                    print("✅ Phase 3 (MLE-Agent) initialized")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Phase 3 (MLE-Agent) initialization warning: {e}")
                self.mle_agent = None
        else:
            self.mle_agent = None
        
        # Phase 4: Advanced Optimization (Agent-Lightning)
        if self.enable_optimization:
            try:
                self.optimizer = AgentLightning(
                    enable_optimization=True,
                    optimization_trials=self.optimization_trials,
                    verbose=False  # Control verbosity at orchestrator level
                )
                if self.verbose:
                    print("✅ Phase 4 (Optimization) initialized")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Phase 4 (Optimization) initialization warning: {e}")
                self.optimizer = None
        else:
            self.optimizer = None
    
    def analyze_dataset(
        self,
        file_path: str,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        run_validation: bool = True,
        run_optimization: bool = True,
        store_results: bool = True
    ) -> Dict[str, Any]:
        """
        Complete end-to-end data science workflow.
        
        Args:
            file_path: Path to dataset CSV file
            target_column: Name of target variable column
            test_size: Test set size (0.0-1.0)
            random_state: Random seed for reproducibility
            run_validation: Run Phase 2 validation
            run_optimization: Run Phase 4 optimization (if False, only Phase 3)
            store_results: Store results in ChromaDB (Phase 1)
        
        Returns:
            Dictionary with complete analysis results from all phases
        """
        start_time = time.time()
        workflow_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.verbose:
            print("\n" + "="*70)
            print(f"🎯 UNIVERSAL DATA SCIENCE WORKFLOW - {workflow_id}")
            print("="*70)
            print(f"📁 Dataset: {file_path}")
            print(f"🎯 Target: {target_column}")
            print(f"⚙️ Test size: {test_size}")
            print(f"🔢 Random state: {random_state}")
            print()
        
        results = {
            'workflow_id': workflow_id,
            'file_path': file_path,
            'target_column': target_column,
            'timestamp': datetime.now().isoformat(),
            'phases': {}
        }
        
        try:
            # Step 1: Load Dataset
            if self.verbose:
                print("📊 Step 1: Loading Dataset...")
            
            df = pd.read_csv(file_path)
            self.current_dataset = df.copy()
            
            results['dataset_info'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'features': list(df.columns),
                'target': target_column,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            }
            
            if self.verbose:
                print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                print(f"   📏 File size: {results['dataset_info']['file_size_mb']:.2f} MB")
                print()
            
            # Step 2: Data Validation (Phase 2)
            # Note: Phase 2 validation requires train/test split
            # For single dataset, we perform basic data quality checks
            if run_validation and self.validator:
                if self.verbose:
                    print("🔍 Step 2: Data Quality Checks...")
                
                validation_start = time.time()
                
                # Basic data quality checks
                validation_results = {
                    'missing_values': df.isnull().sum().sum(),
                    'duplicate_rows': df.duplicated().sum(),
                    'constant_columns': sum(df.nunique() == 1),
                    'data_types': df.dtypes.to_dict()
                }
                
                # Simple risk score based on quality issues
                risk_score = 0
                if validation_results['missing_values'] > len(df) * 0.1:
                    risk_score += 20
                if validation_results['duplicate_rows'] > 0:
                    risk_score += 15
                if validation_results['constant_columns'] > 0:
                    risk_score += 10
                
                # Auto-fix basic issues
                fixed_df = df.copy()
                auto_fix_applied = False
                if risk_score > 0:
                    # Remove duplicates
                    if validation_results['duplicate_rows'] > 0:
                        fixed_df = fixed_df.drop_duplicates()
                        auto_fix_applied = True
                    # Remove constant columns
                    if validation_results['constant_columns'] > 0:
                        for col in fixed_df.columns:
                            if fixed_df[col].nunique() == 1 and col != target_column:
                                fixed_df = fixed_df.drop(columns=[col])
                                auto_fix_applied = True
                
                validation_time = time.time() - validation_start
                
                results['phases']['validation'] = {
                    'risk_score': risk_score,
                    'issues_found': validation_results,
                    'auto_fix_applied': auto_fix_applied,
                    'duration_seconds': validation_time
                }
                
                if self.verbose:
                    print(f"   📊 Risk Score: {risk_score}/100")
                    print(f"   🔧 Auto-fix: {'✅ Applied' if auto_fix_applied else '❌ Not needed'}")
                    print(f"   ⏱️ Duration: {validation_time:.2f}s")
                    print()
                
                # Use fixed data for subsequent phases
                df = fixed_df
            
            # Step 3: Universal ML Analysis (Phase 3)
            if self.mle_agent:
                if self.verbose:
                    print("🤖 Step 3: Universal ML Analysis (Phase 3)...")
                
                mle_start = time.time()
                
                # Load dataset with MLE-Agent
                mle_data = self.mle_agent.load_dataset(file_path, target_column)
                
                # Preprocess data
                processed = self.mle_agent.preprocess_data(mle_data)
                
                # Split into train/test using sklearn
                from sklearn.model_selection import train_test_split
                X = processed['X_train']
                y = processed['y_train']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Update processed data
                processed['X_train'] = X_train
                processed['X_test'] = X_test
                processed['y_train'] = y_train
                processed['y_test'] = y_test
                
                # Train baseline models
                baseline_results = self.mle_agent.train_models(processed)
                
                # Update processed data with task_type from training
                processed['task_type'] = baseline_results['task_type']
                
                # Extract best model info from training results
                best_model = baseline_results['best_model']
                best_score = baseline_results['models'][best_model]['metrics']['cv_mean']
                all_scores = {k: v['metrics']['cv_mean'] for k, v in baseline_results['models'].items()}
                
                mle_time = time.time() - mle_start
                
                results['phases']['mle_agent'] = {
                    'task_type': baseline_results['task_type'],
                    'models_trained': list(baseline_results['models'].keys()),
                    'best_model': best_model,
                    'best_score': best_score,
                    'all_scores': all_scores,
                    'feature_engineering': processed.get('feature_engineering', {}),
                    'duration_seconds': mle_time
                }
                
                if self.verbose:
                    print(f"   📈 Task Type: {baseline_results['task_type']}")
                    print(f"   🏆 Best Model: {best_model}")
                    print(f"   📊 Best Score: {best_score:.4f}")
                    print(f"   ⏱️ Duration: {mle_time:.2f}s")
                    print()
                
                # Store processed data for optimization
                self.current_results['processed_data'] = processed
                self.current_results['baseline_results'] = baseline_results
            
            # Step 4: Advanced Optimization (Phase 4)
            if run_optimization and self.optimizer and self.mle_agent:
                if self.verbose:
                    print("⚡ Step 4: Advanced Optimization (Phase 4)...")
                
                opt_start = time.time()
                
                processed = self.current_results['processed_data']
                
                # Train advanced models with optimization
                opt_results = self.optimizer.train_advanced_models(
                    X_train=processed['X_train'],
                    y_train=processed['y_train'],
                    X_test=processed['X_test'],
                    task_type=processed['task_type'],
                    optimize=True
                )
                
                # Create ensemble
                ensemble = self.optimizer.create_ensemble(task_type=processed['task_type'])
                ensemble.fit(processed['X_train'], processed['y_train'])
                
                # Compute SHAP values for best model
                best_model_name = opt_results['best_model']
                
                # Get best CV score from models dict
                best_cv_score = opt_results['models'][best_model_name]['cv_mean']
                cv_scores = {k: {'mean': v['cv_mean'], 'std': v['cv_std']} 
                            for k, v in opt_results['models'].items()}
                
                shap_results = self.optimizer.compute_shap_values(
                    X_train=processed['X_train'],
                    model_name=best_model_name,
                    max_samples=100
                )
                
                # Evaluate ensemble
                ensemble_train_score = ensemble.score(processed['X_train'], processed['y_train'])
                ensemble_test_score = ensemble.score(processed['X_test'], processed['y_test'])
                
                opt_time = time.time() - opt_start
                
                results['phases']['optimization'] = {
                    'models_trained': list(opt_results['models'].keys()),
                    'best_model': best_model_name,
                    'best_cv_score': best_cv_score,
                    'model_scores': cv_scores,
                    'ensemble_train_score': ensemble_train_score,
                    'ensemble_test_score': ensemble_test_score,
                    'top_features': shap_results['top_5_features'],
                    'feature_importance': shap_results['feature_importance'],
                    'duration_seconds': opt_time
                }
                
                if self.verbose:
                    print(f"   🏆 Best Model: {best_model_name}")
                    print(f"   📊 Best CV Score: {best_cv_score:.4f}")
                    print(f"   🎯 Ensemble Test Score: {ensemble_test_score:.4f}")
                    print(f"   🔝 Top Features: {shap_results['top_5_features'][:3]}")
                    print(f"   ⏱️ Duration: {opt_time:.2f}s")
                    print()
                
                # Store ensemble for future predictions
                self.current_results['ensemble'] = ensemble
                self.current_results['shap_results'] = shap_results
            
            # Step 5: Store Results in ChromaDB (Phase 1)
            if store_results and self.rag_agent:
                if self.verbose:
                    print("💾 Step 5: Storing Results (Phase 1)...")
                
                storage_start = time.time()
                
                # Prepare result text for ChromaDB
                result_text = self._format_results_for_storage(results)
                
                # Store in ChromaDB
                self.rag_agent.add_to_collection(
                    collection_name='results',
                    text=result_text,
                    metadata={
                        'workflow_id': workflow_id,
                        'dataset': file_path,
                        'target': target_column,
                        'timestamp': results['timestamp']
                    }
                )
                
                storage_time = time.time() - storage_start
                
                results['phases']['storage'] = {
                    'stored_in_chromadb': True,
                    'collection': 'results',
                    'duration_seconds': storage_time
                }
                
                if self.verbose:
                    print(f"   ✅ Results stored in ChromaDB")
                    print(f"   ⏱️ Duration: {storage_time:.2f}s")
                    print()
            
            # Calculate total time
            total_time = time.time() - start_time
            results['total_duration_seconds'] = total_time
            
            if self.verbose:
                print("="*70)
                print(f"✅ WORKFLOW COMPLETE - Total Time: {total_time:.2f}s")
                print("="*70)
                print()
            
            # Store in history
            self.workflow_history.append(results)
            self.current_results['workflow_results'] = results
            
            return results
        
        except Exception as e:
            error_time = time.time() - start_time
            results['error'] = str(e)
            results['total_duration_seconds'] = error_time
            
            if self.verbose:
                print(f"❌ Workflow Error: {e}")
            
            raise
    
    def _format_results_for_storage(self, results: Dict[str, Any]) -> str:
        """Format analysis results as text for ChromaDB storage."""
        lines = []
        
        lines.append(f"Workflow ID: {results['workflow_id']}")
        lines.append(f"Dataset: {results['file_path']}")
        lines.append(f"Target: {results['target_column']}")
        lines.append(f"Timestamp: {results['timestamp']}")
        lines.append("")
        
        # Dataset info
        if 'dataset_info' in results:
            info = results['dataset_info']
            lines.append("Dataset Information:")
            lines.append(f"  Rows: {info['rows']}")
            lines.append(f"  Columns: {info['columns']}")
            lines.append(f"  Features: {', '.join(info['features'][:5])}")
            lines.append("")
        
        # Validation results
        if 'validation' in results['phases']:
            val = results['phases']['validation']
            lines.append("Data Validation (Phase 2):")
            lines.append(f"  Risk Score: {val['risk_score']}/100")
            lines.append(f"  Auto-fix Applied: {val['auto_fix_applied']}")
            lines.append("")
        
        # MLE-Agent results
        if 'mle_agent' in results['phases']:
            mle = results['phases']['mle_agent']
            lines.append("Universal ML Analysis (Phase 3):")
            lines.append(f"  Task Type: {mle['task_type']}")
            lines.append(f"  Best Model: {mle['best_model']}")
            best_model_key = mle['best_model']
            lines.append(f"  Best Score: {mle['all_scores'][best_model_key]:.4f}")
            lines.append("")
        
        # Optimization results
        if 'optimization' in results['phases']:
            opt = results['phases']['optimization']
            lines.append("Advanced Optimization (Phase 4):")
            lines.append(f"  Best Model: {opt['best_model']}")
            lines.append(f"  Best CV Score: {opt['best_cv_score']:.4f}")
            lines.append(f"  Ensemble Test Score: {opt['ensemble_test_score']:.4f}")
            lines.append(f"  Top Features: {', '.join(opt['top_features'][:5])}")
            lines.append("")
        
        lines.append(f"Total Duration: {results['total_duration_seconds']:.2f}s")
        
        return "\n".join(lines)
    
    def chat(self, question: str, use_history: bool = True) -> str:
        """
        Conversational interface using Phase 1 RAG.
        
        Args:
            question: Natural language question about analysis
            use_history: Include workflow history in context
        
        Returns:
            Answer from RAG system
        """
        if not self.rag_agent:
            return "Error: RAG system not enabled. Set enable_rag=True."
        
        try:
            # Add current results context
            context = []
            if use_history and self.current_results:
                context.append("Current Analysis Context:")
                if 'workflow_results' in self.current_results:
                    results = self.current_results['workflow_results']
                    context.append(f"Dataset: {results.get('file_path', 'N/A')}")
                    context.append(f"Target: {results.get('target_column', 'N/A')}")
                    
                    if 'optimization' in results.get('phases', {}):
                        opt = results['phases']['optimization']
                        context.append(f"Best Model: {opt['best_model']}")
                        context.append(f"Test Score: {opt['ensemble_test_score']:.4f}")
            
            # Query RAG
            full_question = "\n".join(context) + "\n\nQuestion: " + question if context else question
            answer = self.rag_agent.query('results', full_question, n_results=3)
            
            return answer
        
        except Exception as e:
            return f"Error querying RAG: {e}"
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of all completed workflows."""
        return {
            'total_workflows': len(self.workflow_history),
            'workflows': self.workflow_history
        }
    
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained ensemble from Phase 4.
        
        Args:
            X_new: New data for predictions (must match training features)
        
        Returns:
            Predictions array
        """
        if 'ensemble' not in self.current_results:
            raise ValueError("No ensemble model available. Run analyze_dataset first.")
        
        ensemble = self.current_results['ensemble']
        processed = self.current_results['processed_data']
        
        # Ensure same feature order
        feature_columns = processed['X_train'].columns
        X_new_ordered = X_new[feature_columns]
        
        predictions = ensemble.predict(X_new_ordered)
        return predictions
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            output_file: Optional file path to save report
        
        Returns:
            Report text
        """
        if not self.current_results or 'workflow_results' not in self.current_results:
            return "No analysis results available. Run analyze_dataset first."
        
        results = self.current_results['workflow_results']
        
        lines = []
        lines.append("="*70)
        lines.append("UNIVERSAL DATA SCIENCE ANALYSIS REPORT")
        lines.append("="*70)
        lines.append("")
        
        # Header
        lines.append(f"Workflow ID: {results['workflow_id']}")
        lines.append(f"Dataset: {results['file_path']}")
        lines.append(f"Target: {results['target_column']}")
        lines.append(f"Timestamp: {results['timestamp']}")
        lines.append(f"Total Duration: {results['total_duration_seconds']:.2f}s")
        lines.append("")
        
        # Dataset Info
        if 'dataset_info' in results:
            info = results['dataset_info']
            lines.append("DATASET INFORMATION")
            lines.append("-" * 70)
            lines.append(f"Rows: {info['rows']:,}")
            lines.append(f"Columns: {info['columns']}")
            lines.append(f"File Size: {info['file_size_mb']:.2f} MB")
            lines.append(f"Features: {', '.join(info['features'])}")
            lines.append("")
        
        # Phase 2: Validation
        if 'validation' in results['phases']:
            val = results['phases']['validation']
            lines.append("PHASE 2: DATA VALIDATION")
            lines.append("-" * 70)
            lines.append(f"Risk Score: {val['risk_score']}/100")
            lines.append(f"Auto-fix Applied: {val['auto_fix_applied']}")
            lines.append(f"Duration: {val['duration_seconds']:.2f}s")
            lines.append("")
        
        # Phase 3: MLE-Agent
        if 'mle_agent' in results['phases']:
            mle = results['phases']['mle_agent']
            lines.append("PHASE 3: UNIVERSAL ML ANALYSIS")
            lines.append("-" * 70)
            lines.append(f"Task Type: {mle['task_type']}")
            lines.append(f"Models Trained: {', '.join(mle['models_trained'])}")
            lines.append(f"Best Model: {mle['best_model']}")
            best_model_key = mle['best_model']
            lines.append(f"Best Score: {mle['all_scores'][best_model_key]:.4f}")
            lines.append("")
            lines.append("All Model Scores:")
            for model, score in mle['all_scores'].items():
                lines.append(f"  {model}: {score:.4f}")
            lines.append(f"Duration: {mle['duration_seconds']:.2f}s")
            lines.append("")
        
        # Phase 4: Optimization
        if 'optimization' in results['phases']:
            opt = results['phases']['optimization']
            lines.append("PHASE 4: ADVANCED OPTIMIZATION")
            lines.append("-" * 70)
            lines.append(f"Models Trained: {', '.join(opt['models_trained'])}")
            lines.append(f"Best Model: {opt['best_model']}")
            lines.append(f"Best CV Score: {opt['best_cv_score']:.4f}")
            lines.append("")
            lines.append("Model CV Scores:")
            for model, score_info in opt['model_scores'].items():
                lines.append(f"  {model}: {score_info['mean']:.4f} (±{score_info['std']:.4f})")
            lines.append("")
            lines.append(f"Ensemble Train Score: {opt['ensemble_train_score']:.4f}")
            lines.append(f"Ensemble Test Score: {opt['ensemble_test_score']:.4f}")
            lines.append("")
            lines.append("Top 5 Features (SHAP):")
            for i, feature in enumerate(opt['top_features'][:5], 1):
                importance = opt['feature_importance'][feature]
                lines.append(f"  {i}. {feature}: {importance:.6f}")
            lines.append(f"Duration: {opt['duration_seconds']:.2f}s")
            lines.append("")
        
        lines.append("="*70)
        
        report_text = "\n".join(lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            if self.verbose:
                print(f"📄 Report saved to: {output_file}")
        
        return report_text


def quick_analyze(
    file_path: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    optimization_trials: int = 20
) -> Dict[str, Any]:
    """
    Convenience function for quick end-to-end analysis.
    
    Args:
        file_path: Path to dataset CSV
        target_column: Target variable name
        test_size: Test set proportion
        random_state: Random seed
        optimization_trials: Number of Optuna trials
    
    Returns:
        Complete analysis results
    """
    orchestrator = UniversalDataScienceOrchestrator(
        enable_rag=True,
        enable_validation=True,
        enable_mle=True,
        enable_optimization=True,
        optimization_trials=optimization_trials,
        verbose=True
    )
    
    results = orchestrator.analyze_dataset(
        file_path=file_path,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        run_validation=True,
        run_optimization=True,
        store_results=True
    )
    
    return results


if __name__ == "__main__":
    print("Universal Data Science Orchestrator - Phase 5")
    print("Usage:")
    print("  from complete_system.orchestrator import UniversalDataScienceOrchestrator")
    print("  orchestrator = UniversalDataScienceOrchestrator()")
    print("  results = orchestrator.analyze_dataset('data.csv', 'target')")
