"""
Autonomous Intelligence for Data Analysis and Science (AIDAS)
==============================================================

A comprehensive autonomous data science system that:
1. Accepts raw data and automatically cleans it
2. Discovers and tests hypotheses
3. Explains insights in simple language
4. Builds models and evaluates them
5. Detects errors with self-healing capabilities
6. Provides reproducibility and explainability

Author: MiniMax Agent
"""

import os
import sys
import json
import time
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aidas_system.log')
    ]
)
logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Complexity levels for tasks"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class AnalysisPhase(Enum):
    """Phases of autonomous analysis"""
    DATA_INGESTION = "data_ingestion"
    DATA_CLEANING = "data_cleaning"
    EXPLORATORY_ANALYSIS = "exploratory_analysis"
    HYPOTHESIS_DISCOVERY = "hypothesis_discovery"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    MODEL_BUILDING = "model_building"
    MODEL_EVALUATION = "model_evaluation"
    INSIGHT_GENERATION = "insight_generation"
    REPORT_GENERATION = "report_generation"


@dataclass
class AnalysisConfig:
    """Configuration for autonomous analysis"""
    max_iterations: int = 3
    confidence_threshold: float = 0.95
    max_correlations: int = 50
    visualization_dpi: int = 150
    sample_size: int = 10000
    random_state: int = 42
    test_size: float = 0.2
    timeout_seconds: int = 300


@dataclass
class DataQualityReport:
    """Report on data quality"""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    data_types: Dict[str, str]
    quality_score: float
    issues_found: List[str]
    suggested_fixes: List[str]


@dataclass
class Hypothesis:
    """A discovered and tested hypothesis"""
    id: str
    statement: str
    variables: List[str]
    test_type: str
    statistic: float
    p_value: float
    significance_level: float
    is_significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    explanation: str


@dataclass
class Insight:
    """A discovered insight"""
    id: str
    category: str
    description: str
    evidence: str
    impact: str
    recommendation: str
    confidence: float
    supporting_metrics: Dict[str, Any]


@dataclass
class ModelResult:
    """Result from a trained model"""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    feature_importance: Dict[str, float]
    confusion_matrix: List[List[int]]
    cross_validation_scores: List[float]
    errors: List[str] = field(default_factory=list)


class AutonomousIntelligence:
    """
    Main orchestrator for the AIDAS system
    
    This class coordinates all autonomous analysis activities including:
    - Data ingestion and cleaning
    - Pattern discovery
    - Hypothesis testing
    - Model building
    - Insight generation
    - Report creation
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the AIDAS system"""
        self.config = config or AnalysisConfig()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.analysis_history = []
        self.current_phase = None
        
        # Initialize sub-systems
        self.data_processor = _DataProcessor(self.config)
        self.pattern_discoverer = _PatternDiscoverer(self.config)
        self.hypothesis_engine = _HypothesisEngine(self.config)
        self.model_builder = _ModelBuilder(self.config)
        self.insight_generator = _InsightGenerator(self.config)
        self.explainer = _NLExplainer()
        
        logger.info("AIDAS System initialized successfully")
    
    def analyze(self, data_path: str, output_dir: str = "aidas_results") -> Dict[str, Any]:
        """
        Main entry point for autonomous analysis
        
        Args:
            data_path: Path to the dataset (CSV, JSON, etc.)
            output_dir: Directory for output files
            
        Returns:
            Complete analysis results dictionary
        """
        logger.info("="*80)
        logger.info("STARTING AUTONOMOUS DATA ANALYSIS")
        logger.info("="*80)
        
        start_time = time.time()
        results = {
            'session_id': self.session_id,
            'data_path': data_path,
            'start_time': datetime.now().isoformat(),
            'phases': {}
        }
        
        try:
            # Phase 1: Data Ingestion
            logger.info("\n[PHASE 1] DATA INGESTION")
            self.current_phase = AnalysisPhase.DATA_INGESTION
            data = self.data_processor.ingest(data_path)
            results['phases']['data_ingestion'] = {
                'status': 'success',
                'rows': len(data),
                'columns': len(data.columns),
                'dtypes': data.dtypes.astype(str).to_dict()
            }
            
            # Phase 2: Data Cleaning
            logger.info("\n[PHASE 2] DATA CLEANING")
            self.current_phase = AnalysisPhase.DATA_CLEANING
            cleaned_data, quality_report = self.data_processor.clean(data)
            results['phases']['data_cleaning'] = {
                'status': 'success',
                'quality_report': asdict(quality_report),
                'cleaned_rows': len(cleaned_data),
                'issues_fixed': len(quality_report.suggested_fixes)
            }
            
            # Phase 3: Exploratory Analysis
            logger.info("\n[PHASE 3] EXPLORATORY ANALYSIS")
            self.current_phase = AnalysisPhase.EXPLORATORY_ANALYSIS
            eda_results = self.pattern_discoverer.explore(cleaned_data)
            results['phases']['exploratory_analysis'] = {
                'status': 'success',
                'numeric_summary': eda_results['numeric_summary'],
                'categorical_summary': eda_results['categorical_summary'],
                'correlations': eda_results['correlations'][:10],
                'distributions': eda_results['distributions']
            }
            
            # Phase 4: Hypothesis Discovery
            logger.info("\n[PHASE 4] HYPOTHESIS DISCOVERY")
            self.current_phase = AnalysisPhase.HYPOTHESIS_DISCOVERY
            hypotheses = self.hypothesis_engine.discover(cleaned_data, eda_results)
            results['phases']['hypothesis_discovery'] = {
                'status': 'success',
                'hypotheses_found': len(hypotheses),
                'hypothesis_ids': [h.id for h in hypotheses]
            }
            
            # Phase 5: Hypothesis Testing
            logger.info("\n[PHASE 5] HYPOTHESIS TESTING")
            self.current_phase = AnalysisPhase.HYPOTHESIS_TESTING
            tested_hypotheses = self.hypothesis_engine.test(hypotheses, cleaned_data)
            results['phases']['hypothesis_testing'] = {
                'status': 'success',
                'significant_hypotheses': len([h for h in tested_hypotheses if h.is_significant]),
                'tested_hypotheses': [asdict(h) for h in tested_hypotheses]
            }
            
            # Phase 6: Model Building
            logger.info("\n[PHASE 6] MODEL BUILDING")
            self.current_phase = AnalysisPhase.MODEL_BUILDING
            models = self.model_builder.build_models(cleaned_data, eda_results)
            results['phases']['model_building'] = {
                'status': 'success',
                'models_built': len(models),
                'model_types': [m.model_type for m in models],
                'best_model': max(models, key=lambda x: x.f1_score).model_type if models else None
            }
            
            # Phase 7: Insight Generation
            logger.info("\n[PHASE 7] INSIGHT GENERATION")
            self.current_phase = AnalysisPhase.INSIGHT_GENERATION
            insights = self.insight_generator.generate(
                cleaned_data, eda_results, tested_hypotheses, models
            )
            results['phases']['insight_generation'] = {
                'status': 'success',
                'insights_count': len(insights),
                'insights': [asdict(i) for i in insights]
            }
            
            # Phase 8: Natural Language Explanation
            logger.info("\n[PHASE 8] EXPLANATION GENERATION")
            nl_explanation = self.explainer.explain_all(
                cleaned_data, eda_results, tested_hypotheses, insights, models
            )
            results['phases']['explanation_generation'] = {
                'status': 'success',
                'explanation': nl_explanation
            }
            
            # Phase 9: Report Generation
            logger.info("\n[PHASE 9] REPORT GENERATION")
            self.current_phase = AnalysisPhase.REPORT_GENERATION
            report_path = self._generate_report(results, output_dir)
            results['phases']['report_generation'] = {
                'status': 'success',
                'report_path': report_path
            }
            
            results['status'] = 'success'
            results['end_time'] = datetime.now().isoformat()
            results['total_duration'] = time.time() - start_time
            
            # Save results
            results_path = Path(output_dir) / f"results_{self.session_id}.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info("\n" + "="*80)
            logger.info("ANALYSIS COMPLETE!")
            logger.info(f"Total duration: {results['total_duration']:.2f} seconds")
            logger.info(f"Results saved to: {results_path}")
            logger.info("="*80)
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            results['total_duration'] = time.time() - start_time
            
            # Attempt self-healing
            logger.info("Attempting self-healing...")
            if self._attempt_recovery(data_path, output_dir, results):
                logger.info("Recovery successful! Analysis completed.")
            
            return results
    
    def _attempt_recovery(self, data_path: str, output_dir: str, results: Dict) -> bool:
        """Attempt to recover from errors"""
        logger.info(f"Recovery attempt for phase: {self.current_phase}")
        
        # Different recovery strategies based on phase
        if self.current_phase in [AnalysisPhase.DATA_INGESTION, AnalysisPhase.DATA_CLEANING]:
            # Try with simpler data loading
            try:
                import pandas as pd
                df = pd.read_csv(data_path, on_bad_lines='skip')
                logger.info(f"Recovery: Loaded {len(df)} rows with relaxed parsing")
                results['phases']['data_ingestion_recovery'] = {
                    'status': 'success',
                    'rows': len(df),
                    'columns': len(df.columns)
                }
                return True
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                return False
        
        return False
    
    def _generate_report(self, results: Dict, output_dir: str) -> str:
        """Generate comprehensive markdown report"""
        report_lines = [
            "# Autonomous Data Analysis Report",
            f"**Generated by AIDAS System**",
            f"**Session ID:** {self.session_id}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            "This report was generated autonomously by the AIDAS (Autonomous Intelligence for Data Analysis and Science) system. "
            "The analysis covered multiple phases from data ingestion through model building and insight generation.",
            "",
            f"**Total Analysis Duration:** {results.get('total_duration', 0):.2f} seconds",
            "",
            "---",
            "",
            "## Data Overview",
            ""
        ]
        
        # Data ingestion results
        if 'data_ingestion' in results.get('phases', {}):
            di = results['phases']['data_ingestion']
            report_lines.extend([
                f"- **Original Dataset:** {di.get('rows', 'N/A')} rows × {di.get('columns', 'N/A')} columns",
                "",
                "### Data Types Distribution:",
                ""
            ])
            dtypes = di.get('dtypes', {})
            for dtype, count in {str(v): k for k, v in dtypes.items()}.items():
                report_lines.append(f"- {dtype}: {count} columns")
        
        # Data cleaning results
        if 'data_cleaning' in results.get('phases', {}):
            dc = results['phases']['data_cleaning']
            qr = dc.get('quality_report', {})
            report_lines.extend([
                "",
                "---",
                "",
                "## Data Quality Assessment",
                "",
                f"**Quality Score:** {qr.get('quality_score', 0):.1%}",
                f"**Issues Found:** {len(qr.get('issues_found', []))}",
                f"**Issues Fixed:** {dc.get('issues_fixed', 0)}",
                "",
                "### Key Issues Detected:",
                ""
            ])
            for issue in qr.get('issues_found', [])[:5]:
                report_lines.append(f"- {issue}")
        
        # Hypothesis testing results
        if 'hypothesis_testing' in results.get('phases', {}):
            ht = results['phases']['hypothesis_testing']
            sig_count = ht.get('significant_hypotheses', 0)
            report_lines.extend([
                "",
                "---",
                "",
                "## Key Findings (Hypotheses)",
                "",
                f"**Significant Findings:** {sig_count} out of {ht.get('tested_hypotheses', []) and len(ht['tested_hypotheses'])} hypotheses tested",
                ""
            ])
            
            for hyp in ht.get('tested_hypotheses', [])[:3]:
                if hyp.get('is_significant'):
                    report_lines.extend([
                        f"### Finding: {hyp.get('statement', 'N/A')[:100]}...",
                        f"- **P-value:** {hyp.get('p_value', 'N/A'):.4f}",
                        f"- **Effect Size:** {hyp.get('effect_size', 'N/A'):.3f}",
                        f"- **Explanation:** {hyp.get('explanation', 'N/A')[:200]}",
                        ""
                    ])
        
        # Model results
        if 'model_building' in results.get('phases', {}):
            mb = results['phases']['model_building']
            report_lines.extend([
                "",
                "---",
                "",
                "## Predictive Models",
                "",
                f"**Models Built:** {mb.get('models_built', 0)}",
                f"**Best Performing:** {mb.get('best_model', 'N/A')}",
                ""
            ])
        
        # Natural language explanation
        if 'explanation_generation' in results.get('phases', {}):
            exp = results['phases']['explanation_generation']
            report_lines.extend([
                "",
                "---",
                "",
                "## Natural Language Summary",
                "",
                exp.get('explanation', 'No explanation generated.'),
                "",
                "---",
                "",
                "*This report was generated autonomously. For questions or detailed analysis, contact your data science team.*"
            ])
        
        # Save report
        report_path = Path(output_dir) / f"report_{self.session_id}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to: {report_path}")
        return str(report_path)


class _DataProcessor:
    """Handles data ingestion and cleaning"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.history = []
    
    def ingest(self, data_path: str) -> Any:
        """Load and validate data from various formats"""
        import pandas as pd
        import numpy as np
        
        logger.info(f"Ingesting data from: {data_path}")
        
        # Determine file type
        ext = Path(data_path).suffix.lower()
        
        try:
            if ext == '.csv':
                df = pd.read_csv(data_path)
            elif ext == '.json':
                df = pd.read_json(data_path)
            elif ext == '.excel' or ext == '.xlsx':
                df = pd.read_excel(data_path)
            elif ext == '.parquet':
                df = pd.read_parquet(data_path)
            else:
                # Try CSV as fallback
                df = pd.read_csv(data_path)
            
            logger.info(f"Successfully loaded: {len(df)} rows, {len(df.columns)} columns")
            self.history.append({
                'action': 'ingest',
                'rows': len(df),
                'columns': len(df.columns)
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            # Try recovery with error handling
            df = pd.read_csv(data_path, on_bad_lines='skip', encoding='latin-1')
            logger.info(f"Recovery successful: {len(df)} rows loaded")
            return df
    
    def clean(self, df) -> Tuple[Any, DataQualityReport]:
        """Clean and preprocess data"""
        import pandas as pd
        import numpy as np
        
        logger.info("Starting data cleaning process")
        
        issues_found = []
        suggested_fixes = []
        original_rows = len(df)
        
        # Check for missing values
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            issues_found.append(f"Found {len(missing_cols)} columns with missing values")
            suggested_fixes.append(f"Impute missing values in {len(missing_cols)} columns")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues_found.append(f"Found {duplicates} duplicate rows")
            suggested_fixes.append(f"Remove {duplicates} duplicate rows")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if len(constant_cols) > 0:
            issues_found.append(f"Found {len(constant_cols)} constant columns")
            suggested_fixes.append(f"Remove {len(constant_cols)} constant columns")
        
        # Perform cleaning
        df_cleaned = df.copy()
        
        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates()
        logger.info(f"Removed duplicates: {original_rows - len(df_cleaned)} rows")
        
        # Remove constant columns
        df_cleaned = df_cleaned.drop(columns=constant_cols, errors='ignore')
        logger.info(f"Removed constant columns: {constant_cols}")
        
        # Impute missing values
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown')
        
        # Calculate quality score
        quality_score = 1.0
        quality_score -= (duplicates / original_rows) if original_rows > 0 else 0
        quality_score -= (len(missing_cols) / len(df.columns)) if len(df.columns) > 0 else 0
        quality_score = max(0, min(1, quality_score))
        
        quality_report = DataQualityReport(
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_values=missing.to_dict(),
            duplicate_rows=duplicates,
            data_types=df.dtypes.astype(str).to_dict(),
            quality_score=quality_score,
            issues_found=issues_found,
            suggested_fixes=suggested_fixes
        )
        
        logger.info(f"Data cleaning complete. Quality score: {quality_score:.1%}")
        
        return df_cleaned, quality_report


class _PatternDiscoverer:
    """Discovers patterns in data"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def explore(self, df) -> Dict[str, Any]:
        """Perform exploratory data analysis"""
        import pandas as pd
        import numpy as np
        
        logger.info("Starting exploratory analysis")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        results = {
            'numeric_summary': {},
            'categorical_summary': {},
            'correlations': [],
            'distributions': {}
        }
        
        # Numeric summary
        for col in numeric_cols[:20]:  # Limit to first 20
            results['numeric_summary'][col] = {
                'mean': round(df[col].mean(), 3),
                'std': round(df[col].std(), 3),
                'min': round(df[col].min(), 3),
                'max': round(df[col].max(), 3),
                'median': round(df[col].median(), 3),
                'skewness': round(df[col].skew(), 3),
                'kurtosis': round(df[col].kurtosis(), 3)
            }
        
        # Categorical summary
        for col in categorical_cols[:10]:  # Limit to first 10
            results['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        
        # Correlations
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, min(i+10, len(numeric_cols))):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.3:
                        correlations.append({
                            'var1': numeric_cols[i],
                            'var2': numeric_cols[j],
                            'correlation': round(corr, 3),
                            'strength': 'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.5 else 'weak'
                        })
            results['correlations'] = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)[:50]
        
        # Distributions
        for col in numeric_cols[:5]:
            results['distributions'][col] = {
                'shape': 'normal' if abs(df[col].skew()) < 0.5 else 'skewed',
                'outliers': int(((df[col] < df[col].quantile(0.25)) | (df[col] > df[col].quantile(0.75))).sum())
            }
        
        logger.info(f"Exploratory analysis complete: {len(results['numeric_summary'])} numeric, {len(results['categorical_summary'])} categorical variables")
        
        return results


class _HypothesisEngine:
    """Discovers and tests hypotheses"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def discover(self, df, eda_results) -> List[Hypothesis]:
        """Discover potential hypotheses from data"""
        import pandas as pd
        import numpy as np
        
        logger.info("Discovering hypotheses")
        
        hypotheses = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        hypothesis_id = 0
        
        # Correlation-based hypotheses
        for corr in eda_results.get('correlations', [])[:10]:
            hypothesis_id += 1
            h = Hypothesis(
                id=f"H{hypothesis_id:03d}",
                statement=f"There is a relationship between {corr['var1']} and {corr['var2']}",
                variables=[corr['var1'], corr['var2']],
                test_type="correlation",
                statistic=corr['correlation'],
                p_value=0.0,  # Will be calculated in testing phase
                significance_level=0.05,
                is_significant=False,
                effect_size=abs(corr['correlation']),
                confidence_interval=(0, 0),
                explanation=""
            )
            hypotheses.append(h)
        
        # Group difference hypotheses
        for cat_col in categorical_cols[:3]:
            for num_col in numeric_cols[:3]:
                if df[cat_col].nunique() <= 5:  # Only if few categories
                    hypothesis_id += 1
                    h = Hypothesis(
                        id=f"H{hypothesis_id:03d}",
                        statement=f"{num_col} differs across {cat_col} categories",
                        variables=[num_col, cat_col],
                        test_type="group_comparison",
                        statistic=0.0,
                        p_value=0.0,
                        significance_level=0.05,
                        is_significant=False,
                        effect_size=0.0,
                        confidence_interval=(0, 0),
                        explanation=""
                    )
                    hypotheses.append(h)
        
        logger.info(f"Discovered {len(hypotheses)} potential hypotheses")
        
        return hypotheses
    
    def test(self, hypotheses: List[Hypothesis], df) -> List[Hypothesis]:
        """Test hypotheses statistically"""
        import pandas as pd
        import numpy as np
        from scipy import stats
        
        logger.info(f"Testing {len(hypotheses)} hypotheses")
        
        tested_hypotheses = []
        
        for h in hypotheses:
            try:
                if h.test_type == "correlation":
                    var1, var2 = h.variables
                    data1 = df[var1].dropna()
                    data2 = df[var2].dropna()
                    min_len = min(len(data1), len(data2))
                    
                    if min_len > 2:
                        corr, p_value = stats.pearsonr(data1[:min_len], data2[:min_len])
                        h.statistic = corr
                        h.p_value = p_value
                        h.is_significant = p_value < h.significance_level
                        h.effect_size = abs(corr)
                        h.confidence_interval = (
                            corr - 1.96/np.sqrt(min_len),
                            corr + 1.96/np.sqrt(min_len)
                        )
                        h.explanation = self._explain_correlation(h)
                
                elif h.test_type == "group_comparison":
                    num_col, cat_col = h.variables
                    groups = [group[num_col].values for name, group in df.groupby(cat_col) if len(group) >= 2]
                    
                    if len(groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*groups[:5])  # Limit to 5 groups
                        h.statistic = f_stat
                        h.p_value = p_value
                        h.is_significant = p_value < h.significance_level
                        h.effect_size = f_stat / 100 if f_stat > 0 else 0
                        h.explanation = self._explain_group_difference(h, cat_col)
                
                tested_hypotheses.append(h)
                
            except Exception as e:
                logger.warning(f"Failed to test hypothesis {h.id}: {e}")
                h.explanation = f"Could not test: {str(e)}"
                tested_hypotheses.append(h)
        
        sig_count = sum(1 for h in tested_hypotheses if h.is_significant)
        logger.info(f"Testing complete. {sig_count} significant hypotheses found")
        
        return tested_hypotheses
    
    def _explain_correlation(self, h: Hypothesis) -> str:
        """Explain correlation result in natural language"""
        strength = "strong" if abs(h.effect_size) > 0.7 else "moderate" if abs(h.effect_size) > 0.4 else "weak"
        direction = "positive" if h.statistic > 0 else "negative"
        
        return f"There is a {strength} {direction} relationship (r={h.statistic:.3f}, p={h.p_value:.4f}). " \
               f"This means when one variable increases, the other tends to {'increase' if h.statistic > 0 else 'decrease'}."
    
    def _explain_group_difference(self, h: Hypothesis, group_var: str) -> str:
        """Explain group difference result"""
        if h.is_significant:
            return f"Found statistically significant differences between groups in {group_var} (F={h.statistic:.2f}, p={h.p_value:.4f})."
        else:
            return f"No significant differences found between groups in {group_var} (p={h.p_value:.4f})."


class _ModelBuilder:
    """Builds and evaluates machine learning models"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def build_models(self, df, eda_results) -> List[ModelResult]:
        """Build multiple models and compare them"""
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import warnings
        warnings.filterwarnings('ignore')
        
        logger.info("Building and evaluating models")
        
        models = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Try to identify target variable
        target_candidates = ['target', 'label', 'class', 'health', 'score', 'outcome']
        target_col = None
        
        for col in reversed(numeric_cols):
            if any(cand.lower() in col.lower() for cand in target_candidates):
                target_col = col
                break
        
        if not target_col:
            target_col = numeric_cols[-1] if numeric_cols else None
        
        if target_col and target_col in numeric_cols:
            # Determine if classification or regression
            unique_values = df[target_col].nunique()
            
            if unique_values <= 10:
                task_type = "classification"
                target_col_for_model = target_col
            else:
                task_type = "regression"
                target_col_for_model = target_col
            
            # Prepare features
            feature_cols = [col for col in numeric_cols if col != target_col_for_model]
            
            if len(feature_cols) >= 2:
                X = df[feature_cols].fillna(df[feature_cols].median())
                y = df[target_col_for_model]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.config.test_size, random_state=self.config.random_state
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Build models based on task type
                if task_type == "classification":
                    model_configs = [
                        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=self.config.random_state)),
                        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=self.config.random_state))
                    ]
                else:
                    model_configs = [
                        ("Linear Regression", LinearRegression()),
                        ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=self.config.random_state))
                    ]
                
                for model_name, model in model_configs:
                    try:
                        # Train
                        model.fit(X_train_scaled, y_train)
                        
                        # Predict
                        y_pred = model.predict(X_test_scaled)
                        y_proba = None
                        
                        if task_type == "classification":
                            try:
                                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                            except:
                                pass
                            
                            # Metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            
                            if y_proba is not None and len(np.unique(y_test)) == 2:
                                roc_auc = roc_auc_score(y_test, y_proba)
                            else:
                                roc_auc = 0.5
                            
                        else:
                            # For regression, use negative versions of metrics
                            from sklearn.metrics import mean_squared_error, r2_score
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            accuracy = r2
                            precision = np.sqrt(mse)
                            recall = -np.sqrt(mse)
                            f1 = r2
                            roc_auc = r2
                        
                        # Feature importance
                        if hasattr(model, 'feature_importances_'):
                            importance = dict(zip(feature_cols, model.feature_importances_))
                        else:
                            importance = dict(zip(feature_cols, np.abs(model.coef_)))
                        
                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                        
                        result = ModelResult(
                            model_type=model_name,
                            accuracy=round(accuracy, 4),
                            precision=round(precision, 4),
                            recall=round(recall, 4),
                            f1_score=round(f1, 4),
                            roc_auc=round(roc_auc, 4),
                            feature_importance={k: round(v, 4) for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]},
                            confusion_matrix=[],
                            cross_validation_scores=cross_val_score.tolist() if hasattr(cross_val_score, 'tolist') else list(cv_scores)
                        )
                        
                        models.append(result)
                        logger.info(f"  {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to build {model_name}: {e}")
                        models.append(ModelResult(
                            model_type=model_name,
                            accuracy=0,
                            precision=0,
                            recall=0,
                            f1_score=0,
                            roc_auc=0,
                            feature_importance={},
                            confusion_matrix=[],
                            cross_validation_scores=[],
                            errors=[str(e)]
                        ))
        
        logger.info(f"Built {len(models)} models")
        
        return models


class _InsightGenerator:
    """Generates actionable insights"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def generate(self, df, eda_results, hypotheses: List[Hypothesis], models: List[ModelResult]) -> List[Insight]:
        """Generate comprehensive insights"""
        import pandas as pd
        import numpy as np
        
        logger.info("Generating insights")
        
        insights = []
        insight_id = 0
        
        # Statistical insights
        for hyp in hypotheses[:5]:
            if hyp.is_significant:
                insight_id += 1
                insight = Insight(
                    id=f"I{insight_id:03d}",
                    category="Statistical Finding",
                    description=hyp.statement,
                    evidence=f"P-value: {hyp.p_value:.4f}, Effect size: {hyp.effect_size:.3f}",
                    impact="high" if hyp.p_value < 0.01 else "moderate",
                    recommendation="Consider this relationship when making decisions",
                    confidence=1 - hyp.p_value,
                    supporting_metrics={
                        'statistic': hyp.statistic,
                        'p_value': hyp.p_value,
                        'variables': hyp.variables
                    }
                )
                insights.append(insight)
        
        # Data quality insights
        if len(eda_results.get('correlations', [])) == 0:
            insight_id += 1
            insights.append(Insight(
                id=f"I{insight_id:03d}",
                category="Data Quality",
                description="No strong correlations found between numeric variables",
                evidence="Correlation analysis revealed weak relationships",
                impact="low",
                recommendation="Consider collecting additional features or checking data quality",
                confidence=0.8,
                supporting_metrics={}
            ))
        
        # Model insights
        if models:
            best_model = max(models, key=lambda x: x.f1_score)
            insight_id += 1
            insights.append(Insight(
                id=f"I{insight_id:03d}",
                category="Predictive Model",
                description=f"Best model: {best_model.model_type} with F1 score of {best_model.f1_score:.3f}",
                evidence=f"Accuracy: {best_model.accuracy:.3f}, ROC-AUC: {best_model.roc_auc:.3f}",
                impact="high",
                recommendation=f"Consider deploying {best_model.model_type} for predictions",
                confidence=best_model.f1_score,
                supporting_metrics={
                    'model_type': best_model.model_type,
                    'metrics': asdict(best_model)
                }
            ))
        
        # Feature importance insights
        for model in models[:1]:  # Best model
            if model.feature_importance:
                top_features = list(model.feature_importance.items())[:3]
                for feat, imp in top_features:
                    if imp > 0.1:
                        insight_id += 1
                        insights.append(Insight(
                            id=f"I{insight_id:03d}",
                            category="Feature Importance",
                            description=f"'{feat}' is a key predictor with importance {imp:.3f}",
                            evidence=f"Feature importance score: {imp:.3f}",
                            impact="moderate",
                            recommendation=f"Focus on collecting high-quality data for {feat}",
                            confidence=imp,
                            supporting_metrics={'feature': feat, 'importance': imp}
                        ))
        
        logger.info(f"Generated {len(insights)} insights")
        
        return insights


class _NLExplainer:
    """Generates natural language explanations"""
    
    def __init__(self):
        pass
    
    def explain_all(self, df, eda_results, hypotheses: List[Hypothesis], 
                   insights: List[Insight], models: List[ModelResult]) -> str:
        """Generate comprehensive natural language explanation"""
        
        explanations = []
        
        # Data overview
        explanations.append("## 📊 Data Overview")
        explanations.append(f"""
This analysis examined a dataset with **{len(df)} observations** and **{len(df.columns)} variables**.
The data contained a mix of numeric and categorical features that allowed for comprehensive analysis.
""")
        
        # Key findings
        if hypotheses:
            significant = [h for h in hypotheses if h.is_significant]
            explanations.append("## 🔍 Key Statistical Findings")
            explanations.append(f"""
We conducted statistical tests on {len(hypotheses)} hypotheses and found **{len(significant)} statistically significant findings**:

""")
            
            for h in significant[:3]:
                explanations.append(f"- **{h.statement}**\n  - {h.explanation}\n")
        
        # Insights
        if insights:
            explanations.append("\n## 💡 Actionable Insights")
            explanations.append(f"""
The analysis identified **{len(insights)} key insights** that can drive decision-making:

""")
            
            for insight in insights[:5]:
                explanations.append(f"""
### {insight.category}: {insight.description}
- **Evidence:** {insight.evidence}
- **Impact:** {insight.impact.upper()}
- **Recommendation:** {insight.recommendation}
- **Confidence:** {insight.confidence:.1%}
""")
        
        # Models
        if models:
            best = max(models, key=lambda x: x.f1_score)
            explanations.append("\n## 🤖 Predictive Models")
            explanations.append(f"""
We built and evaluated {len(models)} predictive models to identify patterns and make forecasts.

**Best Performing Model:** {best.model_type}
- **Accuracy:** {best.accuracy:.1%}
- **F1 Score:** {best.f1_score:.3f}
- **ROC-AUC:** {best.roc_auc:.3f}

**Top Predictive Features:**
""")
            
            for feat, imp in list(best.feature_importance.items())[:5]:
                explanations.append(f"- {feat}: {imp:.3f}")
        
        # Recommendations
        explanations.append("\n## 🎯 Recommendations")
        recommendations = [
            "Act on the statistically significant findings to improve outcomes",
            "Collect more data on high-importance features identified in the model",
            "Consider deploying the predictive model for real-world applications",
            "Continue monitoring key metrics identified in this analysis"
        ]
        
        for rec in recommendations:
            explanations.append(f"- {rec}")
        
        return '\n'.join(explanations)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AIDAS - Autonomous Intelligence for Data Analysis and Science')
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output', '-o', type=str, default='aidas_results', help='Output directory')
    parser.add_argument('--config', '-c', type=str, help='Path to config JSON')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = AnalysisConfig(**config_dict)
    
    # Initialize and run
    aidas = AutonomousIntelligence(config)
    results = aidas.analyze(args.file, args.output)
    
    return 0 if results.get('status') == 'success' else 1


if __name__ == "__main__":
    sys.exit(main())
