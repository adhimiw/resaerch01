"""
AGI State Model

Defines the state object that flows through the LangGraph state machine.
Contains all data needed for autonomous analysis.
"""

from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass


class AGIState(TypedDict):
    """
    State object for AGI autonomous analysis workflow.
    Passed between LangGraph nodes.
    """
    
    # ===== INPUT =====
    dataset_path: str
    user_objectives: Optional[List[str]]
    
    # ===== DATASET UNDERSTANDING =====
    dataset_profile: Dict[str, Any]  # Shape, dtypes, summary stats
    domain_knowledge: Dict[str, Any]  # From browser research
    
    # ===== GENERATION =====
    hypotheses: List[Dict[str, Any]]  # Generated hypotheses
    analysis_plan: Dict[str, Any]  # Execution plan
    generated_code: str  # Python code to execute
    notebook_cells: List[Dict[str, Any]]  # Jupyter cells
    
    # ===== VERIFICATION =====
    verification_results: Dict[str, Any]  # Results from 5 layers
    confidence_score: float  # 0-100
    issues_found: List[str]  # Problems detected
    
    # ===== COMPARISON =====
    methodology_results: Dict[str, Any]  # Results from each method
    best_method: str  # Recommended approach
    comparison_report: Dict[str, Any]  # Detailed comparison
    
    # ===== LEARNING =====
    successful_patterns: List[Dict[str, Any]]  # Patterns to store
    kappa: float  # Self-improvement coefficient
    
    # ===== CONTROL =====
    attempts: int  # Current attempt number
    max_attempts: int  # Maximum retry attempts
    is_verified: bool  # Passed verification?
    analysis_id: str  # Unique analysis ID
    current_phase: str  # Which phase we're in
    
    # ===== OUTPUT =====
    final_notebook: str  # Path to .ipynb file
    insights: List[str]  # Key insights discovered
    recommendations: List[str]  # Actionable recommendations
    langfuse_trace_url: str  # Link to trace
    timestamp: str  # ISO format timestamp


@dataclass
class DatasetProfile:
    """Profile of a dataset"""
    rows: int
    columns: int
    dtypes: Dict[str, str]
    data_type: str  # timeseries|tabular|text|mixed
    domain: str  # healthcare|finance|retail|etc
    task_type: str  # classification|regression|clustering
    key_columns: List[str]
    missing_values: Dict[str, int]
    duplicates: int
    quality_score: float
    

@dataclass
class Hypothesis:
    """A testable hypothesis"""
    id: str
    statement: str
    test_strategy: str
    expected_outcome: str
    priority: int  # 1-10
    status: str  # pending|testing|confirmed|rejected


@dataclass
class VerificationResult:
    """Result from verification engine"""
    passed: bool
    confidence: float  # 0-100
    layer_results: Dict[str, Any]
    issues: List[str]
    suggestions: List[str]
    timestamp: datetime


@dataclass
class MethodologyResult:
    """Result from a single methodology"""
    method_name: str
    metrics: Dict[str, float]
    execution_time: float
    code: str
    notebook_cells: List[str]
    

@dataclass
class ComparisonReport:
    """Comparison of multiple methodologies"""
    methods_compared: List[str]
    metric_comparison: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Any]
    best_method: str
    recommendation: str
    trade_offs: Dict[str, str]


@dataclass
class Insight:
    """A discovered insight"""
    text: str
    confidence: float
    supporting_evidence: List[str]
    related_hypotheses: List[str]
    

@dataclass
class AnalysisResult:
    """Complete analysis result"""
    analysis_id: str
    dataset_path: str
    profile: DatasetProfile
    hypotheses: List[Hypothesis]
    verification: VerificationResult
    comparison: ComparisonReport
    insights: List[Insight]
    recommendations: List[str]
    notebook_path: str
    langfuse_trace_url: str
    kappa: float
    timestamp: datetime


def create_initial_state(
    dataset_path: str,
    user_objectives: Optional[List[str]] = None,
    max_attempts: int = 3
) -> AGIState:
    """
    Create initial state for AGI analysis
    
    Args:
        dataset_path: Path to dataset file
        user_objectives: Optional list of user goals
        max_attempts: Maximum verification retry attempts
        
    Returns:
        Initial AGIState
    """
    import uuid
    
    return AGIState(
        # Input
        dataset_path=dataset_path,
        user_objectives=user_objectives or [],
        
        # Dataset Understanding
        dataset_profile={},
        domain_knowledge={},
        
        # Generation
        hypotheses=[],
        analysis_plan={},
        generated_code="",
        notebook_cells=[],
        
        # Verification
        verification_results={},
        confidence_score=0.0,
        issues_found=[],
        
        # Comparison
        methodology_results={},
        best_method="",
        comparison_report={},
        
        # Learning
        successful_patterns=[],
        kappa=0.0,
        
        # Control
        attempts=0,
        max_attempts=max_attempts,
        is_verified=False,
        analysis_id=str(uuid.uuid4()),
        current_phase="initialization",
        
        # Output
        final_notebook="",
        insights=[],
        recommendations=[],
        langfuse_trace_url="",
        timestamp=datetime.now().isoformat()
    )


def validate_state(state: AGIState) -> bool:
    """
    Validate state has required fields
    
    Args:
        state: AGIState to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError if invalid
    """
    required_fields = [
        "dataset_path",
        "analysis_id",
        "attempts",
        "max_attempts"
    ]
    
    for field in required_fields:
        if field not in state:
            raise ValueError(f"Missing required field: {field}")
    
    if state["attempts"] > state["max_attempts"]:
        raise ValueError("Attempts exceeded max_attempts")
    
    return True
