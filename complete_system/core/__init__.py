"""
Complete System Core Module
Autonomous MCP Data Science System
"""

# Core components
from .autonomous_mcp_agent import (
    AutonomousMCPServer,
    AutonomousWorkflowExecutor,
    AutonomousDataScienceAgent,
    AutonomousTask,
    ToolResult,
    TaskStatus,
    autonomous_analysis
)

from .environment_config import (
    EnvironmentManager,
    EnvironmentConfig,
    get_config,
    get_env_manager
)

from .self_healing_executor import (
    SelfHealingExecutor,
    RecoveryConfig,
    RecoveryStrategy,
    ExecutionAttempt,
    FallbackChain,
    RobustMCPTool,
    with_self_healing
)

from .mcp_integration_autonomous import (
    MCPAutonomousIntegration,
    get_autonomous_mcp,
    autonomous_data_analysis
)

from .mistral_autonomous_executor import (
    MistralAutonomousExecutor,
    MistralAPIClient,
    PlanningEngine,
    CodeGenerationEngine,
    ExecutionEngine,
    IterationEngine,
    ExecutionResult,
    ExecutionPlan,
    CodeArtifact,
    ExecutionPhase,
    TaskComplexity,
    autonomous_execute
)

from .ml_autonomous_executor import (
    MLAutonomousExecutor,
    MLCodeGenerator,
    MLDatasetAnalyzer,
    MLModelTrainer,
    MLTaskType,
    ModelType,
    EvaluationMetric,
    DatasetInfo,
    ModelResult,
    ComparisonResult,
    print_comparison_result
)

__all__ = [
    # Autonomous Agent
    'AutonomousMCPServer',
    'AutonomousWorkflowExecutor',
    'AutonomousDataScienceAgent',
    'AutonomousTask',
    'ToolResult',
    'TaskStatus',
    'autonomous_analysis',
    
    # Environment
    'EnvironmentManager',
    'EnvironmentConfig',
    'get_config',
    'get_env_manager',
    
    # Self-Healing
    'SelfHealingExecutor',
    'RecoveryConfig',
    'RecoveryStrategy',
    'ExecutionAttempt',
    'FallbackChain',
    'RobustMCPTool',
    'with_self_healing',
    
    # Integration
    'MCPAutonomousIntegration',
    'get_autonomous_mcp',
    'autonomous_data_analysis',
    
    # Mistral Autonomous Executor
    'MistralAutonomousExecutor',
    'MistralAPIClient',
    'PlanningEngine',
    'CodeGenerationEngine',
    'ExecutionEngine',
    'IterationEngine',
    'ExecutionResult',
    'ExecutionPlan',
    'CodeArtifact',
    'ExecutionPhase',
    'TaskComplexity',
    'autonomous_execute',
    
    # ML Autonomous Executor
    'MLAutonomousExecutor',
    'MLCodeGenerator',
    'MLDatasetAnalyzer',
    'MLModelTrainer',
    'MLTaskType',
    'ModelType',
    'EvaluationMetric',
    'DatasetInfo',
    'ModelResult',
    'ComparisonResult',
    'print_comparison_result'
]

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
