"""
MCP Integration Layer - Updated with Autonomous Features
Provides unified access to all MCP servers with autonomous capabilities
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from .autonomous_mcp_agent import (
    AutonomousMCPServer,
    AutonomousWorkflowExecutor,
    AutonomousDataScienceAgent
)
from .environment_config import get_config
from .self_healing_executor import SelfHealingExecutor, RecoveryConfig


class MCPAutonomousIntegration:
    """
    Enhanced MCP Integration with autonomous capabilities
    
    Features:
    - Automatic server discovery and connection
    - Self-healing tool execution
    - End-to-end workflow automation
    - Environment-aware configuration
    """
    
    def __init__(self):
        self.config = get_config()
        self.mcp_server = AutonomousMCPServer()
        self.executor = AutonomousWorkflowExecutor(self.mcp_server)
        self.agent = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all MCP components"""
        if self._initialized:
            return True
        
        print("🔌 Initializing Autonomous MCP Integration...")
        
        # Initialize the agent
        self.agent = AutonomousDataScienceAgent()
        success = await self.agent.initialize()
        
        self._initialized = success
        return success
    
    async def analyze_dataset(
        self,
        dataset_path: str,
        analysis_goal: str = "comprehensive analysis"
    ) -> Dict[str, Any]:
        """
        Autonomously analyze a dataset end-to-end
        
        Args:
            dataset_path: Path to dataset file
            analysis_goal: Description of analysis goal
            
        Returns:
            Complete analysis results
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.agent.analyze_dataset_autonomous(dataset_path, analysis_goal)
    
    async def execute_autonomous_workflow(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute a custom autonomous workflow
        
        Args:
            tasks: List of task specifications
            
        Returns:
            Workflow execution results
        """
        from .autonomous_mcp_agent import AutonomousTask, TaskStatus
        
        # Convert to AutonomousTask objects
        workflow_tasks = []
        for i, task_spec in enumerate(tasks):
            task = AutonomousTask(
                task_id=task_spec.get('id', f"task_{i}"),
                description=task_spec.get('description', ''),
                tool_name=task_spec.get('tool', ''),
                parameters=task_spec.get('parameters', {}),
                dependencies=task_spec.get('dependencies', []),
                max_retries=task_spec.get('max_retries', 3)
            )
            workflow_tasks.append(task)
        
        # Execute workflow
        results = await self.executor.execute_workflow(workflow_tasks, {})
        
        return {
            'total_tasks': len(workflow_tasks),
            'completed': sum(1 for r in results.values() if r.success),
            'failed': sum(1 for r in results.values() if not r.success),
            'results': {k: v.output if v.success else {'error': v.error} 
                       for k, v in results.items()}
        }
    
    def execute_tool_with_healing(
        self,
        tool_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a tool function with self-healing
        
        Args:
            tool_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises exception
        """
        executor = SelfHealingExecutor(RecoveryConfig(
            max_attempts=self.config.max_retries,
            retry_delay=self.config.retry_delay
        ))
        
        result = executor.execute(tool_func, *args, **kwargs)
        
        if result.success:
            return result.result
        else:
            raise RuntimeError(f"Tool execution failed: {result.error}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            'initialized': self._initialized,
            'mcp_servers': self.mcp_server.health_status if self.mcp_server else {},
            'agent_status': self.agent.get_status() if self.agent else None,
            'config': self.config.to_dict()
        }


# Singleton instance
_autonomous_integration = None

async def get_autonomous_mcp() -> MCPAutonomousIntegration:
    """Get or create autonomous MCP integration"""
    global _autonomous_integration
    if _autonomous_integration is None:
        _autonomous_integration = MCPAutonomousIntegration()
        await _autonomous_integration.initialize()
    return _autonomous_integration


# Convenience function for quick analysis
async def autonomous_data_analysis(
    dataset_path: str,
    analysis_goal: str = "full_analysis"
) -> Dict[str, Any]:
    """
    Quick autonomous data analysis
    
    Usage:
        results = await autonomous_data_analysis("data.csv", "analyze customer churn")
    """
    integration = await get_autonomous_mcp()
    return await integration.analyze_dataset(dataset_path, analysis_goal)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous MCP Data Science")
    parser.add_argument("dataset", help="Path to dataset file")
    parser.add_argument("--goal", default="full analysis", help="Analysis goal")
    
    args = parser.parse_args()
    
    # Run autonomous analysis
    results = asyncio.run(autonomous_data_analysis(args.dataset, args.goal))
    print(json.dumps(results, indent=2, default=str))
