"""
Autonomous MCP Orchestrator - End-to-End Autonomous Tool Execution
Inspired by GitHub Copilot's autonomous agent capabilities

This module provides:
1. Autonomous tool discovery and execution
2. Self-healing workflow with automatic recovery
3. End-to-end task completion without manual intervention
4. Dynamic server management and health monitoring
"""

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of autonomous task execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_name: str = ""
    retry_count: int = 0


@dataclass
class AutonomousTask:
    """Task to be executed autonomously"""
    task_id: str
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    max_retries: int = 3
    retry_delay: float = 1.0
    on_success: Optional[Callable] = None
    on_failure: Optional[Callable] = None


class AutonomousMCPServer:
    """
    Autonomous MCP Server Manager
    Automatically discovers, starts, and manages MCP servers
    """
    
    def __init__(self):
        self.servers: Dict[str, Dict] = {}
        self.tool_registry: Dict[str, Dict] = {}
        self.health_status: Dict[str, bool] = {}
        self.startup_attempts: Dict[str, int] = {}
        self.max_startup_attempts = 3
        
    def discover_servers(self) -> List[Dict[str, Any]]:
        """Automatically discover available MCP servers"""
        discovered = []
        
        # Check for local Pandas MCP server
        pandas_path = Path(__file__).parent / "pandas_mcp_server.py"
        if pandas_path.exists():
            discovered.append({
                'name': 'pandas',
                'type': 'local',
                'path': str(pandas_path),
                'protocol': 'stdio',
                'auto_start': True
            })
        
        # Check for Jupyter MCP server endpoint
        jupyter_endpoints = [
            "http://localhost:8888/mcp",
            "http://127.0.0.1:8888/mcp",
        ]
        for endpoint in jupyter_endpoints:
            discovered.append({
                'name': 'jupyter',
                'type': 'remote',
                'url': endpoint,
                'protocol': 'sse',
                'auto_start': False,
                'start_command': None
            })
        
        # Check for Docker MCP Toolkit
        try:
            result = subprocess.run(
                ["docker", "mcp", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                discovered.append({
                    'name': 'docker',
                    'type': 'gateway',
                    'protocol': 'sse',
                    'auto_start': False,
                    'start_command': ["docker", "mcp", "gateway", "run"]
                })
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError, OSError):
            logger.info("Docker MCP Toolkit not available (not installed or no permission)")
        
        return discovered
    
    async def start_server(self, server_config: Dict[str, Any]) -> bool:
        """Automatically start an MCP server"""
        server_name = server_config['name']
        
        if server_name in self.servers and self.health_status.get(server_name, False):
            return True
        
        self.startup_attempts[server_name] = self.startup_attempts.get(server_name, 0) + 1
        
        if self.startup_attempts[server_name] > self.max_startup_attempts:
            logger.error(f"Max startup attempts reached for {server_name}")
            return False
        
        try:
            if server_config['type'] == 'local':
                # Local server - import directly
                from .pandas_mcp_server import PandasMCPServer
                self.servers[server_name] = {
                    'instance': PandasMCPServer(),
                    'config': server_config
                }
                self.health_status[server_name] = True
                logger.info(f"✅ Started local MCP server: {server_name}")
                return True
            
            elif server_config['type'] == 'remote':
                # Remote server - check health
                health_url = f"{server_config['url']}/healthz"
                import httpx
                try:
                    response = await asyncio.wait_for(
                        httpx.get(health_url, timeout=3),
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        self.servers[server_name] = {
                            'url': server_config['url'],
                            'config': server_config
                        }
                        self.health_status[server_name] = True
                        logger.info(f"✅ Connected to remote MCP server: {server_name}")
                        return True
                except Exception as e:
                    logger.warning(f"Remote server {server_name} not available: {e}")
                    self.health_status[server_name] = False
            
            elif server_config['type'] == 'gateway':
                # Docker MCP Gateway
                # Assume it's available via default port
                self.servers[server_name] = {
                    'config': server_config,
                    'default_port': 12307
                }
                self.health_status[server_name] = True
                logger.info(f"✅ Docker MCP Gateway configured: {server_name}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to start server {server_name}: {e}")
            self.health_status[server_name] = False
        
        return False
    
    async def discover_and_start_all(self) -> Dict[str, bool]:
        """Autonomously discover and start all available servers"""
        results = {}
        servers = self.discover_servers()
        
        for server_config in servers:
            if server_config.get('auto_start', False):
                success = await self.start_server(server_config)
                results[server_config['name']] = success
            else:
                results[server_config['name']] = self.health_status.get(server_config['name'], False)
        
        return results
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get all available tools from all servers"""
        tools = {}
        
        # Pandas MCP tools
        if 'pandas' in self.servers:
            instance = self.servers['pandas'].get('instance')
            if instance:
                pandas_tools = [
                    method for method in dir(instance)
                    if not method.startswith('_') and callable(getattr(instance, method))
                ]
                tools['pandas'] = pandas_tools
        
        # Add Docker MCP known tools
        if 'docker' in self.servers:
            tools['docker'] = [
                'playwright_navigate', 'playwright_screenshot', 'playwright_click',
                'playwright_fill', 'playwright_evaluate', 'playwright_goto',
                'playwright_wait', 'playwright_scrape', 'context7_save', 'context7_load',
                'mcp-find', 'mcp-add', 'mcp-remove', 'mcp-exec', 'mcp-config-set'
            ]
        
        return tools


class AutonomousWorkflowExecutor:
    """
    End-to-End Autonomous Workflow Executor
    Executes multi-step workflows autonomously with self-healing capabilities
    """
    
    def __init__(self, mcp_server: AutonomousMCPServer = None):
        self.mcp_server = mcp_server or AutonomousMCPServer()
        self.task_history: List[AutonomousTask] = []
        self.results_cache: Dict[str, ToolResult] = {}
        self.execution_log: List[Dict] = []
        
    async def execute_workflow(
        self,
        tasks: List[AutonomousTask],
        context: Dict[str, Any] = None
    ) -> Dict[str, ToolResult]:
        """
        Execute a workflow of tasks autonomously
        
        Args:
            tasks: List of tasks to execute
            context: Shared context between tasks
            
        Returns:
            Dictionary mapping task_id to result
        """
        results = {}
        context = context or {}
        
        logger.info(f"🚀 Starting autonomous workflow with {len(tasks)} tasks")
        
        for task in tasks:
            try:
                # Check dependencies
                if task.dependencies:
                    unmet = [d for d in task.dependencies if d not in results or not results[d].success]
                    if unmet:
                        logger.warning(f"Task {task.task_id} skipping due to unmet dependencies: {unmet}")
                        task.status = TaskStatus.SKIPPED
                        continue
                
                # Execute task with retry
                result = await self._execute_with_retry(task, context)
                results[task.task_id] = result
                
                # Update context with result
                if result.success:
                    context[f"{task.task_id}_result"] = result.output
                    task.status = TaskStatus.COMPLETED
                    if task.on_success:
                        task.on_success(result)
                else:
                    task.status = TaskStatus.FAILED
                    if task.on_failure:
                        task.on_failure(result)
                
                # Log execution
                self.execution_log.append({
                    'task_id': task.task_id,
                    'status': task.status.value,
                    'duration': result.duration_seconds,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed with exception: {e}")
                results[task.task_id] = ToolResult(
                    success=False,
                    error=str(e),
                    tool_name=task.tool_name
                )
                task.status = TaskStatus.FAILED
        
        self.task_history.extend(tasks)
        return results
    
    async def _execute_with_retry(
        self,
        task: AutonomousTask,
        context: Dict[str, Any]
    ) -> ToolResult:
        """Execute a single task with automatic retry"""
        last_error = None
        
        for attempt in range(task.max_retries + 1):
            try:
                result = await self._execute_single_task(task, context)
                result.retry_count = attempt
                
                if result.success:
                    return result
                else:
                    last_error = result.error
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Task {task.task_id} attempt {attempt + 1} failed: {e}")
            
            # Wait before retry
            if attempt < task.max_retries:
                await asyncio.sleep(task.retry_delay * (attempt + 1))
        
        return ToolResult(
            success=False,
            error=f"Max retries exceeded. Last error: {last_error}",
            tool_name=task.tool_name,
            retry_count=task.max_retries
        )
    
    async def _execute_single_task(
        self,
        task: AutonomousTask,
        context: Dict[str, Any]
    ) -> ToolResult:
        """Execute a single task"""
        start_time = time.time()
        
        # Get server instance
        server_name = task.tool_name.split('.')[0] if '.' in task.tool_name else 'pandas'
        tool_name = task.tool_name.split('.')[-1] if '.' in task.tool_name else task.tool_name
        
        if server_name not in self.mcp_server.servers:
            # Try to start server
            servers = self.mcp_server.discover_servers()
            for server_config in servers:
                if server_config['name'] == server_name:
                    await self.mcp_server.start_server(server_config)
                    break
        
        if server_name not in self.mcp_server.servers:
            return ToolResult(
                success=False,
                error=f"Server {server_name} not available",
                tool_name=task.tool_name
            )
        
        # Get tool and execute
        server = self.mcp_server.servers[server_name]
        instance = server.get('instance')
        
        if instance and hasattr(instance, tool_name):
            tool = getattr(instance, tool_name)
            try:
                # Merge task parameters with context
                params = task.parameters.copy()
                for key, value in params.items():
                    if isinstance(value, str) and value.startswith('${'):
                        # Replace with context value
                        context_key = value[2:-1]
                        if context_key in context:
                            params[key] = context[context_key]
                
                if asyncio.iscoroutinefunction(tool):
                    output = await tool(**params)
                else:
                    output = tool(**params)
                
                duration = time.time() - start_time
                
                return ToolResult(
                    success=True,
                    output=output,
                    duration_seconds=duration,
                    tool_name=task.tool_name
                )
                
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e),
                    duration_seconds=time.time() - start_time,
                    tool_name=task.tool_name
                )
        else:
            return ToolResult(
                success=False,
                error=f"Tool {tool_name} not found on server {server_name}",
                tool_name=task.tool_name
            )


class AutonomousDataScienceAgent:
    """
    Autonomous Data Science Agent - End-to-End like GitHub Copilot
    
    This agent can:
    1. Autonomously analyze datasets
    2. Discover and use appropriate tools
    3. Handle errors automatically
    4. Complete multi-step workflows without manual intervention
    """
    
    def __init__(self):
        self.mcp_server = AutonomousMCPServer()
        self.executor = AutonomousWorkflowExecutor(self.mcp_server)
        self.context: Dict[str, Any] = {}
        
    async def initialize(self) -> bool:
        """Autonomously initialize all components"""
        logger.info("🤖 Initializing Autonomous Data Science Agent...")
        
        # Discover and start all MCP servers
        server_status = await self.mcp_server.discover_and_start_all()
        
        for server, status in server_status.items():
            if status:
                logger.info(f"   ✅ {server} MCP server ready")
            else:
                logger.warning(f"   ⚠️ {server} MCP server not available")
        
        # Get available tools
        tools = self.mcp_server.get_available_tools()
        total_tools = sum(len(t) for t in tools.values())
        logger.info(f"   📊 Total autonomous tools: {total_tools}")
        
        return any(server_status.values())
    
    async def analyze_dataset_autonomous(
        self,
        dataset_path: str,
        analysis_goal: str = "full_analysis"
    ) -> Dict[str, Any]:
        """
        Autonomously analyze a dataset end-to-end
        
        Args:
            dataset_path: Path to dataset file
            analysis_goal: Description of analysis goal
            
        Returns:
            Complete analysis results
        """
        logger.info(f"📊 Starting autonomous analysis: {dataset_path}")
        
        # Create autonomous workflow
        tasks = self._create_analysis_workflow(dataset_path, analysis_goal)
        
        # Execute workflow autonomously
        results = await self.executor.execute_workflow(tasks, self.context)
        
        # Compile results
        analysis_results = {
            'dataset_path': dataset_path,
            'analysis_goal': analysis_goal,
            'timestamp': datetime.now().isoformat(),
            'task_results': {k: v.output if v.success else {'error': v.error} 
                           for k, v in results.items()},
            'success_rate': sum(1 for r in results.values() if r.success) / len(results),
            'total_duration': sum(r.duration_seconds for r in results.values())
        }
        
        # Extract key insights
        if 'read_data' in results and results['read_data'].success:
            data_info = results['read_data'].output
            analysis_results['dataset_info'] = data_info
        
        if 'explore_data' in results and results['explore_data'].success:
            analysis_results['exploration'] = results['explore_data'].output
        
        if 'analyze_data' in results and results['analyze_data'].success:
            analysis_results['analysis'] = results['analyze_data'].output
        
        logger.info(f"✅ Autonomous analysis complete. Success rate: {analysis_results['success_rate']:.1%}")
        
        return analysis_results
    
    def _create_analysis_workflow(
        self,
        dataset_path: str,
        analysis_goal: str
    ) -> List[AutonomousTask]:
        """Create autonomous workflow for dataset analysis"""
        tasks = []
        
        # Task 1: Read data
        tasks.append(AutonomousTask(
            task_id="read_data",
            description="Load dataset from file",
            tool_name="pandas.read_csv",
            parameters={
                "filepath": dataset_path,
                "name": "dataset"
            }
        ))
        
        # Task 2: Explore data
        tasks.append(AutonomousTask(
            task_id="explore_data",
            description="Get dataset information and statistics",
            tool_name="pandas.info",
            parameters={
                "name": "dataset"
            }
        ))
        
        # Task 3: Describe data
        tasks.append(AutonomousTask(
            task_id="describe_data",
            description="Get statistical summary",
            tool_name="pandas.describe",
            parameters={
                "name": "dataset"
            }
        ))
        
        # Task 4: Check for correlations (if numeric columns exist)
        tasks.append(AutonomousTask(
            task_id="correlation_analysis",
            description="Calculate correlation matrix",
            tool_name="pandas.correlation",
            parameters={
                "name": "dataset"
            },
            dependencies=["explore_data"]
        ))
        
        # Task 5: Find outliers
        tasks.append(AutonomousTask(
            task_id="outlier_detection",
            description="Detect outliers in data",
            tool_name="pandas.find_outliers",
            parameters={
                "name": "dataset",
                "column": "${first_numeric_column}",
                "method": "iqr"
            },
            dependencies=["explore_data"]
        ))
        
        return tasks
    
    async def run_autonomous_task(
        self,
        task_description: str,
        tool_selection: str = "auto",
        parameters: Dict[str, Any] = None
    ) -> ToolResult:
        """
        Run a single autonomous task
        
        Args:
            task_description: Natural language description of task
            tool_selection: "auto" or specific tool name
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        # Auto-select appropriate tool
        if tool_selection == "auto":
            tool_name = self._select_tool_for_task(task_description)
        else:
            tool_name = tool_selection
        
        # Create and execute task
        task = AutonomousTask(
            task_id=f"ad_hoc_{int(time.time())}",
            description=task_description,
            tool_name=tool_name,
            parameters=parameters or {}
        )
        
        result = await self.executor._execute_with_retry(task, {})
        return result
    
    def _select_tool_for_task(self, task_description: str) -> str:
        """Select appropriate tool based on task description"""
        task_lower = task_description.lower()
        
        if "read" in task_lower and "csv" in task_lower:
            return "pandas.read_csv"
        elif "info" in task_lower or "structure" in task_lower:
            return "pandas.info"
        elif "describe" in task_lower or "statistics" in task_lower:
            return "pandas.describe"
        elif "correlation" in task_lower:
            return "pandas.correlation"
        elif "outlier" in task_lower:
            return "pandas.find_outliers"
        elif "trend" in task_lower:
            return "pandas.find_trends"
        elif "filter" in task_lower:
            return "pandas.query"
        elif "group" in task_lower:
            return "pandas.groupby"
        else:
            return "pandas.read_csv"  # Default
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'mcp_servers': self.mcp_server.health_status,
            'available_tools': self.mcp_server.get_available_tools(),
            'pending_tasks': len([t for t in self.executor.task_history 
                                 if t.status == TaskStatus.PENDING]),
            'completed_tasks': len([t for t in self.executor.task_history 
                                   if t.status == TaskStatus.COMPLETED]),
            'failed_tasks': len([t for t in self.executor.task_history 
                                if t.status == TaskStatus.FAILED])
        }


# Convenience function for quick autonomous analysis
async def autonomous_analysis(
    dataset_path: str,
    analysis_goal: str = "full_analysis"
) -> Dict[str, Any]:
    """
    Quick autonomous data analysis
    
    Usage:
        results = await autonomous_analysis("data.csv", "analyze customer behavior")
    """
    agent = AutonomousDataScienceAgent()
    await agent.initialize()
    return await agent.analyze_dataset_autonomous(dataset_path, analysis_goal)


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous MCP Data Science Agent")
    parser.add_argument("dataset", help="Path to dataset file")
    parser.add_argument("--goal", default="full_analysis", help="Analysis goal")
    
    args = parser.parse_args()
    
    # Run autonomous analysis
    results = asyncio.run(autonomous_analysis(args.dataset, args.goal))
    print(json.dumps(results, indent=2, default=str))
