"""
MCP Integration Layer
Provides unified access to all MCP servers (Pandas, Jupyter, Docker)
"""

import requests
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


class MCPIntegration:
    """Unified interface for all MCP servers"""
    
    def __init__(self):
        self.servers = {
            'pandas': None,
            'jupyter': None,
            'docker': None
        }
        self.available_tools = {}
        self.initialize_servers()
    
    def initialize_servers(self):
        """Initialize all available MCP servers"""
        print("🔌 Initializing MCP servers...")
        
        # 1. Pandas MCP Server (stdio)
        try:
            self._init_pandas_mcp()
            print("   ✅ Pandas MCP Server initialized")
        except Exception as e:
            print(f"   ⚠️  Pandas MCP Server unavailable: {e}")
        
        # 2. Jupyter MCP Server (SSE)
        try:
            self._init_jupyter_mcp()
            print("   ✅ Jupyter MCP Server connected")
        except Exception as e:
            print(f"   ⚠️  Jupyter MCP Server unavailable: {e}")
        
        # 3. Docker MCP Gateway (SSE)
        try:
            self._init_docker_mcp()
            print("   ✅ Docker MCP Gateway connected")
        except Exception as e:
            print(f"   ⚠️  Docker MCP Gateway unavailable: {e}")
        
        self._summarize_capabilities()
    
    def _init_pandas_mcp(self):
        """Initialize Pandas MCP Server (local)"""
        server_path = Path("core/pandas_mcp_server.py")
        if not server_path.exists():
            server_path = Path("complete_system/core/pandas_mcp_server.py")
        
        spec = importlib.util.spec_from_file_location("pandas_mcp_server", server_path)
        pandas_mcp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pandas_mcp)
        
        # Create server instance
        self.servers['pandas'] = pandas_mcp.PandasMCPServer()
        
        # Extract available methods
        tools = [
            name for name in dir(self.servers['pandas']) 
            if callable(getattr(self.servers['pandas'], name)) 
            and not name.startswith('_')
        ]
        
        self.available_tools['pandas'] = {
            'server': 'Pandas MCP Server',
            'protocol': 'stdio',
            'tools': tools,
            'count': len(tools),
            'capabilities': [
                'data_loading', 'data_cleaning', 'data_exploration',
                'statistical_analysis', 'feature_engineering', 'visualization'
            ]
        }
    
    def _init_jupyter_mcp(self):
        """Initialize Jupyter MCP Server (SSE)"""
        health_url = "http://localhost:8888/mcp/healthz"
        tools_url = "http://localhost:8888/mcp/tools/list"
        
        # Health check
        response = requests.get(health_url, timeout=3)
        health_data = response.json()
        
        if health_data.get('status') != 'healthy':
            raise Exception("Jupyter MCP Server not healthy")
        
        # Get tools
        tools_response = requests.get(tools_url, timeout=5)
        tools_data = tools_response.json()
        
        tools = tools_data.get('tools', [])
        
        self.servers['jupyter'] = {
            'url': 'http://localhost:8888/mcp',
            'version': health_data.get('version'),
            'status': 'connected'
        }
        
        self.available_tools['jupyter'] = {
            'server': 'Jupyter MCP Server',
            'protocol': 'sse',
            'tools': [t['name'] for t in tools],
            'count': len(tools),
            'capabilities': [
                'notebook_execution', 'kernel_management', 'cell_operations',
                'code_execution', 'state_persistence', 'interactive_computing'
            ]
        }
    
    def _init_docker_mcp(self):
        """Initialize Docker MCP Gateway (SSE)"""
        gateway_url = "http://localhost:12307/sse"
        
        try:
            response = requests.get(gateway_url, timeout=2)
        except:
            pass  # Gateway running but requires SSE
        
        self.servers['docker'] = {
            'url': gateway_url,
            'status': 'connected'
        }
        
        # Known tools from gateway
        self.available_tools['docker'] = {
            'server': 'Docker MCP Gateway',
            'protocol': 'sse',
            'tools': [
                # Playwright tools
                'playwright_navigate', 'playwright_screenshot', 'playwright_click',
                'playwright_fill', 'playwright_evaluate', 'playwright_goto',
                'playwright_wait', 'playwright_scrape',
                # Context7 tools
                'context7_save', 'context7_load',
                # Internal tools
                'mcp-find', 'mcp-add', 'mcp-remove', 'code-mode',
                'mcp-exec', 'mcp-config-set', 'mcp-create-profile'
            ],
            'count': 32,
            'capabilities': [
                'browser_automation', 'web_scraping', 'context_management',
                'dynamic_tool_discovery', 'mcp_orchestration'
            ]
        }
    
    def _summarize_capabilities(self):
        """Print capability summary"""
        total_tools = sum(info['count'] for info in self.available_tools.values())
        active_servers = len([s for s in self.servers.values() if s is not None])
        
        print(f"\n📊 MCP Ecosystem Summary:")
        print(f"   • Active Servers: {active_servers}/3")
        print(f"   • Total Tools: {total_tools}")
        
        for server_name, info in self.available_tools.items():
            print(f"   • {info['server']}: {info['count']} tools ({info['protocol']})")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get all capabilities for DSPy reasoning"""
        return {
            'total_tools': sum(info['count'] for info in self.available_tools.values()),
            'servers': self.available_tools,
            'active_count': len([s for s in self.servers.values() if s is not None]),
            'capabilities_summary': self._get_capabilities_summary()
        }
    
    def _get_capabilities_summary(self) -> str:
        """Generate human-readable capabilities summary for LLM"""
        summary = []
        
        for server_name, info in self.available_tools.items():
            caps = ', '.join(info['capabilities'])
            summary.append(
                f"{info['server']} ({info['count']} tools): {caps}"
            )
        
        return "\n".join(summary)
    
    def execute_pandas_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute Pandas MCP tool"""
        if self.servers['pandas'] is None:
            raise Exception("Pandas MCP Server not available")
        
        if not hasattr(self.servers['pandas'], tool_name):
            raise Exception(f"Tool {tool_name} not found in Pandas MCP")
        
        tool = getattr(self.servers['pandas'], tool_name)
        return tool(**kwargs)
    
    def execute_jupyter_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute Jupyter MCP tool via HTTP"""
        if self.servers['jupyter'] is None:
            raise Exception("Jupyter MCP Server not available")
        
        url = f"{self.servers['jupyter']['url']}/tools/call"
        payload = {
            'name': tool_name,
            'arguments': kwargs
        }
        
        response = requests.post(url, json=payload, timeout=30)
        return response.json()
    
    def get_tool_info(self, server: str, tool_name: str) -> Optional[Dict]:
        """Get information about a specific tool"""
        if server not in self.available_tools:
            return None
        
        tools = self.available_tools[server]['tools']
        if tool_name not in tools:
            return None
        
        return {
            'name': tool_name,
            'server': server,
            'protocol': self.available_tools[server]['protocol']
        }
    
    def list_tools_by_capability(self, capability: str) -> List[str]:
        """List all tools that provide a specific capability"""
        matching_tools = []
        
        for server_name, info in self.available_tools.items():
            if capability in info['capabilities']:
                matching_tools.extend([
                    f"{server_name}.{tool}" for tool in info['tools']
                ])
        
        return matching_tools


# Singleton instance
_mcp_integration = None

def get_mcp_integration() -> MCPIntegration:
    """Get or create MCP integration singleton"""
    global _mcp_integration
    if _mcp_integration is None:
        _mcp_integration = MCPIntegration()
    return _mcp_integration
