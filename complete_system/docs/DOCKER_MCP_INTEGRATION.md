# 🐳 Docker MCP Toolkit Integration Guide

## Overview

The Docker MCP Toolkit enables **dynamic tool discovery** - your agent can search, install, and use 270+ MCP servers on-demand without manual configuration.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Your Agent (DSPy Universal System)                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Docker MCP Gateway (localhost:12307/mcp)               │
│  ┌────────────────────────────────────────────────┐     │
│  │  Dynamic Operations:                           │     │
│  │  • mcp-find: Search catalog (270+ servers)     │     │
│  │  • mcp-add: Install server on-demand           │     │
│  │  • mcp-compose: Combine multiple servers       │     │
│  │  • mcp-list: Show installed servers            │     │
│  └────────────────────────────────────────────────┘     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Containerized MCP Servers (270+ available)             │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │ Jupyter  │ Postgres │ GitHub   │ Slack    │         │
│  ├──────────┼──────────┼──────────┼──────────┤         │
│  │ Pandas   │ Sklearn  │ Browser  │ Context7 │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Dynamic Tool Discovery**
- Agent searches catalog at runtime: `"I need time-series forecasting tools"`
- Docker MCP finds matching servers: `prophet-mcp`, `statsmodels-mcp`
- Installs and connects automatically

### 2. **Zero Manual Configuration**
- No editing `mcp_config.json` for each new tool
- No dependency management
- No environment conflicts

### 3. **Secure Isolation**
- Each MCP server runs in its own container
- Resource limits (1 CPU, 2GB RAM)
- No host filesystem access by default

### 4. **OAuth Integration**
- Automatic OAuth flow for services (GitHub, Notion, etc.)
- Credentials managed securely
- Revoke access from dashboard

## Installation

### Prerequisites
- Docker Desktop (with MCP Toolkit enabled)
- Docker Desktop version >= 4.36.0

### Setup Steps

1. **Install Docker Desktop**
   - Download from: https://www.docker.com/products/docker-desktop
   - Enable MCP Toolkit in Settings > Features > MCP

2. **Verify Installation**
   ```powershell
   docker mcp gateway status
   # Should show: Gateway running on localhost:12307
   ```

3. **Test Connection**
   ```powershell
   curl http://localhost:12307/mcp/catalog
   # Should return JSON with available servers
   ```

## Usage in Python Agent

### Basic Integration

```python
import requests
from typing import List, Dict

class DockerMCPClient:
    """Client for Docker MCP Toolkit Gateway"""
    
    def __init__(self, gateway_url: str = "http://localhost:12307/mcp"):
        self.gateway_url = gateway_url
    
    def search_tools(self, query: str) -> List[Dict]:
        """Search MCP catalog for tools matching query"""
        response = requests.post(
            f"{self.gateway_url}/find",
            json={"query": query}
        )
        return response.json()["servers"]
    
    def install_server(self, server_name: str) -> Dict:
        """Install an MCP server from catalog"""
        response = requests.post(
            f"{self.gateway_url}/add",
            json={"server": server_name}
        )
        return response.json()
    
    def list_installed(self) -> List[str]:
        """List currently installed MCP servers"""
        response = requests.get(f"{self.gateway_url}/list")
        return response.json()["servers"]
    
    def invoke_tool(self, server: str, tool: str, params: Dict) -> Dict:
        """Invoke a tool from installed server"""
        response = requests.post(
            f"{self.gateway_url}/invoke",
            json={
                "server": server,
                "tool": tool,
                "params": params
            }
        )
        return response.json()
```

### Integration with DSPy Agent

```python
class DynamicToolSelector(dspy.Module):
    """Uses Docker MCP to find and install tools on-demand"""
    
    def __init__(self):
        super().__init__()
        self.mcp_client = DockerMCPClient()
        self.selector = ChainOfThought(ToolSelectionSignature)
    
    def forward(self, task: str, data_type: str):
        # Step 1: Ask LLM what tools are needed
        selection = self.selector(
            task=task,
            data_characteristics=data_type
        )
        
        # Step 2: Search Docker MCP catalog
        available_servers = self.mcp_client.search_tools(
            query=selection.tool_search_query
        )
        
        # Step 3: Install top match
        if available_servers:
            server = available_servers[0]
            self.mcp_client.install_server(server['name'])
            return server
        
        return None
```

## Available MCP Servers (Examples)

### Data Science
- `jupyter-mcp` - Execute Python notebooks
- `pandas-mcp` - DataFrame operations
- `sklearn-mcp` - ML model training
- `prophet-mcp` - Time-series forecasting
- `statsmodels-mcp` - Statistical analysis

### Databases
- `postgres-mcp` - PostgreSQL queries
- `mysql-mcp` - MySQL operations
- `mongodb-mcp` - NoSQL database
- `sqlite-mcp` - Lightweight SQL

### Web & APIs
- `browser-mcp` - Web scraping
- `github-mcp` - GitHub integration
- `slack-mcp` - Slack messaging
- `notion-mcp` - Notion workspace

### AI & ML
- `openai-mcp` - OpenAI API
- `huggingface-mcp` - Model hub
- `context7-mcp` - Vector database
- `llama-mcp` - Local LLM

## Security Considerations

### Resource Limits
- CPU: 1 core per container
- Memory: 2GB per container
- Network: Isolated by default

### Filesystem Access
```python
# By default: NO filesystem access
# To grant access (use sparingly):
docker mcp add pandas-mcp --mount /data:/workspace/data:ro
```

### OAuth Authentication
```python
# Automatic OAuth flow
mcp_client.install_server("github-mcp")
# Opens browser → Authorize → Returns to app
# No manual token management!
```

## Troubleshooting

### Gateway Not Running
```powershell
docker mcp gateway start
```

### Server Install Failed
```powershell
docker mcp logs <server-name>
```

### Clear All Servers
```powershell
docker mcp prune
```

## Integration with Current System

### File: `core/docker_mcp_client.py`
Create a client wrapper (already in system)

### File: `core/dspy_universal_agent.py`
Add dynamic tool selection:
```python
def _select_tools(self, task: str, data_type: str):
    # Use Docker MCP to find appropriate tools
    tools = self.mcp_client.search_tools(
        query=f"{task} {data_type} analysis tools"
    )
    # Install and use
    for tool in tools[:3]:  # Top 3 matches
        self.mcp_client.install_server(tool['name'])
```

## Next Steps

1. ✅ Install Docker Desktop with MCP Toolkit
2. ✅ Verify gateway is running
3. ✅ Update `dspy_universal_agent.py` to use dynamic tools
4. ✅ Test with multi-dataset analysis
5. ✅ Document in paper as novel contribution

## Resources

- Docker MCP Docs: https://docs.docker.com/ai/mcp-catalog-and-toolkit/toolkit/
- MCP Catalog: https://mcp.docker.com/catalog
- GitHub: https://github.com/docker/mcp-toolkit
