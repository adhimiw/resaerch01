# Docker MCP Toolkit Setup Guide

## Prerequisites

### 1. Install Docker Desktop
Download and install the latest version: https://www.docker.com/products/docker-desktop

### 2. Enable Docker MCP Toolkit (Beta Feature)

1. Open Docker Desktop settings
2. Select **Beta features**
3. Check **Enable Docker MCP Toolkit**
4. Select **Apply & Restart**

## Quick Start

### Step 1: Install MCP Servers

**Via Docker Desktop UI:**
1. Open Docker Desktop
2. Navigate to **MCP Toolkit** → **Catalog** tab
3. Search and add servers (e.g., "GitHub Official", "Playwright")
4. Configure authentication if required (e.g., OAuth for GitHub)

**Popular MCP Servers:**
- **GitHub Official** - Repository management, issues, PRs
- **Playwright** - Browser automation
- **Filesystem** - File operations
- **PostgreSQL** - Database access
- **Brave Search** - Web search capabilities

### Step 2: Connect Your Application

**For Python apps (like our Universal Agent), add to `mcp_config.json`:**

```json
{
  "servers": {
    "MCP_DOCKER": {
      "command": "docker",
      "args": ["mcp", "gateway", "run"],
      "type": "stdio"
    }
  }
}
```

**That's it!** Docker MCP Toolkit manages all installed servers through this single gateway.

### Step 3: Verify Connection

```powershell
# Check MCP servers are accessible
docker mcp gateway run

# List installed servers in Docker Desktop
# Go to: MCP Toolkit → Catalog → Installed tab
```

## Available MCP Servers (Growing Catalog)

Browse the full catalog in Docker Desktop or at: https://mcp-catalog.docker.com

**Categories:**
- **Development**: GitHub, GitLab, Azure DevOps
- **Browsers**: Playwright, Puppeteer
- **Databases**: PostgreSQL, MySQL, MongoDB, Redis
- **Cloud**: AWS, Google Cloud, Azure
- **Search**: Brave Search, Google Search
- **Files**: Filesystem, S3, Google Drive
- **Analytics**: Pandas, DuckDB
- **And many mowith Universal Agent

### Option 1: Update mcp_config.json (Recommended)

```json
{
  "mcpServers": {
    "MCP_DOCKER": {
      "command": "docker",
      "args": ["mcp", "gateway", "run"],
      "type": "stdio",
      "description": "Docker MCP Toolkit - All installed servers"
    },
    "pandas-mcp": {
      "command": "python",
      "args": ["core/pandas_mcp_server.py"],
      "description": "Custom pandas MCP (20+ tools)"
    }
  }
}Verify Setup

### Check Installed Servers

**In Docker Desktop:**
1. Open **MCP Toolkit** → **Catalog**
2. Select **Installed** tab
3. See all active MCP servers

**Via CLI:**
```powershell
# Test the gateway command
docker mcp gateway run
# (Should start without errors - press Ctrl+C to stop)

# Check Docker Desktop MCP settings
# Settings → Beta features → Docker MCP Toolkit (should be enabled)
```

### Test with Supported Clients

Docker MCP Toolkit works with these AI clients:
- ✅ Claude Desktop / Claude Code
- ✅ Cursor
- ✅ Visual Studio Code (GitHub Copilot)
- ✅ Continue
- ✅ Zed
- ✅ OpenCode
- ✅ Goose
- And more...

Many of these have **Connect** buttons in Docker Desktop under **MCP Toolkit → Clients**

## Troubleshooting

**"Enable Docker MCP Toolkit" not visible?**
- Update Docker Desktop to latest version
- Beta feature only available in recent versions

**Gateway command fails?**
```powershell
# Check Docker is running
docker ps

# Verify MCP Toolkit is enabled
# Docker Desktop → Settings → Beta features
```

**Servers not showing up?**
- Restart Docker Desktop
- Re-install server from Catalog
- Check server status in MCP Toolkit UI

**Authentication issues (GitHub, etc.)?**
- Click server in Catalog
- Select **Configuration** tab
- Re-authenticate via OAuth

## Learn More

- **Official Docs**: https://docs.docker.com/ai/mcp-catalog-and-toolkit/get-started/
- **MCP Catalog**: https://mcp-catalog.docker.com
- **MCP Specification**: https://spec.modelcontextprotocol.io/
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Send MCP requests via stdio
request = {
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
}
process.stdin.write(json.dumps(request) + "\n")
process.stdin.flush()

# Read response
response = process.stdout.readline()
tools = json.loads(response)
print(f"Available tools: {len(tools['result']['tools'])}")
```

### Option 3: Auto-Discovery (Future Enhancement)

The agent can automatically detect and use Docker MCP tools:

```python
from core.dspy_universal_agent import UniversalAgenticDataScience

agent = UniversalAgenticDataScience(
    enable_docker_mcp=True  # Automatically connects to gateway
)

# Agent discovers GitHub, Playwright, etc. tools
result = agent.analyze("data/
)

# Agent will automatically discover and use relevant tools
result = agent.analyze("data/my_dataset.csv")
```

## Troubleshooting

**Gateway not starting?**
- Check Docker Desktop is running
- Verify port 12307 is not in use: `netstat -ano | Select-String 12307`

**Connection refused?**
- Restart gateway: `docker restart mcp-gateway`
- Check logs: `docker logs mcp-gateway`

**Slow tool discovery?**
- Gateway caches server list for 5 minutes
- Force refresh: `docker restart mcp-gateway`
