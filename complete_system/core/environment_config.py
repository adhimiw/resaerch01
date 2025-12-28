"""
Environment Configuration Manager
Automatically loads environment variables and handles configuration
Eliminates hardcoded API keys and platform-specific paths
"""

import os
import json
import platform
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

# Make dotenv optional - graceful fallback if not installed
try:
    import dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


@dataclass
class EnvironmentConfig:
    """Environment configuration with smart defaults"""
    # API Keys
    mistral_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Server Configuration
    jupyter_mcp_url: str = "http://localhost:8888/mcp"
    docker_mcp_port: int = 12307
    mcp_timeout: int = 30
    
    # Paths (auto-detected)
    workspace_dir: Path = Path.home() / "mcp_workspace"
    data_dir: Path = Path.home() / "mcp_workspace" / "data"
    results_dir: Path = Path.home() / "mcp_workspace" / "results"
    
    # Execution Settings
    max_retries: int = 3
    retry_delay: float = 1.0
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'mistral_api_key': self.mistral_api_key[:8] + "..." if self.mistral_api_key else "",
            'langfuse_public_key': self.langfuse_public_key[:8] + "..." if self.langfuse_public_key else "",
            'workspace_dir': str(self.workspace_dir),
            'data_dir': str(self.data_dir),
            'results_dir': str(self.results_dir),
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay
        }


class EnvironmentManager:
    """
    Smart environment configuration manager
    Auto-detects platform, loads .env files, provides defaults
    """
    
    _instance: Optional['EnvironmentManager'] = None
    _config: Optional[EnvironmentConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from .env and environment variables"""
        # Try to load .env file (optional dotenv)
        env_paths = [
            Path(".env"),
            Path(__file__).parent.parent / ".env",
            Path.home() / ".mcp.env",
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                if HAS_DOTENV:
                    dotenv.load_dotenv(env_path)
                else:
                    # Manual .env parsing
                    self._parse_env_file(env_path)
                break
        
        # Detect platform
        system = platform.system()
        
        # Set up paths based on platform
        if system == "Windows":
            workspace = Path(os.environ.get('USERPROFILE', Path.home())) / "mcp_workspace"
        else:
            workspace = Path.home() / "mcp_workspace"
        
        # Create config
        self._config = EnvironmentConfig(
            mistral_api_key=os.environ.get('MISTRAL_API_KEY', ''),
            langfuse_public_key=os.environ.get('LANGFUSE_PUBLIC_KEY', ''),
            langfuse_secret_key=os.environ.get('LANGFUSE_SECRET_KEY', ''),
            openai_api_key=os.environ.get('OPENAI_API_KEY', ''),
            anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY', ''),
            jupyter_mcp_url=os.environ.get('JUPYTER_MCP_URL', 'http://localhost:8888/mcp'),
            docker_mcp_port=int(os.environ.get('DOCKER_MCP_PORT', 12307)),
            max_retries=int(os.environ.get('MAX_RETRIES', 3)),
            retry_delay=float(os.environ.get('RETRY_DELAY', 1.0)),
            workspace_dir=workspace,
            data_dir=workspace / "data",
            results_dir=workspace / "results"
        )
        
        # Create directories
        self._config.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._config.data_dir.mkdir(parents=True, exist_ok=True)
        self._config.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_env_file(self, env_path: Path):
        """Manually parse .env file if dotenv is not available"""
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception:
            pass  # Silently ignore if .env file can't be parsed
    
    @property
    def config(self) -> EnvironmentConfig:
        """Get current configuration"""
        if self._config is None:
            self._load_configuration()
        return self._config
    
    def get_api_key(self, provider: str = "mistral") -> str:
        """Get API key for provider"""
        keys = {
            "mistral": self.config.mistral_api_key,
            "openai": self.config.openai_api_key,
            "anthropic": self.config.anthropic_api_key,
            "langfuse_public": self.config.langfuse_public_key,
            "langfuse_secret": self.config.langfuse_secret_key
        }
        return keys.get(provider.lower(), '')
    
    def is_configured(self) -> bool:
        """Check if minimum configuration is present"""
        return bool(self.config.mistral_api_key or self.config.openai_api_key)


# Global access
def get_env_manager() -> EnvironmentManager:
    """Get environment manager instance"""
    return EnvironmentManager()


def get_config() -> EnvironmentConfig:
    """Get current configuration"""
    return get_env_manager().config
