"""
Self-Healing Execution Utilities
Provides automatic error recovery, retry mechanisms, and fallback strategies
"""

import asyncio
import time
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Strategies for self-healing"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    CUSTOM = "custom"


@dataclass
class ExecutionAttempt:
    """Record of an execution attempt"""
    attempt_number: int
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RecoveryConfig:
    """Configuration for self-healing behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    fallback_function: Optional[Callable] = None
    allowed_exceptions: tuple = (Exception,)
    log_attempts: bool = True


class SelfHealingExecutor:
    """
    Self-healing execution wrapper with automatic recovery
    """
    
    def __init__(self, config: RecoveryConfig = None):
        self.config = config or RecoveryConfig()
        self.execution_history: List[ExecutionAttempt] = []
    
    def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> ExecutionAttempt:
        """
        Execute function with self-healing
        
        Returns:
            ExecutionAttempt with result and metadata
        """
        attempt = 0
        last_error = None
        start_time = time.time()
        
        while attempt < self.config.max_attempts:
            attempt += 1
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                execution_attempt = ExecutionAttempt(
                    attempt_number=attempt,
                    success=True,
                    result=result,
                    duration_ms=duration_ms
                )
                
                self.execution_history.append(execution_attempt)
                
                if self.config.log_attempts:
                    logger.info(f"✅ Attempt {attempt} succeeded in {duration_ms:.1f}ms")
                
                return execution_attempt
                
            except self.config.allowed_exceptions as e:
                last_error = str(e)
                duration_ms = (time.time() - start_time) * 1000
                
                execution_attempt = ExecutionAttempt(
                    attempt_number=attempt,
                    success=False,
                    error=last_error,
                    duration_ms=duration_ms
                )
                
                self.execution_history.append(execution_attempt)
                
                if self.config.log_attempts:
                    logger.warning(f"⚠️ Attempt {attempt} failed: {last_error}")
                
                # Apply recovery strategy
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    
                    if self.config.strategy == RecoveryStrategy.RETRY:
                        time.sleep(delay)
                    
                    elif self.config.strategy == RecoveryStrategy.FALLBACK:
                        if self.config.fallback_function:
                            return self._execute_fallback(func, args, kwargs)
                    
                    elif self.config.strategy == RecoveryStrategy.SKIP:
                        logger.info("⏭️ Skipping due to repeated failures")
                        break
        
        # All attempts failed
        duration_ms = (time.time() - start_time) * 1000
        return ExecutionAttempt(
            attempt_number=attempt,
            success=False,
            error=f"All {attempt} attempts failed. Last error: {last_error}",
            duration_ms=duration_ms
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        delay = self.config.initial_delay * (self.config.exponential_base ** (attempt - 1))
        return min(delay, self.config.max_delay)
    
    def _execute_fallback(
        self,
        original_func: Callable,
        args: tuple,
        kwargs: dict
    ) -> ExecutionAttempt:
        """Execute fallback function"""
        if self.config.fallback_function:
            try:
                start_time = time.time()
                result = self.config.fallback_function(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                return ExecutionAttempt(
                    attempt_number=1,
                    success=True,
                    result=result,
                    duration_ms=duration_ms
                )
            except Exception as e:
                return ExecutionAttempt(
                    attempt_number=1,
                    success=False,
                    error=f"Fallback also failed: {str(e)}"
                )
        
        return ExecutionAttempt(
            attempt_number=1,
            success=False,
            error="No fallback available"
        )
    
    def get_best_result(self) -> Any:
        """Get result from most successful attempt"""
        successful = [e for e in self.execution_history if e.success]
        if successful:
            return successful[0].result
        return None
    
    def get_failure_summary(self) -> Dict:
        """Get summary of failures"""
        return {
            'total_attempts': len(self.execution_history),
            'successful_attempts': sum(1 for e in self.execution_history if e.success),
            'failed_attempts': sum(1 for e in self.execution_history if not e.success),
            'success_rate': sum(1 for e in self.execution_history if e.success) / len(self.execution_history) if self.execution_history else 0
        }


def with_self_healing(
    max_attempts: int = 3,
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    allowed_exceptions: tuple = (Exception,)
):
    """
    Decorator for self-healing function execution
    
    Usage:
        @with_self_healing(max_attempts=3, strategy=RecoveryStrategy.RETRY)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = RecoveryConfig(
                max_attempts=max_attempts,
                strategy=strategy,
                allowed_exceptions=allowed_exceptions
            )
            executor = SelfHealingExecutor(config)
            result = executor.execute(func, *args, **kwargs)
            
            if result.success:
                return result.result
            else:
                # Raise exception with context
                raise RuntimeError(f"Self-healing execution failed: {result.error}")
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            config = RecoveryConfig(
                max_attempts=max_attempts,
                strategy=strategy,
                allowed_exceptions=allowed_exceptions
            )
            executor = SelfHealingExecutor(config)
            
            # For async, just run sync version
            result = executor.execute(func, *args, **kwargs)
            
            if result.success:
                return result.result
            else:
                raise RuntimeError(f"Self-healing execution failed: {result.error}")
        
        # Preserve async nature
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


class FallbackChain:
    """
    Chain of functions to try in order
    Falls back to next function if one fails
    """
    
    def __init__(self):
        self.chain: List[Callable] = []
    
    def add(self, func: Callable, *args, **default_kwargs):
        """Add function to chain"""
        self.chain.append((func, args, default_kwargs))
    
    def execute(self, *primary_args, **primary_kwargs) -> Any:
        """
        Execute chain, returning first successful result
        
        Returns:
            Result from first successful function or raises exception
        """
        errors = []
        
        for func, args, default_kwargs in self.chain:
            merged_kwargs = {**default_kwargs, **primary_kwargs}
            
            try:
                result = func(*args, *primary_args, **merged_kwargs)
                logger.info(f"✅ Fallback chain succeeded with {func.__name__}")
                return result
            except Exception as e:
                errors.append(f"{func.__name__}: {str(e)}")
                logger.warning(f"⚠️ {func.__name__} failed, trying next...")
        
        raise RuntimeError(f"All fallback options failed: {'; '.join(errors)}")


# Convenience class for common patterns
class RobustMCPTool:
    """
    Wrapper for MCP tools with self-healing capabilities
    """
    
    def __init__(self, server_instance, tool_name: str):
        self.server = server_instance
        self.tool_name = tool_name
        self.executor = SelfHealingExecutor()
    
    def __call__(self, **kwargs) -> Any:
        """Execute tool with self-healing"""
        if not hasattr(self.server, self.tool_name):
            raise AttributeError(f"Tool {self.tool_name} not found")
        
        tool = getattr(self.server, self.tool_name)
        
        def execute_tool():
            return tool(**kwargs)
        
        result = self.executor.execute(execute_tool)
        
        if result.success:
            return result.result
        else:
            raise RuntimeError(f"Tool execution failed: {result.error}")
    
    def with_fallback(self, fallback_tool: str, fallback_server=None):
        """Add fallback tool"""
        self.executor.config.strategy = RecoveryStrategy.FALLBACK
        self.executor.config.fallback_function = lambda **kw: getattr(
            fallback_server or self.server, fallback_tool
        )(**kw)
        return self
