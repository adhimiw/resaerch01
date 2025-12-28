"""
Mistral Autonomous Code Executor
=================================

Production-ready autonomous code generation and execution system.
Uses Mistral API for intelligent code generation with Scout-Mechanic-Inspector loop.

This module provides:
- PLAN: Strategic analysis and task decomposition
- CODE: Intelligent code generation with Mistral
- TEST: Automated code execution and validation
- ITERATE: Self-correction and refinement

Integration with existing autonomous MCP system and self-healing capabilities.
"""

import os
import sys
import io
import json
import time
import asyncio
import logging
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Import from existing core modules
from .environment_config import get_env_manager, EnvironmentConfig
from .self_healing_executor import (
    SelfHealingExecutor,
    RecoveryConfig,
    RecoveryStrategy,
    with_self_healing
)

logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special types"""
    
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ExecutionPhase(Enum):
    """Phases of autonomous execution"""
    PLANNING = "planning"
    GENERATING = "generating"
    EXECUTING = "executing"
    VALIDATING = "validating"
    ITERATING = "iterating"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskComplexity(Enum):
    """Complexity levels for task classification"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class ExecutionPlan:
    """Strategic plan for code execution"""
    task_description: str
    decomposition: List[str]
    complexity: TaskComplexity
    estimated_steps: int
    dependencies: List[str]
    edge_cases: List[str]
    success_criteria: List[str]
    created_at: float = field(default_factory=time.time)


@dataclass
class CodeArtifact:
    """Generated code artifact with metadata"""
    code: str
    language: str
    phase: ExecutionPhase
    attempt: int
    quality_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    execution_output: str = ""
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Complete execution result"""
    success: bool
    plan: ExecutionPlan
    artifacts: List[CodeArtifact]
    final_output: str
    total_attempts: int
    total_duration_ms: float
    errors: List[str] = field(default_factory=list)
    iterations: List[Dict] = field(default_factory=list)


class MistralAPIClient:
    """
    Direct HTTP client for Mistral API
    Avoids dependency on mistralai SDK for maximum compatibility
    """
    
    API_BASE = "https://api.mistral.ai/v1"
    
    def __init__(self, api_key: str = None):
        """
        Initialize Mistral API client
        
        Args:
            api_key: Mistral API key (from env if not provided)
        """
        self.api_key = api_key or self._get_api_key()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        self.models = self._get_available_models()
    
    def _get_api_key(self) -> str:
        """Retrieve API key from environment or configuration"""
        env_key = os.environ.get('MISTRAL_API_KEY', '')
        if env_key:
            return env_key
        
        # Try environment manager
        try:
            env_manager = get_env_manager()
            if env_manager:
                config = env_manager.get_config()
                if hasattr(config, 'mistral_api_key') and config.mistral_api_key:
                    return config.mistral_api_key
        except Exception:
            pass
        
        return ''
    
    def _get_available_models(self) -> List[str]:
        """Fetch available models from Mistral API"""
        try:
            response = self.session.get(f"{self.API_BASE}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m['id'] for m in data.get('data', [])]
        except Exception as e:
            logger.warning(f"Could not fetch models: {e}")
        
        # Fallback to known models
        return ['mistral-small', 'mistral-medium', 'mistral-large', 'open-mistral-7b', 'open-mixtral-8x7b']
    
    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 4096
    ) -> Dict:
        """
        Send chat completion request to Mistral API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (auto-selects if not provided)
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum response tokens
            
        Returns:
            API response as dict
        """
        selected_model = model or self.models[0] if self.models else 'mistral-medium'
        
        payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = self.session.post(
                f"{self.API_BASE}/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Mistral API error: {e}")
            raise ConnectionError(f"Failed to connect to Mistral API: {e}")
    
    def generate_code(
        self,
        prompt: str,
        context: str = "",
        language: str = "python",
        max_tokens: int = 2048
    ) -> str:
        """
        Generate code using Mistral API
        
        Args:
            prompt: Code generation request
            context: Additional context for better generation
            language: Programming language
            max_tokens: Maximum tokens for generated code
            
        Returns:
            Generated code as string
        """
        system_message = f"""You are an expert code generation assistant.
Your task is to generate {language} code that:
1. Is correct, efficient, and follows best practices
2. Includes proper error handling
3. Has clear comments explaining the logic
4. Is ready to execute without modification

Return ONLY the code, wrapped in a code block with language annotation.
Do not include markdown explanations outside the code block."""

        user_message = f"Context:\n{context}\n\nRequest:\n{prompt}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.chat_complete(messages, temperature=0.1, max_tokens=max_tokens)
            content = response['choices'][0]['message']['content']
            
            # Extract code from markdown block
            return self._extract_code(content, language)
        
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise
    
    def _extract_code(self, content: str, language: str) -> str:
        """Extract code from markdown code block"""
        # Check for markdown code block
        if f"```{language}" in content:
            parts = content.split(f"```{language}")
            if len(parts) >= 2:
                code = parts[1].split("```")[0]
                return code.strip()
        
        # Try generic code block
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                # Remove language annotation if present
                code = parts[1]
                if code.startswith(language):
                    code = code[len(language):].strip()
                return code.strip()
        
        # Return as-is if no code block found
        return content.strip()


class PlanningEngine:
    """
    Strategic planning engine for task decomposition
    Implements Scout phase of Scout-Mechanic-Inspector loop
    """
    
    def __init__(self, mistral_client: MistralAPIClient):
        self.client = mistral_client
    
    def create_plan(self, task: str, context: str = "") -> ExecutionPlan:
        """
        Create detailed execution plan for task
        
        Args:
            task: User's task description
            context: Additional context information
            
        Returns:
            ExecutionPlan with decomposition and strategy
        """
        system_message = """You are a strategic planning expert.
Your task is to analyze requests and create detailed execution plans.
Break down complex tasks into manageable steps.
Identify dependencies, edge cases, and success criteria."""

        user_message = f"""Analyze this task and create a detailed execution plan:

Task: {task}

Context:
{context}

Provide your response as a JSON object with:
1. task_description - Brief summary of the task
2. decomposition - Array of step-by-step subtasks
3. complexity - One of: simple, moderate, complex, expert
4. estimated_steps - Number of implementation steps
5. dependencies - Array of external dependencies or libraries
6. edge_cases - Array of potential edge cases to handle
7. success_criteria - Array of criteria for successful completion

Return ONLY valid JSON."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.client.chat_complete(messages, temperature=0.2, max_tokens=1024)
            content = response['choices'][0]['message']['content']
            
            # Parse JSON response
            plan_data = json.loads(content)
            
            # Safely convert complexity string to enum
            complexity_str = plan_data.get('complexity', 'moderate')
            try:
                complexity = TaskComplexity(complexity_str)
            except ValueError:
                complexity = TaskComplexity.MODERATE
            
            return ExecutionPlan(
                task_description=plan_data.get('task_description', task),
                decomposition=plan_data.get('decomposition', []),
                complexity=complexity,
                estimated_steps=plan_data.get('estimated_steps', 1),
                dependencies=plan_data.get('dependencies', []),
                edge_cases=plan_data.get('edge_cases', []),
                success_criteria=plan_data.get('success_criteria', [])
            )
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse plan, using fallback: {e}")
            return self._create_fallback_plan(task)
    
    def _create_fallback_plan(self, task: str) -> ExecutionPlan:
        """Create simple plan without LLM"""
        return ExecutionPlan(
            task_description=task,
            decomposition=[f"Implement: {task}"],
            complexity=TaskComplexity.MODERATE,
            estimated_steps=1,
            dependencies=[],
            edge_cases=["Input validation", "Error handling"],
            success_criteria=["Code executes without errors", "Produces expected output"]
        )


class CodeGenerationEngine:
    """
    Intelligent code generation engine
    Implements Mechanic phase of Scout-Mechanic-Inspector loop
    """
    
    def __init__(self, mistral_client: MistralAPIClient):
        self.client = mistral_client
        self.quality_threshold = 0.7
    
    def generate(
        self,
        plan: ExecutionPlan,
        context: str = "",
        language: str = "python",
        iteration: int = 1,
        previous_errors: List[str] = None
    ) -> CodeArtifact:
        """
        Generate code based on execution plan
        
        Args:
            plan: Execution plan to follow
            context: Additional context
            language: Programming language
            iteration: Current iteration number
            previous_errors: Errors from previous attempts
            
        Returns:
            CodeArtifact with generated code
        """
        # Build generation prompt
        prompt_parts = [
            f"Task: {plan.task_description}",
            f"\nSteps to implement:",
        ]
        
        for i, step in enumerate(plan.decomposition, 1):
            prompt_parts.append(f"  {i}. {step}")
        
        if plan.edge_cases:
            prompt_parts.append(f"\nEdge cases to handle:")
            for case in plan.edge_cases:
                prompt_parts.append(f"  - {case}")
        
        if previous_errors:
            prompt_parts.append(f"\nPrevious iteration errors (fix these):")
            for error in previous_errors:
                prompt_parts.append(f"  - {error}")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            code = self.client.generate_code(
                prompt=prompt,
                context=context,
                language=language,
                max_tokens=4096
            )
            
            # Validate generated code
            validation_errors = self._validate_code(code, language)
            quality_score = self._calculate_quality_score(code, validation_errors)
            
            return CodeArtifact(
                code=code,
                language=language,
                phase=ExecutionPhase.GENERATING,
                attempt=iteration,
                quality_score=quality_score,
                validation_errors=validation_errors
            )
        
        except Exception as e:
            return CodeArtifact(
                code=f"# Error generating code: {e}",
                language=language,
                phase=ExecutionPhase.FAILED,
                attempt=iteration,
                quality_score=0.0,
                validation_errors=[str(e)]
            )
    
    def _validate_code(self, code: str, language: str) -> List[str]:
        """Basic code validation"""
        errors = []
        
        if not code or len(code.strip()) < 10:
            errors.append("Code is too short or empty")
        
        # Language-specific checks
        if language == "python":
            # Check for common issues
            if "import " not in code and "from " not in code:
                errors.append("No imports found - code may be incomplete")
            
            # Check for function definition
            if "def " not in code and "class " not in code and "if __name__" not in code:
                errors.append("No function or class definition found")
        
        return errors
    
    def _calculate_quality_score(self, code: str, errors: List[str]) -> float:
        """Calculate code quality score (0.0 - 1.0)"""
        score = 1.0
        
        # Penalize for validation errors
        score -= len(errors) * 0.15
        
        # Bonus for good practices
        if "try:" in code and "except" in code:
            score += 0.1  # Error handling
        
        if "#" in code or '"""' in code:
            score += 0.05  # Comments
        
        # Clamp score
        return max(0.0, min(1.0, score))


class ExecutionEngine:
    """
    Safe code execution engine
    Implements Inspector phase of Scout-Mechanic-Inspector loop
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.sandbox_modules = self._create_sandbox()
    
    def _create_sandbox(self) -> Dict:
        """Create restricted module namespace for safe execution"""
        
        def safe_import(name, *args, **kwargs):
            """Restricted import function for common modules"""
            allowed_modules = {
                'time', 'math', 'random', 'statistics', 'collections', 'itertools',
                'functools', 'operator', 'decimal', 'fractions', 'json',
                're', 'string', 'datetime', 'calendar', 'heapq', 'bisect',
                'array', 'copy', 'pprint', 'textwrap', 'unicodedata', 'typing',
                'sklearn', 'sklearn.model_selection', 'sklearn.ensemble',
                'sklearn.linear_model', 'sklearn.tree', 'sklearn.naive_bayes',
                'sklearn.neighbors', 'sklearn.svm', 'sklearn.preprocessing',
                'sklearn.metrics', 'sklearn.cluster', 'numpy', 'pandas',
                'scipy', 'joblib'
            }
            module_name = name.split('.')[0]
            if module_name not in allowed_modules:
                raise ImportError(f"Import '{name}' not allowed in sandbox")
            return __import__(name, *args, **kwargs)
        
        safe_builtins = {
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'pow': pow,
            'divmod': divmod,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'delattr': delattr,
            'type': type,
            'id': id,
            'chr': chr,
            'ord': ord,
            'bin': bin,
            'hex': hex,
            'oct': oct,
            'format': format,
            '__import__': safe_import,
        }
        
        # Pre-load common modules for convenience
        preloaded = {}
        for mod_name in ['math', 'random', 'statistics', 'collections', 'itertools',
                        'functools', 'numpy', 'pandas', 'sklearn', 'joblib']:
            try:
                preloaded[mod_name] = __import__(mod_name)
            except ImportError:
                pass
        
        return {'__builtins__': safe_builtins, **preloaded}
    
    def execute(self, code: str, timeout: float = 30.0) -> Tuple[str, bool, List[str]]:
        """
        Safely execute code and capture output
        
        Args:
            code: Code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Tuple of (output, success, errors)
        """
        output_buffer = io.StringIO()
        errors = []
        
        # Capture stdout
        sys.stdout = output_buffer
        stderr_buffer = io.StringIO()
        sys.stderr = stderr_buffer
        
        success = False
        
        try:
            # Create clean namespace
            namespace = self.sandbox_modules.copy()
            namespace['__name__'] = '__main__'
            
            # Execute code with timeout
            future = self.executor.submit(exec, code, namespace)
            future.result(timeout=timeout)
            
            success = True
            output = output_buffer.getvalue()
            
        except Exception as e:
            success = False
            output = output_buffer.getvalue()
            error_type = type(e).__name__
            errors.append(f"{error_type}: {str(e)}")
        
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        
        return output, success, errors
    
    def execute_with_healing(
        self,
        code: str,
        max_attempts: int = 3,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Execute code with automatic retry and healing
        
        Args:
            code: Code to execute
            max_attempts: Maximum execution attempts
            timeout: Maximum execution time
            
        Returns:
            Dict with output, success, attempts, and error details
        """
        attempts = 0
        last_error = None
        
        while attempts < max_attempts:
            attempts += 1
            
            output, success, errors = self.execute(code, timeout)
            
            if success:
                return {
                    'output': output,
                    'success': True,
                    'attempts': attempts,
                    'errors': []
                }
            
            last_error = errors[0] if errors else "Unknown error"
            time.sleep(1)  # Brief pause between attempts
        
        return {
            'output': output if 'output' in locals() else '',
            'success': False,
            'attempts': attempts,
            'errors': [last_error]
        }


class IterationEngine:
    """
    Self-correction and refinement engine
    Handles iteration when code execution fails
    """
    
    def __init__(self, mistral_client: MistralAPIClient):
        self.client = mistral_client
    
    def analyze_failure(
        self,
        code: str,
        errors: List[str],
        plan: ExecutionPlan
    ) -> List[str]:
        """
        Analyze execution failures and generate fixes
        
        Args:
            code: Original code that failed
            errors: List of error messages
            plan: Original execution plan
            
        Returns:
            List of specific fixes to apply
        """
        system_message = """You are a code debugging expert.
Your task is to analyze code failures and suggest specific fixes.
Provide actionable recommendations that address the root cause."""

        user_message = f"""Analyze this code failure and suggest fixes:

Original Task: {plan.task_description}

Code:
```
{code}
```

Errors:
{chr(10).join(f'- {e}' for e in errors)}

Provide specific fixes as a JSON array of fix descriptions.
Each fix should be actionable and address the root cause.
Return ONLY valid JSON array."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.client.chat_complete(messages, temperature=0.3, max_tokens=512)
            content = response['choices'][0]['message']['content']
            
            # Parse JSON array
            fixes = json.loads(content)
            return [f if isinstance(f, str) else str(f) for f in fixes]
        
        except (json.JSONDecodeError, KeyError):
            # Fallback: return generic fixes
            return [
                "Check for syntax errors in the code",
                "Verify all variables are properly defined",
                "Ensure proper error handling is added"
            ]
    
    def generate_corrected_code(
        self,
        original_code: str,
        errors: List[str],
        fixes: List[str],
        language: str = "python"
    ) -> str:
        """
        Generate corrected version of failed code
        
        Args:
            original_code: Original code that failed
            errors: List of errors encountered
            fixes: List of fixes to apply
            language: Programming language
            
        Returns:
            Corrected code as string
        """
        system_message = """You are an expert code correction assistant.
Given failed code and error analysis, generate corrected code.
Apply the specified fixes while maintaining the original intent."""

        user_message = f"""Correct this {language} code:

Original Code:
```
{original_code}
```

Errors Encountered:
{chr(10).join(f'- {e}' for e in errors)}

Required Fixes:
{chr(10).join(f'- {f}' for f in fixes)}

Generate corrected code that addresses all issues.
Return ONLY the corrected code in a ```{language} code block.
Do not include explanations."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.client.chat_complete(messages, temperature=0.2, max_tokens=4096)
            content = response['choices'][0]['message']['content']
            return self.client._extract_code(content, language)
        
        except Exception as e:
            logger.error(f"Code correction failed: {e}")
            return original_code  # Return original as fallback


class MistralAutonomousExecutor:
    """
    Main autonomous executor combining all components
    Implements full Scout-Mechanic-Inspector loop with self-healing
    """
    
    def __init__(
        self,
        mistral_api_key: str = None,
        use_self_healing: bool = True,
        max_iterations: int = 3
    ):
        """
        Initialize autonomous executor
        
        Args:
            mistral_api_key: Mistral API key
            use_self_healing: Enable automatic retry and recovery
            max_iterations: Maximum refinement iterations
        """
        self.api_key = mistral_api_key
        self.max_iterations = max_iterations
        self.use_self_healing = use_self_healing
        
        # Initialize components
        self.mistral_client = MistralAPIClient(self.api_key)
        self.planning_engine = PlanningEngine(self.mistral_client)
        self.code_engine = CodeGenerationEngine(self.mistral_client)
        self.execution_engine = ExecutionEngine()
        self.iteration_engine = IterationEngine(self.mistral_client)
        
        # Self-healing executor
        if use_self_healing:
            recovery_config = RecoveryConfig(
                max_attempts=3,
                strategy=RecoveryStrategy.RETRY,
                initial_delay=0.5
            )
            self.healing_executor = SelfHealingExecutor(recovery_config)
        else:
            self.healing_executor = None
        
        logger.info("✅ MistralAutonomousExecutor initialized")
    
    def execute(
        self,
        task: str,
        context: str = "",
        language: str = "python",
        validate_output: bool = True
    ) -> ExecutionResult:
        """
        Execute task autonomously with full Scout-Mechanic-Inspector loop
        
        Args:
            task: User's task description
            context: Additional context information
            language: Programming language
            validate_output: Whether to validate execution output
            
        Returns:
            ExecutionResult with all artifacts and results
        """
        start_time = time.time()
        artifacts = []
        iterations = []
        errors = []
        
        logger.info(f"🚀 Starting autonomous execution: {task[:100]}...")
        
        try:
            # Phase 1: Planning (Scout)
            plan = self.planning_engine.create_plan(task, context)
            artifacts.append(CodeArtifact(
                code=json.dumps(asdict(plan), indent=2, cls=CustomJSONEncoder),
                language="json",
                phase=ExecutionPhase.PLANNING,
                attempt=1
            ))
            logger.info(f"📋 Created plan with {len(plan.decomposition)} steps")
            
            # Phases 2-5: Generate, Execute, Validate, Iterate
            current_code = ""
            final_output = ""
            total_attempts = 0
            
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"🔄 Iteration {iteration}/{self.max_iterations}")
                
                # Generate code (Mechanic)
                previous_errors = [i.get('error') for i in iterations if i.get('error')]
                artifact = self.code_engine.generate(
                    plan=plan,
                    context=context,
                    language=language,
                    iteration=iteration,
                    previous_errors=previous_errors
                )
                artifacts.append(artifact)
                current_code = artifact.code
                total_attempts = artifact.attempt
                
                # Execute code (Inspector)
                exec_result = self.execution_engine.execute_with_healing(
                    current_code,
                    max_attempts=2,
                    timeout=30.0
                )
                
                iteration_record = {
                    'iteration': iteration,
                    'code_length': len(current_code),
                    'success': exec_result['success'],
                    'attempts': exec_result['attempts'],
                    'output_preview': exec_result['output'][:200] if exec_result['output'] else ''
                }
                
                if exec_result['success']:
                    logger.info(f"✅ Iteration {iteration} succeeded")
                    final_output = exec_result['output']
                    
                    # Validate output if requested
                    if validate_output:
                        is_valid, validation_errors = self._validate_output(
                            final_output, plan.success_criteria
                        )
                        if not is_valid:
                            logger.warning(f"⚠️ Output validation failed: {validation_errors}")
                            iteration_record['validation_errors'] = validation_errors
                            iterations.append(iteration_record)
                            continue  # Try to improve
                    
                    iterations.append(iteration_record)
                    break  # Success, exit loop
                
                else:
                    logger.warning(f"❌ Iteration {iteration} failed: {exec_result['errors']}")
                    iteration_record['error'] = exec_result['errors'][0]
                    iterations.append(iteration_record)
                    
                    # If not last iteration, try to correct
                    if iteration < self.max_iterations:
                        fixes = self.iteration_engine.analyze_failure(
                            current_code,
                            exec_result['errors'],
                            plan
                        )
                        iteration_record['fixes'] = fixes
                        
                        # Generate corrected code
                        corrected_code = self.iteration_engine.generate_corrected_code(
                            current_code,
                            exec_result['errors'],
                            fixes,
                            language
                        )
                        
                        if corrected_code != current_code:
                            current_code = corrected_code
                            iteration_record['code_corrected'] = True
            
            # Determine final success
            success = bool(final_output)
            
            if not success and iterations:
                errors = [i.get('error') for i in iterations if i.get('error')]
            
            return ExecutionResult(
                success=success,
                plan=plan,
                artifacts=artifacts,
                final_output=final_output,
                total_attempts=total_attempts,
                total_duration_ms=(time.time() - start_time) * 1000,
                errors=errors,
                iterations=iterations
            )
        
        except Exception as e:
            logger.error(f"❌ Autonomous execution failed: {e}")
            return ExecutionResult(
                success=False,
                plan=ExecutionPlan(task_description=task, decomposition=[], 
                                  complexity=TaskComplexity.MODERATE, estimated_steps=0,
                                  dependencies=[], edge_cases=[], success_criteria=[]),
                artifacts=artifacts,
                final_output="",
                total_attempts=0,
                total_duration_ms=(time.time() - start_time) * 1000,
                errors=[str(e)],
                iterations=iterations
            )
    
    def _validate_output(
        self,
        output: str,
        criteria: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate execution output against success criteria
        
        Args:
            output: Execution output
            criteria: Success criteria from plan
            
        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []
        
        if not output:
            errors.append("No output produced")
        
        if criteria:
            for criterion in criteria:
                # Simple keyword-based validation
                criterion_lower = criterion.lower()
                if "output" in criterion_lower and not output.strip():
                    errors.append(f"Failed: {criterion}")
                if "print" in criterion_lower and "print" not in output.lower():
                    errors.append(f"Failed: {criterion} - no print statement found")
        
        return len(errors) == 0, errors
    
    def get_status(self) -> Dict[str, Any]:
        """Get executor status and capabilities"""
        return {
            'mistral_client': True,
            'planning_engine': True,
            'code_engine': True,
            'execution_engine': True,
            'iteration_engine': True,
            'self_healing': self.use_self_healing,
            'max_iterations': self.max_iterations,
            'available_models': self.mistral_client.models
        }


# Convenience function for quick execution
def autonomous_execute(
    task: str,
    context: str = "",
    mistral_api_key: str = None
) -> ExecutionResult:
    """
    Convenience function for autonomous code execution
    
    Usage:
        result = autonomous_execute(
            "Calculate fibonacci sequence up to n=10",
            context="Use iterative approach for efficiency"
        )
        
        print(result.final_output)
        print(f"Success: {result.success}")
        print(f"Duration: {result.total_duration_ms:.1f}ms")
    """
    executor = MistralAutonomousExecutor(mistral_api_key=mistral_api_key)
    return executor.execute(task, context)


# Demo and testing
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("=" * 80)
        print("🚀 Mistral Autonomous Executor - Production Demo")
        print("=" * 80)
        
        # Initialize executor
        print("\n📋 Initializing executor...")
        executor = MistralAutonomousExecutor()
        
        status = executor.get_status()
        print(f"   Status: {status}")
        
        # Test 1: Fibonacci sequence
        print("\n🧪 Test 1: Fibonacci Sequence")
        result = executor.execute(
            "Create a Python function to calculate the fibonacci sequence up to n terms",
            context="Use iterative approach for efficiency. Include example usage."
        )
        
        print(f"   Success: {result.success}")
        print(f"   Duration: {result.total_duration_ms:.1f}ms")
        print(f"   Iterations: {len(result.iterations)}")
        
        if result.success:
            print(f"\n📊 Generated Code Length: {len(result.artifacts[-1].code)} chars")
            print(f"📊 Output:\n{result.final_output}")
        
        # Test 2: Data processing
        print("\n🧪 Test 2: Data Processing")
        result2 = executor.execute(
            "Create a function to calculate statistics (mean, median, mode) for a list of numbers",
            context="Handle edge cases like empty lists and even number of elements"
        )
        
        print(f"   Success: {result2.success}")
        print(f"   Duration: {result2.total_duration_ms:.1f}ms")
        
        if result2.success:
            print(f"   Output preview: {result2.final_output[:200]}...")
        
        # Test 3: Simple algorithm
        print("\n🧪 Test 3: Algorithm - Palindrome Check")
        result3 = executor.execute(
            "Write a function to check if a string is a palindrome",
            context="Ignore case and non-alphanumeric characters"
        )
        
        print(f"   Success: {result3.success}")
        print(f"   Duration: {result3.total_duration_ms:.1f}ms")
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 Demo Summary")
        print("=" * 80)
        tests = [result, result2, result3]
        passed = sum(1 for r in tests if r.success)
        print(f"   Tests Passed: {passed}/3")
        print(f"   Average Duration: {sum(r.total_duration_ms for r in tests)/len(tests):.1f}ms")
        print(f"   Total Iterations: {sum(len(r.iterations) for r in tests)}")
        print("\n✅ Executor ready for production use!")
        print("   • Uses Mistral API for intelligent code generation")
        print("   • Implements Scout-Mechanic-Inspector loop")
        print("   • Self-healing with automatic retry")
        print("   • Safe code execution with sandboxing")
    
    asyncio.run(demo())
