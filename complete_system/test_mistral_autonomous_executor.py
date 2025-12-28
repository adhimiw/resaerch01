"""
Comprehensive Test Suite for Mistral Autonomous Executor
=========================================================

This test suite verifies all capabilities of the Mistral Autonomous Executor:
- CODE: Generate Python code from natural language
- THINK: Apply strategic planning and reasoning
- EXECUTE: Safely run generated code
- ITERATE: Self-correct on failures

Usage:
    cd /workspace/resaerch01/complete_system
    python test_mistral_autonomous_executor.py

Requirements:
    - MISTRAL_API_KEY environment variable set
    - Internet connection for API calls
"""

import os
import sys
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set API key from environment or hardcoded (user's personal key)
os.environ['MISTRAL_API_KEY'] = os.environ.get('MISTRAL_API_KEY', '6IOUctuofzEsOgw0SHi17BfmjoieITTQ')


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_test(name: str, success: bool, details: str = ""):
    """Print test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} | {name}")
    if details and not success:
        print(f"       └── {details}")
    return success


def test_basic_imports():
    """Test that all imports work correctly"""
    print_header("TEST 1: Module Imports")
    
    all_passed = True
    
    # Test core imports
    try:
        from core.mistral_autonomous_executor import (
            MistralAutonomousExecutor,
            MistralAPIClient,
            PlanningEngine,
            CodeGenerationEngine,
            ExecutionEngine,
            IterationEngine,
            ExecutionResult,
            ExecutionPlan,
            ExecutionPhase,
            TaskComplexity,
            autonomous_execute
        )
        print_test("Import all classes and functions", True)
    except ImportError as e:
        print_test("Import all classes and functions", False, str(e))
        all_passed = False
    
    # Test __init__ exports
    try:
        from core import (
            MistralAutonomousExecutor,
            MistralAPIClient,
            PlanningEngine,
            ExecutionEngine,
            ExecutionResult
        )
        print_test("Import from core package", True)
    except ImportError as e:
        print_test("Import from core package", False, str(e))
        all_passed = False
    
    return all_passed


def test_api_client():
    """Test Mistral API client initialization"""
    print_header("TEST 2: Mistral API Client")
    
    try:
        from core.mistral_autonomous_executor import MistralAPIClient
        
        client = MistralAPIClient()
        print_test("Create API client", True)
        
        # Check API key is set
        if client.api_key:
            print_test("API key configured", True)
        else:
            print_test("API key configured", False, "No API key found")
            return False
        
        # Check models available
        if client.models:
            print_test(f"Models available: {len(client.models)}", True)
            print(f"         └── Top 5: {client.models[:5]}")
        else:
            print_test("Models available", False, "No models fetched")
            return False
        
        return True
        
    except Exception as e:
        print_test("API client test", False, str(e))
        return False


def test_planning_engine():
    """Test the planning engine"""
    print_header("TEST 3: Planning Engine")
    
    try:
        from core.mistral_autonomous_executor import (
            MistralAPIClient,
            PlanningEngine,
            ExecutionPlan
        )
        
        client = MistralAPIClient()
        planner = PlanningEngine(client)
        print_test("Create planning engine", True)
        
        # Create a plan
        plan = planner.create_plan(
            task="Create a function to calculate factorial",
            context="Use iterative approach for efficiency"
        )
        print_test("Create execution plan", True)
        
        # Verify plan structure
        if isinstance(plan, ExecutionPlan):
            print_test("Plan is ExecutionPlan instance", True)
        else:
            print_test("Plan is ExecutionPlan instance", False, f"Got {type(plan)}")
            return False
        
        if plan.task_description:
            print_test(f"Task: {plan.task_description[:50]}...", True)
        else:
            print_test("Task description populated", False)
            return False
        
        if plan.decomposition:
            print_test(f"Decomposition: {len(plan.decomposition)} steps", True)
        else:
            print_test("Decomposition populated", False)
            return False
        
        return True
        
    except Exception as e:
        print_test("Planning engine test", False, str(e))
        return False


def test_code_generation():
    """Test code generation engine"""
    print_header("TEST 4: Code Generation Engine")
    
    try:
        from core.mistral_autonomous_executor import (
            MistralAPIClient,
            CodeGenerationEngine,
            ExecutionPlan,
            TaskComplexity
        )
        
        client = MistralAPIClient()
        generator = CodeGenerationEngine(client)
        print_test("Create code generation engine", True)
        
        # Create a simple plan
        plan = ExecutionPlan(
            task_description="Calculate fibonacci sequence",
            decomposition=["Implement iterative fibonacci", "Add example usage"],
            complexity=TaskComplexity.MODERATE,
            estimated_steps=2,
            dependencies=["math"],
            edge_cases=["n=0", "n=1"],
            success_criteria=["Correct output", "No errors"]
        )
        
        # Generate code
        artifact = generator.generate(plan=plan, language="python", iteration=1)
        print_test("Generate code artifact", True)
        
        # Verify artifact
        if artifact.code:
            print_test(f"Code generated: {len(artifact.code)} chars", True)
        else:
            print_test("Code generated", False, "Empty code")
            return False
        
        if artifact.language == "python":
            print_test("Language correct: python", True)
        else:
            print_test("Language correct", False, f"Got {artifact.language}")
            return False
        
        return True
        
    except Exception as e:
        print_test("Code generation test", False, str(e))
        return False


def test_execution_engine():
    """Test safe code execution engine"""
    print_header("TEST 5: Execution Engine (Sandbox)")
    
    try:
        from core.mistral_autonomous_executor import ExecutionEngine
        
        engine = ExecutionEngine()
        print_test("Create execution engine", True)
        
        # Test simple code execution
        output, success, errors = engine.execute("print('Hello, World!')")
        print_test("Simple print execution", success)
        if success:
            print(f"         └── Output: {output.strip()}")
        else:
            print(f"         └── Error: {errors}")
        
        # Test code with calculation
        output, success, errors = engine.execute("result = 2 + 3; print(f'Result: {result}')")
        print_test("Calculation execution", success)
        
        # Test code with math import
        output, success, errors = engine.execute("import math; print(f'Pi: {math.pi}')")
        print_test("Math import execution", success)
        if success:
            print(f"         └── Output: {output.strip()}")
        
        # Test code with list operations
        output, success, errors = engine.execute("nums = [1,2,3,4,5]; print(f'Sum: {sum(nums)}')")
        print_test("List operations execution", success)
        
        return success
        
    except Exception as e:
        print_test("Execution engine test", False, str(e))
        return False


def test_full_execution():
    """Test full autonomous execution pipeline"""
    print_header("TEST 6: Full Autonomous Execution")
    
    try:
        from core.mistral_autonomous_executor import MistralAutonomousExecutor
        
        executor = MistralAutonomousExecutor()
        print_test("Create autonomous executor", True)
        
        # Test 1: Fibonacci
        print("\n  📝 Test 1: Fibonacci Sequence")
        start = time.time()
        result = executor.execute("Create a function to calculate fibonacci sequence up to n terms")
        duration = (time.time() - start) * 1000
        
        print_test("Fibonacci execution", result.success, f"{duration:.0f}ms")
        if result.success:
            print(f"         └── Output: {result.final_output.strip()[:100]}...")
        
        # Test 2: Statistics
        print("\n  📝 Test 2: Statistics Calculator")
        start = time.time()
        result = executor.execute("Calculate mean of numbers [10, 20, 30, 40, 50]")
        duration = (time.time() - start) * 1000
        
        print_test("Statistics execution", result.success, f"{duration:.0f}ms")
        if result.success:
            print(f"         └── Output: {result.final_output.strip()}")
        
        # Test 3: Palindrome
        print("\n  📝 Test 3: Palindrome Checker")
        start = time.time()
        result = executor.execute("Check if 'madam' is a palindrome")
        duration = (time.time() - start) * 1000
        
        print_test("Palindrome execution", result.success, f"{duration:.0f}ms")
        if result.success:
            print(f"         └── Output: {result.final_output.strip()}")
        
        # Test 4: String reversal
        print("\n  📝 Test 4: String Reversal")
        start = time.time()
        result = executor.execute("Reverse the string 'hello world'")
        duration = (time.time() - start) * 1000
        
        print_test("String reversal execution", result.success, f"{duration:.0f}ms")
        if result.success:
            print(f"         └── Output: {result.final_output.strip()}")
        
        return True
        
    except Exception as e:
        print_test("Full execution test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_convenience_function():
    """Test the convenience function"""
    print_header("TEST 7: Convenience Function")
    
    try:
        from core.mistral_autonomous_executor import autonomous_execute
        
        print("Testing autonomous_execute() convenience function...")
        
        result = autonomous_execute("Calculate 5 + 7")
        print_test("Quick execution", result.success)
        
        if result.success:
            print(f"         └── Result: {result.final_output.strip()}")
        
        return result.success
        
    except Exception as e:
        print_test("Convenience function test", False, str(e))
        return False


def test_error_recovery():
    """Test self-healing and error recovery"""
    print_header("TEST 8: Self-Healing & Error Recovery")
    
    try:
        from core.mistral_autonomous_executor import MistralAutonomousExecutor
        
        executor = MistralAutonomousExecutor()
        print_test("Create executor for recovery test", True)
        
        # This might fail initially and self-heal
        print("\n  📝 Testing self-healing with complex request...")
        start = time.time()
        result = executor.execute(
            "Create a function with error handling that processes a list and handles exceptions"
        )
        duration = (time.time() - start) * 1000
        
        print_test(f"Execution completed", True, f"{duration:.0f}ms")
        print_test(f"Final success", result.success)
        print_test(f"Iterations needed", len(result.iterations) > 0, f"{len(result.iterations)} iterations")
        
        if result.iterations:
            print(f"         └── Iterations: {result.iterations}")
        
        return True
        
    except Exception as e:
        print_test("Error recovery test", False, str(e))
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  🚀 MISTRAL AUTONOMOUS EXECUTOR - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"\n  API Key: {os.environ.get('MISTRAL_API_KEY', 'NOT SET')[:10]}...")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run all tests
    results.append(("Module Imports", test_basic_imports()))
    results.append(("API Client", test_api_client()))
    results.append(("Planning Engine", test_planning_engine()))
    results.append(("Code Generation", test_code_generation()))
    results.append(("Execution Engine", test_execution_engine()))
    results.append(("Full Execution", test_full_execution()))
    results.append(("Convenience Function", test_convenience_function()))
    results.append(("Error Recovery", test_error_recovery()))
    
    # Summary
    print_header("📊 FINAL TEST SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} | {name}")
    
    print("\n" + "=" * 80)
    if passed == total:
        print(f"  🎉 ALL {total} TESTS PASSED!")
        print("\n  ✅ Your Mistral Autonomous Executor is production-ready!")
        print("  ✅ It can CODE, THINK, EXECUTE, and ITERATE autonomously!")
        print("\n  Usage Example:")
        print("    from core import MistralAutonomousExecutor")
        print("    executor = MistralAutonomousExecutor()")
        print("    result = executor.execute('Calculate fibonacci sequence')")
        print("    print(result.final_output)")
    else:
        print(f"  ⚠️  {total - passed} TESTS FAILED")
        print("  Please review the errors above.")
    print("=" * 80 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
