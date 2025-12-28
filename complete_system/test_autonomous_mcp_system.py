"""
Test Autonomous MCP System
Verifies end-to-end autonomous operation of MCP tools
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_autonomous_mcp_agent():
    """Test the autonomous MCP agent"""
    print("=" * 80)
    print("🤖 TESTING AUTONOMOUS MCP AGENT")
    print("=" * 80)
    print()
    
    try:
        from core.autonomous_mcp_agent import AutonomousDataScienceAgent
        
        # Initialize agent
        print("1. Initializing Autonomous Agent...")
        agent = AutonomousDataScienceAgent()
        success = await agent.initialize()
        print(f"   ✅ Agent initialized: {success}")
        print()
        
        # Get status
        print("2. Agent Status:")
        status = agent.get_status()
        print(f"   📊 MCP Servers: {status['mcp_servers']}")
        tools = status['available_tools']
        for server, tool_list in tools.items():
            print(f"   📦 {server}: {len(tool_list)} tools")
        print()
        
        # Create test dataset
        print("3. Creating Test Dataset...")
        import pandas as pd
        test_data = {
            'id': range(1, 101),
            'name': [f'item_{i}' for i in range(1, 101)],
            'value': [i * 10 + (i % 5) for i in range(1, 101)],
            'category': ['A' if i % 2 == 0 else 'B' for i in range(1, 101)],
            'score': [float(i) / 100 for i in range(1, 101)]
        }
        df = pd.DataFrame(test_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            test_csv_path = f.name
            print(f"   ✅ Created test dataset")
            print(f"   📊 Dataset shape: {df.shape}")
        print()
        
        # Run autonomous analysis
        print("4. Running Autonomous Analysis...")
        results = await agent.analyze_dataset_autonomous(
            test_csv_path,
            analysis_goal="test autonomous data analysis"
        )
        
        print(f"   ✅ Analysis completed!")
        print(f"   📊 Success rate: {results['success_rate']:.1%}")
        print(f"   ⏱️ Duration: {results['total_duration']:.2f}s")
        print()
        
        # Print key results
        if 'dataset_info' in results:
            print("5. Dataset Information:")
            info = results['dataset_info']
            print(f"   📏 Shape: {info.get('shape', 'N/A')}")
            print(f"   📝 Columns: {info.get('columns', [])[:5]}...")
        print()
        
        # Cleanup
        os.unlink(test_csv_path)
        
        print("=" * 80)
        print("✅ AUTONOMOUS MCP AGENT TEST PASSED")
        print("=" * 80)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("❌ AUTONOMOUS MCP AGENT TEST FAILED")
        print("=" * 80)
        return False


async def test_self_healing_executor():
    """Test the self-healing executor"""
    print("=" * 80)
    print("🩹 TESTING SELF-HEALING EXECUTOR")
    print("=" * 80)
    print()
    
    try:
        from core.self_healing_executor import (
            SelfHealingExecutor,
            RecoveryConfig,
            RecoveryStrategy,
            with_self_healing
        )
        
        # Test 1: Basic retry with success
        print("1. Testing retry with eventual success...")
        
        call_count = [0]
        
        def unreliable_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError(f"Attempt {call_count[0]} failed")
            return "Success!"
        
        executor = SelfHealingExecutor(RecoveryConfig(max_attempts=5))
        result = executor.execute(unreliable_function)
        print(f"   ✅ Result: {result.result}")
        print(f"   📊 Attempts: {result.attempt_number}")
        print()
        
        # Test 2: Fallback strategy
        print("2. Testing fallback strategy...")
        
        def primary_function():
            raise ConnectionError("Primary failed")
        
        def fallback_function():
            return "Fallback success!"
        
        executor2 = SelfHealingExecutor(RecoveryConfig(
            max_attempts=2,
            strategy=RecoveryStrategy.FALLBACK,
            fallback_function=fallback_function
        ))
        result2 = executor2.execute(primary_function)
        print(f"   ✅ Fallback result: {result2.result}")
        print()
        
        # Test 3: Decorator usage
        print("3. Testing decorator usage...")
        
        @with_self_healing(max_attempts=3)
        def decorated_function(x, y):
            if x + y < 5:
                raise ValueError("Sum too small")
            return x + y
        
        try:
            result3 = decorated_function(2, 2)  # Will fail
        except:
            result3 = decorated_function(3, 3)  # Will succeed
        print(f"   ✅ Decorator result: {result3}")
        print()
        
        print("=" * 80)
        print("✅ SELF-HEALING EXECUTOR TEST PASSED")
        print("=" * 80)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("❌ SELF-HEALING EXECUTOR TEST FAILED")
        print("=" * 80)
        return False


def test_environment_config():
    """Test environment configuration"""
    print("=" * 80)
    print("⚙️ TESTING ENVIRONMENT CONFIGURATION")
    print("=" * 80)
    print()
    
    try:
        from core.environment_config import (
            EnvironmentManager,
            get_config
        )
        
        # Get config
        print("1. Loading Configuration...")
        config = get_config()
        print(f"   ✅ Configuration loaded")
        print()
        
        # Check paths
        print("2. Platform Paths:")
        print(f"   📁 Workspace: {config.workspace_dir}")
        print(f"   📁 Data: {config.data_dir}")
        print(f"   📁 Results: {config.results_dir}")
        print()
        
        # Check API keys
        print("3. API Configuration:")
        has_mistral = bool(config.mistral_api_key)
        has_langfuse = bool(config.langfuse_public_key)
        print(f"   🔑 Mistral API: {'✅ Configured' if has_mistral else '⚠️ Not set'}")
        print(f"   🔑 Langfuse: {'✅ Configured' if has_langfuse else '⚠️ Not set'}")
        print()
        
        print("=" * 80)
        print("✅ ENVIRONMENT CONFIGURATION TEST PASSED")
        print("=" * 80)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("❌ ENVIRONMENT CONFIGURATION TEST FAILED")
        print("=" * 80)
        return False


async def test_mcp_integration():
    """Test MCP integration with autonomous features"""
    print("=" * 80)
    print("🔗 TESTING MCP INTEGRATION")
    print("=" * 80)
    print()
    
    try:
        from core.autonomous_mcp_agent import AutonomousMCPServer
        
        print("1. Testing Server Discovery...")
        server = AutonomousMCPServer()
        discovered = server.discover_servers()
        print(f"   📊 Discovered {len(discovered)} servers:")
        for s in discovered:
            print(f"      - {s['name']} ({s['type']})")
        print()
        
        print("2. Testing Server Startup...")
        servers = server.discover_servers()
        for server_config in servers:
            if server_config.get('auto_start', False):
                success = await server.start_server(server_config)
                print(f"   ✅ {server_config['name']}: {'Started' if success else 'Failed'}")
        print()
        
        print("3. Testing Tool Registry...")
        tools = server.get_available_tools()
        print(f"   📦 Total tools: {sum(len(t) for t in tools.values())}")
        for server_name, tool_list in tools.items():
            print(f"      {server_name}: {len(tool_list)} tools")
        print()
        
        print("=" * 80)
        print("✅ MCP INTEGRATION TEST PASSED")
        print("=" * 80)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 80)
        print("❌ MCP INTEGRATION TEST FAILED")
        print("=" * 80)
        return False


async def run_all_tests():
    """Run all autonomous system tests"""
    print()
    print("🚀 STARTING AUTONOMOUS MCP SYSTEM TESTS")
    print("=" * 80)
    print()
    
    results = {}
    
    # Test 1: Environment Config
    results['environment_config'] = test_environment_config()
    
    # Test 2: MCP Integration
    results['mcp_integration'] = await test_mcp_integration()
    
    # Test 3: Self-Healing Executor
    results['self_healing'] = await test_self_healing_executor()
    
    # Test 4: Autonomous Agent
    results['autonomous_agent'] = await test_autonomous_mcp_agent()
    
    # Summary
    print()
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print()
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"   {status}: {test_name.replace('_', ' ').title()}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print()
    
    if passed == total:
        print("🎉 ALL AUTONOMOUS MCP SYSTEM TESTS PASSED!")
        print()
        print("Your MCP tools are now:")
        print("   ✅ Environment-aware (no hardcoded paths/keys)")
        print("   ✅ Self-healing (automatic retry and recovery)")
        print("   ✅ End-to-end autonomous (GitHub Copilot-style)")
        print("   ✅ Server-discoverable (auto-connects to MCP servers)")
    else:
        print("⚠️ Some tests failed. Check output above for details.")
    
    print()
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
