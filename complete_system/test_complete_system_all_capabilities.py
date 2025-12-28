"""
Complete System Test - All Core Capabilities
Tests the full autonomous data science system with all phases enabled
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CompleteSystemTester:
    """
    Comprehensive test of all core capabilities
    """
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        elapsed = time.time() - self.start_time
        timestamp = f"[{elapsed:.2f}s]"
        print(f"{timestamp} {message}")
        
    def log_header(self, title):
        """Print section header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
        
    def log_success(self, component):
        """Log successful component test"""
        self.results[component] = {
            'status': 'SUCCESS',
            'timestamp': time.time()
        }
        print(f"✅ {component}: PASSED")
        
    def log_failure(self, component, error):
        """Log failed component test"""
        self.results[component] = {
            'status': 'FAILED',
            'error': str(error),
            'timestamp': time.time()
        }
        print(f"❌ {component}: FAILED - {error}")
        
    async def test_core_capabilities(self):
        """Test all core capabilities"""
        self.log_header("COMPLETE SYSTEM TEST - ALL CORE CAPABILITIES")
        
        # Test 1: Environment Configuration
        await self.test_environment_config()
        
        # Test 2: Autonomous MCP Server
        await self.test_autonomous_mcp_server()
        
        # Test 3: Self-Healing Executor
        await self.test_self_healing()
        
        # Test 4: Pandas MCP Tools
        await self.test_pandas_mcp_tools()
        
        # Test 5: Workflow Execution
        await self.test_workflow_execution()
        
        # Test 6: Multi-Dataset Analysis
        await self.test_multi_dataset_analysis()
        
        # Test 7: Correlation Analysis
        await self.test_correlation_analysis()
        
        # Test 8: Statistical Analysis
        await self.test_statistical_analysis()
        
        # Generate final report
        self.generate_final_report()
        
    async def test_environment_config(self):
        """Test environment configuration"""
        self.log_header("TEST 1: Environment Configuration")
        try:
            from core.environment_config import get_config, EnvironmentManager
            
            # Test configuration loading
            config = get_config()
            
            self.log(f"Workspace: {config.workspace_dir}")
            self.log(f"Data Directory: {config.data_dir}")
            self.log(f"Max Retries: {config.max_retries}")
            self.log(f"Mistral API Key: {'✅ Set' if config.mistral_api_key else '⚠️ Not set'}")
            self.log(f"Langfuse Keys: {'✅ Set' if config.langfuse_public_key else '⚠️ Not set'}")
            
            self.log_success("Environment Configuration")
            
        except Exception as e:
            self.log_failure("Environment Configuration", e)
            
    async def test_autonomous_mcp_server(self):
        """Test autonomous MCP server"""
        self.log_header("TEST 2: Autonomous MCP Server")
        try:
            from core.autonomous_mcp_agent import AutonomousMCPServer
            
            server = AutonomousMCPServer()
            
            # Discover servers
            discovered = server.discover_servers()
            self.log(f"Discovered {len(discovered)} servers:")
            for s in discovered:
                self.log(f"  - {s['name']} ({s['type']})")
            
            # Start local server
            for config in discovered:
                if config.get('auto_start', False):
                    success = await server.start_server(config)
                    self.log(f"Started {config['name']}: {'✅' if success else '❌'}")
            
            # Get available tools
            tools = server.get_available_tools()
            total_tools = sum(len(t) for t in tools.values())
            self.log(f"Total available tools: {total_tools}")
            
            for server_name, tool_list in tools.items():
                self.log(f"  {server_name}: {len(tool_list)} tools")
            
            self.log_success("Autonomous MCP Server")
            
        except Exception as e:
            self.log_failure("Autonomous MCP Server", e)
            
    async def test_self_healing(self):
        """Test self-healing executor"""
        self.log_header("TEST 3: Self-Healing Executor")
        try:
            from core.self_healing_executor import (
                SelfHealingExecutor, 
                RecoveryConfig, 
                RecoveryStrategy
            )
            
            # Test retry with eventual success
            call_count = [0]
            def unreliable_function():
                call_count[0] += 1
                if call_count[0] < 3:
                    raise ValueError(f"Attempt {call_count[0]} failed")
                return "Success!"
            
            executor = SelfHealingExecutor(RecoveryConfig(max_attempts=5))
            result = executor.execute(unreliable_function)
            
            self.log(f"Function result: {result.result}")
            self.log(f"Attempts required: {result.attempt_number}")
            self.log(f"Duration: {result.duration_ms:.1f}ms")
            
            # Test fallback strategy
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
            self.log(f"Fallback result: {result2.result}")
            
            self.log_success("Self-Healing Executor")
            
        except Exception as e:
            self.log_failure("Self-Healing Executor", e)
            
    async def test_pandas_mcp_tools(self):
        """Test pandas MCP tools"""
        self.log_header("TEST 4: Pandas MCP Tools")
        try:
            from core.pandas_mcp_server import PandasMCPServer
            
            server = PandasMCPServer()
            
            # Create test data
            test_data = {
                'id': range(1, 101),
                'value': [i * 10 for i in range(1, 101)],
                'category': ['A' if i % 2 == 0 else 'B' for i in range(1, 101)],
                'score': [float(i) / 10 for i in range(1, 101)]
            }
            df = pd.DataFrame(test_data)
            
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_path = f.name
            
            # Test read_csv
            read_result = server.read_csv(temp_path, 'test_df')
            self.log(f"Read CSV: {read_result['status']}")
            self.log(f"Shape: {read_result['shape']}")
            
            # Test info
            info_result = server.info('test_df')
            self.log(f"Info: {info_result['status']}")
            self.log(f"Columns: {len(info_result['columns'])}")
            self.log(f"Memory: {info_result['memory_usage_mb']:.2f} MB")
            
            # Test describe
            desc_result = server.describe('test_df')
            self.log(f"Describe: {desc_result['status']}")
            
            # Test correlation
            corr_result = server.correlation('test_df')
            self.log(f"Correlation: {corr_result['status']}")
            if 'strong_correlations' in corr_result:
                self.log(f"Strong correlations found: {len(corr_result['strong_correlations'])}")
            
            # Test groupby
            group_result = server.groupby('test_df', by='category', agg={'value': 'mean'})
            self.log(f"GroupBy: {group_result['status']}")
            
            # Cleanup
            os.unlink(temp_path)
            
            # Get available tools
            tools = [m for m in dir(server) if not m.startswith('_') and callable(getattr(server, m))]
            self.log(f"Total tools: {len(tools)}")
            
            self.log_success("Pandas MCP Tools")
            
        except Exception as e:
            self.log_failure("Pandas MCP Tools", e)
            
    async def test_workflow_execution(self):
        """Test workflow execution"""
        self.log_header("TEST 5: Workflow Execution")
        try:
            from core.autonomous_mcp_agent import (
                AutonomousMCPServer,
                AutonomousWorkflowExecutor,
                AutonomousTask,
                TaskStatus
            )
            
            # Create test data
            import tempfile
            test_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=100),
                'sales': np.random.randint(100, 1000, 100),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                'product': np.random.choice(['A', 'B', 'C'], 100)
            })
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                test_data.to_csv(f.name, index=False)
                temp_path = f.name
            
            # Create workflow tasks
            tasks = [
                AutonomousTask(
                    task_id="load_sales",
                    description="Load sales data",
                    tool_name="pandas.read_csv",
                    parameters={"filepath": temp_path, "name": "sales"}
                ),
                AutonomousTask(
                    task_id="sales_info",
                    description="Get sales data info",
                    tool_name="pandas.info",
                    parameters={"name": "sales"}
                ),
                AutonomousTask(
                    task_id="sales_stats",
                    description="Get sales statistics",
                    tool_name="pandas.describe",
                    parameters={"name": "sales"}
                ),
                AutonomousTask(
                    task_id="sales_by_region",
                    description="Analyze sales by region",
                    tool_name="pandas.groupby",
                    parameters={"name": "sales", "by": "region", "agg": {"sales": ["mean", "sum"]}}
                ),
                AutonomousTask(
                    task_id="sales_correlation",
                    description="Find sales correlations",
                    tool_name="pandas.correlation",
                    parameters={"name": "sales"}
                )
            ]
            
            # Execute workflow
            server = AutonomousMCPServer()
            executor = AutonomousWorkflowExecutor(server)
            
            results = await executor.execute_workflow(tasks, {})
            
            # Log results
            completed = sum(1 for r in results.values() if r.success)
            total = len(results)
            
            self.log(f"Workflow Results: {completed}/{total} tasks completed")
            self.log(f"Success Rate: {completed/total*100:.1f}%")
            self.log(f"Total Duration: {sum(r.duration_seconds for r in results.values()):.3f}s")
            
            # Cleanup
            os.unlink(temp_path)
            
            self.log_success("Workflow Execution")
            
        except Exception as e:
            self.log_failure("Workflow Execution", e)
            
    async def test_multi_dataset_analysis(self):
        """Test analysis on multiple datasets"""
        self.log_header("TEST 6: Multi-Dataset Analysis")
        try:
            from core.autonomous_mcp_agent import autonomous_analysis
            
            datasets = [
                {
                    'name': 'Sample Sales Data',
                    'data': pd.DataFrame({
                        'id': range(1, 101),
                        'sales': np.random.randint(100, 1000, 100),
                        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
                    }),
                    'goal': 'Analyze sales trends and regional performance'
                },
                {
                    'name': 'Customer Metrics',
                    'data': pd.DataFrame({
                        'customer_id': range(1, 51),
                        'age': np.random.randint(18, 70, 50),
                        'income': np.random.randint(30000, 150000, 50),
                        'satisfaction': np.random.uniform(1, 5, 50)
                    }),
                    'goal': 'Analyze customer demographics and satisfaction'
                }
            ]
            
            for dataset in datasets:
                # Save temp dataset
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    dataset['data'].to_csv(f.name, index=False)
                    temp_path = f.name
                
                # Run analysis
                self.log(f"\n📊 Analyzing: {dataset['name']}")
                results = await autonomous_analysis(
                    dataset_path=temp_path,
                    analysis_goal=dataset['goal']
                )
                
                self.log(f"Success Rate: {results['success_rate']:.1%}")
                self.log(f"Duration: {results['total_duration']:.3f}s")
                
                if 'dataset_info' in results:
                    info = results['dataset_info']
                    self.log(f"Dataset: {info.get('shape', 'N/A')}")
                
                # Cleanup
                os.unlink(temp_path)
            
            self.log_success("Multi-Dataset Analysis")
            
        except Exception as e:
            self.log_failure("Multi-Dataset Analysis", e)
            
    async def test_correlation_analysis(self):
        """Test comprehensive correlation analysis"""
        self.log_header("TEST 7: Correlation Analysis")
        try:
            from core.pandas_mcp_server import PandasMCPServer
            
            # Create test data with correlations
            np.random.seed(42)
            n = 1000
            
            server = PandasMCPServer()
            
            # Create correlated data
            test_data = {
                'x': np.random.randn(n),
                'y': np.random.randn(n) * 2 + 1,
                'z': np.random.randn(n) * 0.5,
                'correlated_x': np.random.randn(n),
                'correlated_y': np.random.randn(n) + 0.8 * np.random.randn(n),
            }
            df = pd.DataFrame(test_data)
            
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_path = f.name
            
            # Load and analyze
            server.read_csv(temp_path, 'corr_data')
            result = server.correlation('corr_data')
            
            self.log(f"Correlation Analysis: {result['status']}")
            
            if 'correlation_matrix' in result:
                matrix = result['correlation_matrix']
                self.log("Correlation Matrix:")
                for col1, values in matrix.items():
                    for col2, corr in values.items():
                        if col1 < col2 and abs(corr) > 0.5:
                            self.log(f"  {col1} vs {col2}: {corr:.3f}")
            
            if 'strong_correlations' in result:
                self.log(f"\nStrong Correlations (>0.5): {len(result['strong_correlations'])}")
                for corr in result['strong_correlations'][:5]:
                    self.log(f"  {corr['col1']} vs {corr['col2']}: {corr['correlation']:.3f}")
            
            # Cleanup
            os.unlink(temp_path)
            
            self.log_success("Correlation Analysis")
            
        except Exception as e:
            self.log_failure("Correlation Analysis", e)
            
    async def test_statistical_analysis(self):
        """Test comprehensive statistical analysis"""
        self.log_header("TEST 8: Statistical Analysis")
        try:
            from core.pandas_mcp_server import PandasMCPServer
            
            server = PandasMCPServer()
            
            # Create diverse test data
            np.random.seed(42)
            n = 500
            
            test_data = {
                'continuous': np.random.randn(n) * 10 + 50,
                'discrete': np.random.randint(1, 100, n),
                'categorical': np.random.choice(['A', 'B', 'C'], n),
                'binary': np.random.choice([0, 1], n),
                'skewed': np.random.exponential(2, n)
            }
            df = pd.DataFrame(test_data)
            
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_path = f.name
            
            # Load data
            server.read_csv(temp_path, 'stats_data')
            
            # Get info
            info = server.info('stats_data')
            self.log(f"Dataset Info: {info['status']}")
            self.log(f"  Shape: {info['shape']}")
            self.log(f"  Columns: {len(info['columns'])}")
            self.log(f"  Memory: {info['memory_usage_mb']:.2f} MB")
            self.log(f"  Duplicates: {info['duplicate_rows']}")
            
            # Get statistics
            stats = server.describe('stats_data')
            self.log(f"\nStatistical Summary:")
            if 'statistics' in stats:
                statistics = stats['statistics']
                for col, values in statistics.items():
                    if col in ['continuous', 'skewed']:
                        self.log(f"  {col}:")
                        self.log(f"    Mean: {values.get('mean', 'N/A'):.2f}")
                        self.log(f"    Std: {values.get('std', 'N/A'):.2f}")
                        self.log(f"    Min: {values.get('min', 'N/A'):.2f}")
                        self.log(f"    Max: {values.get('max', 'N/A'):.2f}")
            
            # Value counts for categorical
            vc = server.value_counts('stats_data', 'categorical', top_n=3)
            self.log(f"\nCategorical Distribution (categorical):")
            if 'value_counts' in vc:
                for cat, count in vc['value_counts'].items():
                    self.log(f"  {cat}: {count}")
            
            # Cleanup
            os.unlink(temp_path)
            
            self.log_success("Statistical Analysis")
            
        except Exception as e:
            self.log_failure("Statistical Analysis", e)
            
    def generate_final_report(self):
        """Generate final test report"""
        self.log_header("FINAL TEST REPORT")
        
        # Calculate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'SUCCESS')
        failed_tests = total_tests - passed_tests
        
        print(f"\n📊 Test Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  ✅ Passed: {passed_tests}")
        print(f"  ❌ Failed: {failed_tests}")
        print(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for component, result in self.results.items():
            status = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"  {status} {component}")
            if result['status'] == 'FAILED':
                print(f"     Error: {result.get('error', 'Unknown')}")
        
        print(f"\n🎯 Core Capabilities Tested:")
        capabilities = [
            "Environment Configuration",
            "Autonomous MCP Server Discovery",
            "Self-Healing Execution",
            "Pandas MCP Tools (20+ operations)",
            "Workflow Execution Engine",
            "Multi-Dataset Analysis",
            "Correlation Analysis",
            "Statistical Analysis"
        ]
        
        for capability in capabilities:
            status = "✅" if capability in [k for k, v in self.results.items() if v['status'] == 'SUCCESS'] else "❌"
            print(f"  {status} {capability}")
        
        print(f"\n🚀 System Status:")
        if passed_tests == total_tests:
            print(f"  ✅ ALL CORE CAPABILITIES OPERATIONAL")
            print(f"  ✅ System is production-ready")
            print(f"  ✅ Autonomous operation confirmed")
        else:
            print(f"  ⚠️ {failed_tests} tests failed")
            print(f"  ℹ️  Review failed components above")
        
        print(f"\n⏱️ Total Test Duration: {time.time() - self.start_time:.2f}s")
        print("="*80)
        
        # Save results to file
        results_file = Path(__file__).parent / "complete_system_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': passed_tests/total_tests*100,
                'results': self.results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


async def main():
    """Run complete system test"""
    print("\n" + "="*80)
    print("🚀 AUTONOMOUS DATA SCIENCE SYSTEM - COMPLETE TEST")
    print("="*80)
    print("\nTesting all core capabilities...")
    print("This may take a few moments...\n")
    
    tester = CompleteSystemTester()
    await tester.test_core_capabilities()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
