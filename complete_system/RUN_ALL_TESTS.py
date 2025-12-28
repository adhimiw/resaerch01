"""
EXECUTE ALL TESTS - Complete System Validation
Runs all tests in sequence and generates final report
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

def run_test(script_name, description, timeout=900):
    """Run a test script and capture results"""
    print("="*100)
    print(f"🧪 RUNNING: {description}")
    print("="*100)
    
    script_path = Path(__file__).parent / 'tests' / script_name
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        
        return {
            'test': script_name,
            'description': description,
            'success': success,
            'duration_seconds': duration,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
    except subprocess.TimeoutExpired:
        print(f"⚠️  Test timed out after {timeout} seconds")
        return {
            'test': script_name,
            'description': description,
            'success': False,
            'duration_seconds': timeout,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"❌ Error running test: {e}")
        import traceback
        traceback.print_exc()
        return {
            'test': script_name,
            'description': description,
            'success': False,
            'error': str(e)
        }


def main():
    print("="*100)
    print("🚀 COMPLETE SYSTEM TEST EXECUTION")
    print("="*100)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    # Define test sequence
    tests = [
        ('test_spotify_integration.py', 'Test 1: DSPy with Spotify Data (15 min)', 900),
        ('test_multi_dataset.py', 'Test 2: Multi-Dataset Universal Capability (60 min)', 3600),
    ]
    
    results = []
    
    # Run all tests
    for script, desc, timeout in tests:
        result = run_test(script, desc, timeout)
        results.append(result)
    
    # Generate final report
    print("\n" + "="*100)
    print("📊 FINAL TEST REPORT")
    print("="*100)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r.get('duration_seconds', 0) for r in results)
    
    print(f"\n📈 Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   ✅ Passed: {passed_tests}")
    print(f"   ❌ Failed: {failed_tests}")
    print(f"   ⏱️  Total Duration: {total_duration/60:.1f} minutes")
    print(f"   📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\n📝 Test Details:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = result.get('duration_seconds', 0)
        print(f"   {status} | {result['description']} | {duration/60:.1f} min")
    
    # Save report
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    report_file = results_dir / f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests/total_tests)*100,
                'total_duration_minutes': total_duration/60
            },
            'tests': results
        }, f, indent=2)
    
    print(f"\n📁 Full report saved to: {report_file}")
    print(f"📊 View Langfuse dashboard: https://cloud.langfuse.com")
    
    print("\n" + "="*100)
    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
    else:
        print(f"⚠️  {failed_tests} TEST(S) FAILED - REVIEW REQUIRED")
    print("="*100)
    
    return results


if __name__ == "__main__":
    results = main()
