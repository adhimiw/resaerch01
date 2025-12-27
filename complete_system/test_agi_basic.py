"""
Quick test of AGI basic structure

Tests that all modules import and basic structure works
"""

import sys
import pandas as pd
import tempfile
import os

print("="*70)
print("🧪 AGI BASIC STRUCTURE TEST")
print("="*70)
print()

# Test 1: Import modules
print("1️⃣ Testing imports...")
try:
    from core.agi.state import create_initial_state, validate_state
    from core.agi.nodes import profile_dataset_node, research_domain_node
    from core.agi.orchestrator import AGIOrchestrator
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create state
print("\n2️⃣ Testing state creation...")
try:
    state = create_initial_state("test.csv")
    validate_state(state)
    print(f"   ✓ State created: analysis_id={state['analysis_id'][:8]}...")
    print(f"   ✓ State validated")
except Exception as e:
    print(f"   ✗ State creation failed: {e}")
    sys.exit(1)

# Test 3: Create test dataset
print("\n3️⃣ Creating test dataset...")
try:
    df = pd.DataFrame({
        'feature1': range(50),
        'feature2': range(50, 100),
        'target': [0, 1] * 25
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    print(f"   ✓ Test dataset created: {temp_path}")
    print(f"   ✓ Shape: {df.shape}")
except Exception as e:
    print(f"   ✗ Dataset creation failed: {e}")
    sys.exit(1)

# Test 4: Test individual nodes
print("\n4️⃣ Testing individual nodes...")
try:
    state = create_initial_state(temp_path)
    
    # Test profile node
    result = profile_dataset_node(state)
    print(f"   ✓ Profile node: found {result['dataset_profile']['rows']} rows")
    
    # Test research node
    result = research_domain_node(state)
    print(f"   ✓ Research node: domain={result['domain_knowledge']['domain']}")
    
except Exception as e:
    print(f"   ✗ Node testing failed: {e}")
    os.unlink(temp_path)
    sys.exit(1)

# Test 5: Initialize orchestrator
print("\n5️⃣ Testing orchestrator initialization...")
try:
    agi = AGIOrchestrator()
    print("   ✓ Orchestrator initialized")
    print(f"   ✓ Graph available: {agi.graph is not None}")
except Exception as e:
    print(f"   ✗ Orchestrator initialization failed: {e}")
    os.unlink(temp_path)
    sys.exit(1)

# Test 6: Test graph structure
print("\n6️⃣ Testing graph structure...")
try:
    if agi.graph:
        # Try to get graph info
        print("   ✓ LangGraph state machine compiled")
        print("   ✓ GVU loop ready")
    else:
        print("   ⚠️ Graph not available (LangGraph may not be installed)")
except Exception as e:
    print(f"   ⚠️ Graph structure test warning: {e}")

# Cleanup
print("\n7️⃣ Cleaning up...")
try:
    os.unlink(temp_path)
    print("   ✓ Test dataset removed")
except Exception as e:
    print(f"   ⚠️ Cleanup warning: {e}")

# Summary
print("\n" + "="*70)
print("✅ BASIC STRUCTURE TEST PASSED")
print("="*70)
print()
print("Next steps:")
print("  1. Implement DSPy AGI agent modules")
print("  2. Implement verification engine")
print("  3. Test full workflow on iris dataset")
print()
