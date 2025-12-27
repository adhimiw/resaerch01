"""
Test AGI Orchestrator and State Machine

Tests the complete GVU loop with mock data
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.agi.state import create_initial_state, validate_state, AGIState
from core.agi.nodes import (
    profile_dataset_node,
    research_domain_node,
    generate_hypotheses_node,
    should_retry_or_continue
)


def test_create_initial_state():
    """Test initial state creation"""
    state = create_initial_state("test.csv")
    
    assert state["dataset_path"] == "test.csv"
    assert state["attempts"] == 0
    assert state["max_attempts"] == 3
    assert state["is_verified"] == False
    assert len(state["analysis_id"]) > 0
    

def test_validate_state():
    """Test state validation"""
    state = create_initial_state("test.csv")
    
    # Should pass
    assert validate_state(state) == True
    
    # Should fail with missing field
    bad_state = {"dataset_path": "test.csv"}
    with pytest.raises(ValueError):
        validate_state(bad_state)


def test_profile_dataset_node():
    """Test dataset profiling node"""
    # Create test dataset
    import pandas as pd
    import tempfile
    
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': ['x', 'y', 'z', 'x', 'y'],
        'target': [0, 1, 0, 1, 0]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        state = create_initial_state(temp_path)
        result = profile_dataset_node(state)
        
        assert "dataset_profile" in result
        assert result["dataset_profile"]["rows"] == 5
        assert result["dataset_profile"]["columns"] == 3
        assert result["dataset_profile"]["data_type"] in ["timeseries", "tabular"]
    finally:
        os.unlink(temp_path)


def test_research_domain_node():
    """Test domain research node"""
    state = create_initial_state("test.csv")
    result = research_domain_node(state)
    
    assert "domain_knowledge" in result
    assert "domain" in result["domain_knowledge"]


def test_generate_hypotheses_node():
    """Test hypothesis generation node"""
    state = create_initial_state("test.csv")
    result = generate_hypotheses_node(state)
    
    assert "hypotheses" in result
    assert len(result["hypotheses"]) >= 1
    
    # Check hypothesis structure
    h = result["hypotheses"][0]
    assert "id" in h
    assert "statement" in h
    assert "test_strategy" in h


def test_should_retry_or_continue():
    """Test verification decision logic"""
    # Test: verified -> continue
    state = create_initial_state("test.csv")
    state["is_verified"] = True
    assert should_retry_or_continue(state) == "continue"
    
    # Test: not verified, attempts < max -> retry
    state["is_verified"] = False
    state["attempts"] = 1
    state["max_attempts"] = 3
    assert should_retry_or_continue(state) == "retry"
    
    # Test: not verified, attempts >= max -> end
    state["attempts"] = 3
    state["max_attempts"] = 3
    assert should_retry_or_continue(state) == "end"


@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow (requires LangGraph)"""
    try:
        from core.agi.orchestrator import AGIOrchestrator
        import pandas as pd
        import tempfile
        
        # Create test dataset
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': [0, 1] * 50
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Initialize orchestrator
            agi = AGIOrchestrator()
            
            # Run analysis
            result = await agi.analyze(temp_path)
            
            # Verify results
            assert result is not None
            assert result.get("is_verified") == True or result.get("attempts") >= result.get("max_attempts")
            assert "insights" in result
            assert "recommendations" in result
            
            # Check statistics
            stats = agi.get_statistics()
            assert stats["analyses_completed"] == 1
            
        finally:
            os.unlink(temp_path)
            
    except ImportError as e:
        pytest.skip(f"LangGraph not available: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
