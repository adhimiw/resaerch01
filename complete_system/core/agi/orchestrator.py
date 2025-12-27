"""
AGI Orchestrator - Main Brain

Implements the Generator-Verifier-Updater (GVU) framework using LangGraph.
Coordinates all components for autonomous data science analysis.

Based on: "Self-Improving AI Agents through Self-Play" (arxiv 2512.02731)
"""

import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not available. Install with: pip install langgraph")

from .state import AGIState, create_initial_state, validate_state, AnalysisResult
from .nodes import (
    profile_dataset_node,
    research_domain_node,
    generate_hypotheses_node,
    plan_analysis_node,
    generate_code_node,
    execute_jupyter_node,
    verify_results_node,
    self_critique_node,
    compare_methods_node,
    synthesize_insights_node,
    update_knowledge_node,
    should_retry_or_continue
)


class AGIOrchestrator:
    """
    Main AGI Orchestrator implementing GVU framework
    
    Generator → Verifier → Updater loop with LangGraph state machine
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AGI Orchestrator
        
        Args:
            config: Configuration dictionary with:
                - llm: LLM configuration
                - mcp_servers: MCP server configurations
                - verification: Verification settings
                - self_improvement: Self-improvement settings
                - observability: Langfuse settings
                - storage: Storage paths
        """
        self.config = config or self._load_default_config()
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None
        
        # Statistics
        self.analyses_completed = 0
        self.total_time = 0.0
        self.kappa_history = []
        
        print("🧠 AGI Orchestrator initialized")
        if not LANGGRAPH_AVAILABLE:
            print("   ⚠️ Running in degraded mode (no LangGraph)")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../config/agi_config.json"
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Fallback defaults
        return {
            "verification": {
                "confidence_threshold": 70,
                "max_retry_attempts": 3
            },
            "self_improvement": {
                "enabled": True,
                "kappa_window": 10
            }
        }
    
    def _build_graph(self) -> Any:
        """
        Build LangGraph state machine
        
        Flow:
        profile_dataset → research_domain → generate_hypotheses → 
        plan_analysis → generate_code → execute_jupyter → verify_results →
        [if verified] compare_methods → synthesize_insights → update_knowledge → END
        [if not verified & attempts < max] self_critique → generate_code (retry)
        [if max attempts] END
        """
        if not LANGGRAPH_AVAILABLE:
            return None
        
        workflow = StateGraph(AGIState)
        
        # Add all nodes
        workflow.add_node("profile_dataset", profile_dataset_node)
        workflow.add_node("research_domain", research_domain_node)
        workflow.add_node("generate_hypotheses", generate_hypotheses_node)
        workflow.add_node("plan_analysis", plan_analysis_node)
        workflow.add_node("generate_code", generate_code_node)
        workflow.add_node("execute_jupyter", execute_jupyter_node)
        workflow.add_node("verify_results", verify_results_node)
        workflow.add_node("self_critique", self_critique_node)
        workflow.add_node("compare_methods", compare_methods_node)
        workflow.add_node("synthesize_insights", synthesize_insights_node)
        workflow.add_node("update_knowledge", update_knowledge_node)
        
        # Set entry point
        workflow.set_entry_point("profile_dataset")
        
        # Add sequential edges (Generator phase)
        workflow.add_edge("profile_dataset", "research_domain")
        workflow.add_edge("research_domain", "generate_hypotheses")
        workflow.add_edge("generate_hypotheses", "plan_analysis")
        workflow.add_edge("plan_analysis", "generate_code")
        workflow.add_edge("generate_code", "execute_jupyter")
        workflow.add_edge("execute_jupyter", "verify_results")
        
        # Conditional edge after verification (Verifier phase)
        workflow.add_conditional_edges(
            "verify_results",
            should_retry_or_continue,
            {
                "retry": "self_critique",      # Failed verification, retry
                "continue": "compare_methods",  # Passed verification, continue
                "end": END                      # Max attempts, end
            }
        )
        
        # Self-critique loops back to generate_code
        workflow.add_edge("self_critique", "generate_code")
        
        # Sequential edges (Synthesis phase)
        workflow.add_edge("compare_methods", "synthesize_insights")
        workflow.add_edge("synthesize_insights", "update_knowledge")
        
        # Update knowledge goes to END (Updater phase complete)
        workflow.add_edge("update_knowledge", END)
        
        # Compile graph
        return workflow.compile()
    
    async def analyze(
        self,
        dataset_path: str,
        objectives: Optional[List[str]] = None,
        max_attempts: int = 3
    ) -> AnalysisResult:
        """
        Perform autonomous analysis on dataset
        
        This is the main entry point for AGI autonomous analysis.
        
        Args:
            dataset_path: Path to dataset CSV file
            objectives: Optional list of user objectives/goals
            max_attempts: Maximum verification retry attempts
            
        Returns:
            AnalysisResult with complete analysis
            
        Raises:
            ValueError: If dataset_path doesn't exist
            RuntimeError: If LangGraph not available
        """
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset not found: {dataset_path}")
        
        print("\n" + "="*70)
        print("🎯 AGI AUTONOMOUS ANALYSIS")
        print("="*70)
        print(f"📁 Dataset: {dataset_path}")
        if objectives:
            print(f"🎯 Objectives: {', '.join(objectives)}")
        print(f"⚙️ Max attempts: {max_attempts}")
        print()
        
        start_time = datetime.now()
        
        # Create initial state
        initial_state = create_initial_state(
            dataset_path=dataset_path,
            user_objectives=objectives,
            max_attempts=max_attempts
        )
        
        # Validate state
        validate_state(initial_state)
        
        # Run state machine
        print("🚀 Running GVU loop...\n")
        try:
            final_state = await self.graph.ainvoke(initial_state)
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            raise
        
        # Calculate total time
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Update statistics
        self.analyses_completed += 1
        self.total_time += total_time
        self.kappa_history.append(final_state.get("kappa", 0.0))
        
        # Print summary
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📊 Confidence: {final_state.get('confidence_score', 0):.0f}/100")
        print(f"🔄 Attempts: {final_state.get('attempts', 0)}")
        print(f"✓  Verified: {final_state.get('is_verified', False)}")
        print(f"🏆 Best method: {final_state.get('best_method', 'N/A')}")
        print(f"💡 Insights: {len(final_state.get('insights', []))}")
        print(f"📈 κ (kappa): {final_state.get('kappa', 0.0):.3f}")
        print(f"📓 Notebook: {final_state.get('final_notebook', 'N/A')}")
        print("="*70 + "\n")
        
        # Convert to AnalysisResult (TODO: implement conversion)
        return final_state
    
    def analyze_sync(
        self,
        dataset_path: str,
        objectives: Optional[List[str]] = None,
        max_attempts: int = 3
    ) -> AnalysisResult:
        """
        Synchronous version of analyze()
        
        Convenience wrapper for non-async contexts.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.analyze(dataset_path, objectives, max_attempts)
        )
    
    async def chat(
        self,
        query: str,
        analysis_id: Optional[str] = None
    ) -> str:
        """
        Conversational interface
        
        Chat with the agent about current or past analyses.
        
        Args:
            query: User question
            analysis_id: Optional analysis ID for context
            
        Returns:
            Agent response
        """
        # TODO: Integrate conversational agent
        return f"Chat response to: {query} (analysis_id: {analysis_id})"
    
    def get_improvement_coefficient(self) -> float:
        """
        Calculate self-improvement coefficient κ (kappa)
        
        κ > 0: Agent is improving
        κ = 0: Agent is plateauing
        κ < 0: Agent is degrading
        
        Returns:
            Kappa value
        """
        if len(self.kappa_history) < 2:
            return 0.0
        
        # Simple linear regression on kappa values
        from scipy.stats import linregress
        x = list(range(len(self.kappa_history)))
        y = self.kappa_history
        
        result = linregress(x, y)
        return result.slope
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            "analyses_completed": self.analyses_completed,
            "total_time_seconds": self.total_time,
            "average_time_seconds": self.total_time / self.analyses_completed if self.analyses_completed > 0 else 0,
            "kappa_history": self.kappa_history,
            "current_kappa": self.get_improvement_coefficient(),
            "timestamp": datetime.now().isoformat()
        }
    
    def visualize_graph(self, output_path: Optional[str] = None):
        """
        Visualize the state machine graph
        
        Args:
            output_path: Path to save visualization (PNG)
        """
        if not self.graph:
            print("⚠️ Graph not available")
            return
        
        try:
            # Try to generate visualization
            from IPython.display import Image, display
            
            img = self.graph.get_graph().draw_mermaid_png()
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(img)
                print(f"✓ Graph saved to {output_path}")
            else:
                display(Image(img))
        except Exception as e:
            print(f"⚠️ Could not visualize graph: {e}")
            print("   Try: pip install pygraphviz")


def quick_analyze(
    dataset_path: str,
    objectives: Optional[List[str]] = None
) -> AnalysisResult:
    """
    Quick analysis convenience function
    
    Args:
        dataset_path: Path to dataset
        objectives: Optional objectives
        
    Returns:
        AnalysisResult
    """
    orchestrator = AGIOrchestrator()
    return orchestrator.analyze_sync(dataset_path, objectives)


if __name__ == "__main__":
    print("AGI Orchestrator - Main Brain")
    print("Usage:")
    print("  from core.agi.orchestrator import AGIOrchestrator")
    print("  agi = AGIOrchestrator()")
    print("  result = await agi.analyze('data.csv')")
    print()
    print("Or use quick_analyze:")
    print("  from core.agi.orchestrator import quick_analyze")
    print("  result = quick_analyze('data.csv')")
