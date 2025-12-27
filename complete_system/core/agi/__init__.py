"""
AGI Autonomous Agent Package

Implements Generator-Verifier-Updater (GVU) framework for self-improving
autonomous data science agent.

Based on research paper: "Self-Improving AI Agents through Self-Play" (arxiv 2512.02731)
"""

__version__ = "1.0.0"
__author__ = "AGI Research Team"

from .orchestrator import AGIOrchestrator
from .state import AGIState
from .dspy_agi_agent import DSPyAGIAgent

__all__ = [
    "AGIOrchestrator",
    "AGIState",
    "DSPyAGIAgent",
]
