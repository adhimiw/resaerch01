"""Multi-persona management for LLM Council conversations."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
import json
import os
from .config import DATA_DIR

class PersonaRole(str, Enum):
    EXPERT = "expert"
    MODERATOR = "moderator"
    SKEPTIC = "skeptic"
    ADVOCATE = "advocate"
    ANALYST = "analyst"

class Persona(BaseModel):
    """Represents a persona that can participate in council discussions."""
    id: str
    name: str
    role: PersonaRole
    description: str
    expertise: List[str]
    personality: str  # Brief personality description
    system_prompt: str  # The actual system prompt for this persona
    is_active: bool = True
    created_at: str
    avatar_color: str  # Color for UI display

class PersonaManager:
    """Manages personas for multi-person council discussions."""
    
    DEFAULT_PERSONAS = [
        Persona(
            id="scientist",
            name="Dr. Sarah Chen",
            role=PersonaRole.EXPERT,
            description="Data scientist with expertise in machine learning and statistics",
            expertise=["machine learning", "statistics", "data analysis", "research methodology"],
            personality="Analytical, evidence-based, methodical",
            system_prompt="""You are Dr. Sarah Chen, a senior data scientist with expertise in machine learning and statistics. 
You always approach problems from a data-driven perspective, emphasizing evidence and quantitative analysis.
When discussing topics, you reference relevant research, statistical methods, and empirical findings.
You are thorough and methodical in your reasoning.""",
            created_at=datetime.utcnow().isoformat(),
            avatar_color="#3b82f6"
        ),
        Persona(
            id="philosopher",
            name="Prof. Marcus Webb",
            role=PersonaRole.MODERATOR,
            description="Philosophy professor specializing in ethics and logic",
            expertise=["ethics", "logic", "philosophy", "critical thinking"],
            personality="Thoughtful, questioning, Socratic",
            system_prompt="""You are Professor Marcus Webb, a philosophy professor specializing in ethics and logic.
You approach discussions with careful questioning, logical analysis, and ethical consideration.
You often explore edge cases, assumptions, and philosophical implications of arguments.
Your style is calm, questioning, and intellectually rigorous.""",
            created_at=datetime.utcnow().isoformat(),
            avatar_color="#8b5cf6"
        ),
        Persona(
            id="engineer",
            name="Alex Rivera",
            role=PersonaRole.ANALYST,
            description="Software engineer focused on practical implementation",
            expertise=["software engineering", "system design", "implementation", "optimization"],
            personality="Pragmatic, solution-oriented, practical",
            system_prompt="""You are Alex Rivera, a senior software engineer with expertise in system design and implementation.
You focus on practical solutions, feasibility, and real-world constraints.
When discussing topics, you consider implementation details, trade-offs, and operational concerns.
You are pragmatic and solution-oriented.""",
            created_at=datetime.utcnow().isoformat(),
            avatar_color="#10b981"
        ),
        Persona(
            id="skeptic",
            name="Jordan Taylor",
            role=PersonaRole.SKEPTIC,
            description="Critical thinker who questions assumptions",
            expertise=["critical thinking", "fallacy detection", "evidence evaluation"],
            personality="Skeptical, questioning, thorough",
            system_prompt="""You are Jordan Taylor, a critical thinker known for questioning assumptions and demanding evidence.
You approach claims with healthy skepticism, looking for flaws in reasoning, unsupported assertions, and logical fallacies.
You ask probing questions like "What evidence supports this?" and "What are the counterarguments?"
Your style is rigorous but constructive.""",
            created_at=datetime.utcnow().isoformat(),
            avatar_color="#f59e0b"
        ),
        Persona(
            id="entrepreneur",
            name="Morgan Lee",
            role=PersonaRole.ADVOCATE,
            description="Business strategist focused on opportunities and growth",
            expertise=["business strategy", "opportunity analysis", "market dynamics"],
            personality="Optimistic, opportunity-focused, strategic",
            system_prompt="""You are Morgan Lee, a business strategist known for identifying opportunities and driving growth.
You approach problems from a strategic perspective, considering market dynamics, competitive advantages, and growth potential.
You focus on opportunities, innovation, and strategic positioning.
Your style is energetic, strategic, and opportunity-focused.""",
            created_at=datetime.utcnow().isoformat(),
            avatar_color="#ec4899"
        ),
    ]
    
    def __init__(self):
        self.personas_dir = os.path.join(DATA_DIR, "personas")
        os.makedirs(self.personas_dir, exist_ok=True)
        self._ensure_default_personas()
    
    def _ensure_default_personas(self):
        """Ensure default personas exist."""
        for persona in self.DEFAULT_PERSONAS:
            self.save_persona(persona)
    
    def get_personas_path(self, persona_id: str) -> str:
        """Get the file path for a persona."""
        return os.path.join(self.personas_dir, f"{persona_id}.json")
    
    def save_persona(self, persona: Persona):
        """Save a persona to storage."""
        path = self.get_personas_path(persona.id)
        with open(path, 'w') as f:
            json.dump(persona.model_dump(), f, indent=2)
    
    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """Get a specific persona by ID."""
        path = self.get_personas_path(persona_id)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        return Persona(**data)
    
    def list_personas(self, active_only: bool = True) -> List[Persona]:
        """List all available personas."""
        personas = []
        for filename in os.listdir(self.personas_dir):
            if filename.endswith('.json'):
                path = os.path.join(self.personas_dir, filename)
                with open(path, 'r') as f:
                    data = json.load(f)
                persona = Persona(**data)
                if not active_only or persona.is_active:
                    personas.append(persona)
        return personas
    
    def create_persona(
        self,
        name: str,
        role: PersonaRole,
        description: str,
        expertise: List[str],
        personality: str,
        system_prompt: str,
        avatar_color: str = None
    ) -> Persona:
        """Create a new persona."""
        import uuid
        persona_id = str(uuid.uuid4())[:8]
        
        if avatar_color is None:
            import random
            colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899", "#ef4444", "#06b6d4"]
            avatar_color = random.choice(colors)
        
        persona = Persona(
            id=persona_id,
            name=name,
            role=role,
            description=description,
            expertise=expertise,
            personality=personality,
            system_prompt=system_prompt,
            created_at=datetime.utcnow().isoformat(),
            avatar_color=avatar_color
        )
        
        self.save_persona(persona)
        return persona
    
    def update_persona(self, persona_id: str, **updates) -> Optional[Persona]:
        """Update an existing persona."""
        persona = self.get_persona(persona_id)
        if persona is None:
            return None
        
        for key, value in updates.items():
            if hasattr(persona, key):
                setattr(persona, key, value)
        
        self.save_persona(persona)
        return persona
    
    def delete_persona(self, persona_id: str) -> bool:
        """Delete a persona."""
        path = self.get_personas_path(persona_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
    
    def get_persona_by_role(self, role: PersonaRole) -> List[Persona]:
        """Get all personas with a specific role."""
        return [p for p in self.list_personas() if p.role == role]


# Singleton instance
persona_manager = PersonaManager()
