"""Multi-person council orchestration for persona-based discussions."""
from typing import List, Dict, Any, Tuple, Optional
from .openrouter import query_model
from .config import COUNCIL_MODELS
from .personas import persona_manager, Persona


async def stage1_persona_responses(
    user_query: str,
    persona_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Stage 1: Each persona provides their initial response to the query.
    
    Args:
        user_query: The user's question
        persona_ids: List of persona IDs to consult
    
    Returns:
        List of dicts with 'persona' and 'response' keys
    """
    personas = [persona_manager.get_persona(pid) for pid in persona_ids]
    personas = [p for p in personas if p is not None]
    
    stage1_results = []
    for persona in personas:
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": persona.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Use the first council model for persona responses (can be customized)
        response = await query_model(COUNCIL_MODELS[0], messages)
        
        if response is not None:
            stage1_results.append({
                "persona": persona.model_dump(),
                "response": response.get('content', ''),
                "model": COUNCIL_MODELS[0]
            })
    
    return stage1_results


async def stage2_persona_discussion(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    persona_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Stage 2: Personas discuss and respond to each other's viewpoints.
    
    Args:
        user_query: The original user query
        stage1_results: Initial responses from Stage 1
        persona_ids: List of persona IDs participating
    
    Returns:
        List of discussion responses from each persona
    """
    personas = [persona_manager.get_persona(pid) for pid in persona_ids]
    personas = [p for p in personas if p is not None]
    
    # Build discussion context
    responses_text = "\n\n".join([
        f"**{result['persona']['name']}** ({result['persona']['role'].value}):\n{result['response']}"
        for result in stage1_results
    ])
    
    discussion_prompt = f"""You are engaged in a multi-person discussion about the following question:

Question: {user_query}

Here are the initial responses from all participants:

{responses_text}

Your task is to:
1. Acknowledge the points made by other participants
2. Add your own perspective, expertise, and insights
3. Engage constructively with the discussion
4. Provide your refined position on the question

Remember your role: {personas[0].personality if personas else 'Thoughtful contributor'}
"""

    stage2_results = []
    for persona in personas:
        messages = [
            {"role": "system", "content": persona.system_prompt},
            {"role": "user", "content": discussion_prompt}
        ]
        
        response = await query_model(COUNCIL_MODELS[0], messages)
        
        if response is not None:
            stage2_results.append({
                "persona": persona.model_dump(),
                "response": response.get('content', ''),
                "model": COUNCIL_MODELS[0]
            })
    
    return stage2_results


async def stage3_persona_synthesis(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_persona_id: str = None
) -> Dict[str, Any]:
    """
    Stage 3: Synthesize a final response from the discussion.
    
    Args:
        user_query: The original user query
        stage1_results: Initial responses
        stage2_results: Discussion responses
        chairman_persona_id: Optional persona to act as chairman
    
    Returns:
        Dict with 'model' and 'response' keys (plus 'persona' if using a persona)
    """
    # Build comprehensive synthesis context
    stage1_text = "\n\n".join([
        f"**{result['persona']['name']}** initially said:\n{result['response']}"
        for result in stage1_results
    ])
    
    stage2_text = "\n\n".join([
        f"**{result['persona']['name']}** in discussion:\n{result['response']}"
        for result in stage2_results
    ])
    
    synthesis_prompt = f"""You are synthesizing the results of a multi-person council discussion on the following question:

Question: {user_query}

STAGE 1 - Initial Responses:
{stage1_text}

STAGE 2 - Discussion:
{stage2_text}

Your task is to synthesize all perspectives into a single, comprehensive answer that:
1. Incorporates the key insights from each participant
2. Acknowledges areas of agreement and disagreement
3. Provides a balanced, well-reasoned conclusion
4. References the different viewpoints that contributed to the final answer

Synthesize a final response:"""

    messages = [{"role": "user", "content": synthesis_prompt}]
    
    # Use a regular model for synthesis
    response = await query_model(COUNCIL_MODELS[0], messages)
    
    if response is None:
        return {
            "model": COUNCIL_MODELS[0],
            "response": "Error: Unable to generate final synthesis."
        }
    
    result = {
        "model": COUNCIL_MODELS[0],
        "response": response.get('content', '')
    }
    
    # If using a chairman persona, add that info
    if chairman_persona_id:
        chairman = persona_manager.get_persona(chairman_persona_id)
        if chairman:
            result["persona"] = chairman.model_dump()
    
    return result


async def run_multi_person_council(
    user_query: str,
    persona_ids: List[str],
    chairman_persona_id: str = None
) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete multi-person council process.
    
    Args:
        user_query: The user's question
        persona_ids: List of persona IDs to participate
        chairman_persona_id: Optional persona to synthesize the final answer
    
    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    # Stage 1: Collect initial responses
    stage1_results = await stage1_persona_responses(user_query, persona_ids)
    
    if not stage1_results:
        return [], [], {
            "model": COUNCIL_MODELS[0],
            "response": "No personas responded. Please try again."
        }, {}
    
    # Stage 2: Discussion
    stage2_results = await stage2_persona_discussion(user_query, stage1_results, persona_ids)
    
    # Stage 3: Synthesis
    stage3_result = await stage3_persona_synthesis(
        user_query,
        stage1_results,
        stage2_results,
        chairman_persona_id
    )
    
    # Prepare metadata
    metadata = {
        "persona_ids": persona_ids,
        "chairman_persona_id": chairman_persona_id,
        "participant_count": len(stage1_results)
    }
    
    return stage1_results, stage2_results, stage3_result, metadata


async def generate_multi_person_title(
    user_query: str,
    persona_ids: List[str]
) -> str:
    """
    Generate a title for a multi-person conversation.
    
    Args:
        user_query: The first user message
        persona_ids: List of participating persona IDs
    
    Returns:
        A short title (3-5 words)
    """
    personas = [persona_manager.get_persona(pid) for pid in persona_ids]
    personas = [p for p in personas if p is not None]
    persona_names = ", ".join([p.name for p in personas[:3]])
    if len(personas) > 3:
        persona_names += " et al."
    
    title_prompt = f"""Generate a very short title (3-5 words maximum) for a discussion between {persona_names} about the following question.

Question: {user_query}

The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Title:"""

    messages = [{"role": "user", "content": title_prompt}]
    
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)
    
    if response is None:
        return f"Discussion with {persona_names}"
    
    title = response.get('content', f"Discussion with {persona_names}").strip()
    title = title.strip('"\'')
    if len(title) > 50:
        title = title[:47] + "..."
    
    return title
