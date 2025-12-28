"""FastAPI backend for LLM Council with Multi-Person Support."""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import json
import asyncio
from . import storage
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings
from .personas import persona_manager, PersonaRole
from .multi_person_council import run_multi_person_council, generate_multi_person_title

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Request/Response Models ====================

class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    use_multi_person: bool = False
    persona_ids: List[str] = []

class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str

class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int
    use_multi_person: bool = False
    persona_ids: List[str] = []

class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]
    use_multi_person: bool = False
    persona_ids: List[str] = []

class PersonaCreateRequest(BaseModel):
    """Request to create a new persona."""
    name: str
    role: PersonaRole
    description: str
    expertise: List[str]
    personality: str
    system_prompt: str
    avatar_color: Optional[str] = None

class PersonaResponse(BaseModel):
    """Persona data for API responses."""
    id: str
    name: str
    role: PersonaRole
    description: str
    expertise: List[str]
    personality: str
    is_active: bool
    created_at: str
    avatar_color: str

class MultiPersonMessageRequest(BaseModel):
    """Request to send a message in a multi-person conversation."""
    content: str
    persona_ids: List[str]
    chairman_persona_id: Optional[str] = None

# ==================== Health Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API", "mode": "multi-person"}

@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "ok",
        "service": "LLM Council API",
        "version": "2.0.0",
        "features": ["multi-council", "multi-person"]
    }

# ==================== Conversation Endpoints ====================

@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()

@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(
        conversation_id,
        use_multi_person=request.use_multi_person,
        persona_ids=request.persona_ids
    )
    return conversation

@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0
    
    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # Generate title if first message
    if is_first_message:
        if conversation.get("use_multi_person", False) and conversation.get("persona_ids"):
            title = await generate_multi_person_title(request.content, conversation["persona_ids"])
        else:
            title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the council process
    if conversation.get("use_multi_person", False) and conversation.get("persona_ids"):
        # Multi-person council
        stage1_results, stage2_results, stage3_result, metadata = await run_multi_person_council(
            request.content,
            conversation["persona_ids"]
        )
    else:
        # Standard council
        stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
            request.content
        )

    # Add assistant message
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
        metadata
    )

    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }

@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the council process.
    Returns Server-Sent Events as each stage completes.
    """
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            storage.add_user_message(conversation_id, request.content)

            # Start title generation in parallel
            title_task = None
            if is_first_message:
                if conversation.get("use_multi_person", False) and conversation.get("persona_ids"):
                    title_task = asyncio.create_task(
                        generate_multi_person_title(request.content, conversation["persona_ids"])
                    )
                else:
                    title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Determine which council to use
            use_multi_person = conversation.get("use_multi_person", False)
            persona_ids = conversation.get("persona_ids", [])

            # Stage 1
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            
            if use_multi_person and persona_ids:
                from .multi_person_council import stage1_persona_responses
                stage1_results = await stage1_persona_responses(request.content, persona_ids)
            else:
                stage1_results = await stage1_collect_responses(request.content)
            
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            
            if use_multi_person and persona_ids:
                from .multi_person_council import stage2_persona_discussion
                stage2_results = await stage2_persona_discussion(request.content, stage1_results, persona_ids)
                label_to_model = {p['name']: p['name'] for p in stage1_results}
                aggregate_rankings = []
            else:
                stage2_results, label_to_model = await stage2_collect_rankings(request.content, stage1_results)
                aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            
            yield f"data: {json.dumps({
                'type': 'stage2_complete',
                'data': stage2_results,
                'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}
            })}\n\n"

            # Stage 3
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            
            if use_multi_person and persona_ids:
                from .multi_person_council import stage3_persona_synthesis
                stage3_result = await stage3_persona_synthesis(
                    request.content, stage1_results, stage2_results
                )
            else:
                stage3_result = await stage3_synthesize_final(
                    request.content, stage1_results, stage2_results
                )
            
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Complete title
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save message
            metadata = {
                "label_to_model": label_to_model if not use_multi_person else {},
                "aggregate_rankings": aggregate_rankings,
                "persona_ids": persona_ids if use_multi_person else [],
                "use_multi_person": use_multi_person
            }
            storage.add_assistant_message(
                conversation_id, stage1_results, stage2_results, stage3_result, metadata
            )

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# ==================== Persona Endpoints ====================

@app.get("/api/personas", response_model=List[PersonaResponse])
async def list_personas(active_only: bool = Query(True, description="Only return active personas")):
    """List all available personas."""
    personas = persona_manager.list_personas(active_only=active_only)
    return [PersonaResponse(**p.model_dump()) for p in personas]

@app.get("/api/personas/{persona_id}", response_model=PersonaResponse)
async def get_persona(persona_id: str):
    """Get a specific persona."""
    persona = persona_manager.get_persona(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return PersonaResponse(**persona.model_dump())

@app.post("/api/personas", response_model=PersonaResponse)
async def create_persona(request: PersonaCreateRequest):
    """Create a new persona."""
    persona = persona_manager.create_persona(
        name=request.name,
        role=request.role,
        description=request.description,
        expertise=request.expertise,
        personality=request.personality,
        system_prompt=request.system_prompt,
        avatar_color=request.avatar_color
    )
    return PersonaResponse(**persona.model_dump())

@app.put("/api/personas/{persona_id}", response_model=PersonaResponse)
async def update_persona(persona_id: str, request: PersonaCreateRequest):
    """Update an existing persona."""
    updates = request.model_dump(exclude_unset=True)
    persona = persona_manager.update_persona(persona_id, **updates)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return PersonaResponse(**persona.model_dump())

@app.delete("/api/personas/{persona_id}")
async def delete_persona(persona_id: str):
    """Delete a persona."""
    success = persona_manager.delete_persona(persona_id)
    if not success:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"status": "deleted", "id": persona_id}

@app.post("/api/personas/{persona_id}/activate")
async def activate_persona(persona_id: str):
    """Activate a persona."""
    persona = persona_manager.update_persona(persona_id, is_active=True)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"status": "activated", "id": persona_id}

@app.post("/api/personas/{persona_id}/deactivate")
async def deactivate_persona(persona_id: str):
    """Deactivate a persona."""
    persona = persona_manager.update_persona(persona_id, is_active=False)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"status": "deactivated", "id": persona_id}

@app.get("/api/personas/roles")
async def list_persona_roles():
    """List all available persona roles."""
    return {"roles": [role.value for role in PersonaRole]}

# ==================== Multi-Person Chat ====================

@app.post("/api/multi-person/chat")
async def multi_person_chat(request: MultiPersonMessageRequest):
    """
    Run a multi-person council discussion.
    Creates a temporary conversation and returns results immediately.
    """
    if not request.persona_ids:
        raise HTTPException(status_code=400, detail="At least one persona is required")

    stage1_results, stage2_results, stage3_result, metadata = await run_multi_person_council(
        request.content,
        request.persona_ids,
        request.chairman_persona_id
    )

    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }

@app.post("/api/multi-person/chat/stream")
async def multi_person_chat_stream(request: MultiPersonMessageRequest):
    """
    Run a multi-person council discussion with streaming.
    """
    if not request.persona_ids:
        raise HTTPException(status_code=400, detail="At least one persona is required")

    async def event_generator():
        try:
            from .multi_person_council import (
                stage1_persona_responses,
                stage2_persona_discussion,
                stage3_persona_synthesis
            )

            # Stage 1
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_persona_responses(request.content, request.persona_ids)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results = await stage2_persona_discussion(request.content, stage1_results, request.persona_ids)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results})}\n\n"

            # Stage 3
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_persona_synthesis(
                request.content, stage1_results, stage2_results, request.chairman_persona_id
            )
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
