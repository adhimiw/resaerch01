# LLM Council - Multi-Person Edition

![header.jpg](header.jpg)

## Overview

LLM Council is a local web application that enables **multiple AI personas** to collaborate and discuss your questions in real-time. Using OpenRouter's unified API, multiple personas can engage in meaningful discussions, providing diverse perspectives while sharing a single API key.

## Features

### Core Features
- **Multi-Person Discussions**: Create and manage custom personas with unique expertise and personalities
- **Collaborative Decision Making**: Personas discuss, debate, and synthesize answers together
- **Single API Key**: All personas use one OpenRouter API key
- **3-Stage Council Process**: Initial responses → Discussion → Final synthesis
- **Streaming Support**: Real-time updates as personas contribute

### Persona System
- **5 Default Personas**:
  - **Dr. Sarah Chen** - Data Scientist (Expert)
  - **Prof. Marcus Webb** - Philosophy Professor (Moderator)
  - **Alex Rivera** - Software Engineer (Analyst)
  - **Jordan Taylor** - Critical Thinker (Skeptic)
  - **Morgan Lee** - Business Strategist (Advocate)

- **Custom Personas**: Create personas with:
  - Name and role
  - Expertise areas
  - Personality description
  - Custom system prompt
  - Custom avatar color

### Persona Roles
- **Expert**: Brings deep domain knowledge
- **Moderator**: Facilitates discussion, ensures balance
- **Skeptic**: Questions assumptions, demands evidence
- **Advocate**: Presents strong positions, opportunities
- **Analyst**: Focuses on data, implementation, trade-offs

## Architecture

```
User Query
    │
    ├── Stage 1: Initial Responses
    │   └── Each persona provides their perspective
    │
    ├── Stage 2: Discussion
    │   └── Personas engage with each other's viewpoints
    │
    └── Stage 3: Synthesis
        └── Final answer incorporating all perspectives
```

## Tech Stack

- **Backend**: FastAPI (Python 3.10+), async httpx, OpenRouter API
- **Frontend**: React + Vite, Server-Sent Events for streaming
- **Storage**: JSON files in `data/conversations/`
- **Package Management**: uv for Python, npm for JavaScript

## Setup

### 1. Install Dependencies

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Key

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

Get your API key from [openrouter.ai](https://openrouter.ai/).

### 3. Configure Models (Optional)

Edit `backend/config.py`:
```python
COUNCIL_MODELS = [
    "openai/gpt-4o",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4-20250514",
]
CHAIRMAN_MODEL = "google/gemini-2.5-flash"
```

## Running the Application

### Option 1: Use the start script
```bash
./start.sh
```

### Option 2: Run manually

**Terminal 1 (Backend):**
```bash
uv run python -m backend.main
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## API Endpoints

### Conversations
- `GET /api/conversations` - List all conversations
- `POST /api/conversations` - Create new conversation
- `GET /api/conversations/{id}` - Get conversation
- `POST /api/conversations/{id}/message` - Send message
- `POST /api/conversations/{id}/message/stream` - Stream response

### Personas
- `GET /api/personas` - List all personas
- `GET /api/personas/{id}` - Get persona
- `POST /api/personas` - Create persona
- `PUT /api/personas/{id}` - Update persona
- `DELETE /api/personas/{id}` - Delete persona
- `POST /api/personas/{id}/activate` - Activate persona
- `POST /api/personas/{id}/deactivate` - Deactivate persona

### Multi-Person Chat
- `POST /api/multi-person/chat` - Run multi-person discussion
- `POST /api/multi-person/chat/stream` - Stream discussion

## Creating a Multi-Person Conversation

1. Click "Personas" tab
2. Select personas to participate (click the checkbox on each)
3. Click "New Conversation"
4. Ask your question
5. Watch the personas discuss and synthesize an answer

## Example Use Cases

1. **Decision Making**: Get multiple perspectives on business decisions
2. **Technical Analysis**: Analyze trade-offs from engineering, data, and business angles
3. **Ethical Debates**: Explore ethical implications with different viewpoints
4. **Creative Problem Solving**: Combine diverse thinking styles

## Vibe Code Alert

This project was enhanced from the original LLM Council to support multi-person discussions. It's designed for exploration and customization - modify personas, adjust prompts, and adapt it to your needs!

## License

MIT
