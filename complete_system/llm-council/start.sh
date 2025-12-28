#!/bin/bash
# LLM Council - Multi-Person Edition - Start script
echo "=========================================="
echo "  LLM Council - Multi-Person Edition"
echo "=========================================="
echo ""

# Check for API key
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "Please create a .env file with your OPENROUTER_API_KEY"
    echo "Example: OPENROUTER_API_KEY=sk-or-v1-..."
    echo ""
    exit 1
fi

# Start backend
echo "Starting backend on http://localhost:8001..."
uv run python -m backend.main &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend
echo "Starting frontend on http://localhost:5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "✓ LLM Council is running!"
echo ""
echo "  Backend:  http://localhost:8001"
echo "  Frontend: http://localhost:5173"
echo ""
echo "  Multi-Person Features:"
echo "  - Click 'Personas' tab to manage personas"
echo "  - Select multiple personas for discussions"
echo "  - Create custom personas with expertise"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=========================================="

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
