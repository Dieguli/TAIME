#!/usr/bin/env bash
# TAIME v2.0 Development Script
# Runs backend with reload + Vite dev server concurrently

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  TAIME v2.0 Development Mode"
echo "============================================"
echo

# Check if setup has been run
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Initialize database if needed
echo "Initializing database..."
python -c "from taime_api.db.engine import init_db; init_db()"
echo "✅ Database initialized"

# Cleanup function
cleanup() {
    echo
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

echo
echo "Starting development servers..."
echo "→ Backend:  http://${HOST}:${PORT}"
echo "→ Frontend: http://localhost:5173"
echo "→ API docs: http://${HOST}:${PORT}/docs"
echo
echo "Press Ctrl+C to stop both servers"
echo

# Start backend with reload
uvicorn taime_api.main:app --host "$HOST" --port "$PORT" --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend dev server
cd apps/web
npm run dev &
FRONTEND_PID=$!
cd ../..

# Wait for any process to exit
wait
