#!/usr/bin/env bash
# TAIME v2.0 Production Start Script
# Builds frontend and runs single Uvicorn process

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  TAIME v2.0 Start"
echo "============================================"
echo

# Check if setup has been run
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Build frontend
echo "Building frontend..."
cd apps/web
npm run build
cd ../..
echo "✅ Frontend built to apps/api/static/"

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Initialize database
echo "Initializing database..."
python -c "from taime_api.db.engine import init_db; init_db()"
echo "✅ Database initialized"

# Start server
echo
echo "Starting TAIME v2.0 server..."
echo "→ http://${HOST}:${PORT}"
echo "→ API docs at http://${HOST}:${PORT}/docs"
echo
echo "Press Ctrl+C to stop"
echo

uvicorn taime_api.main:app --host "$HOST" --port "$PORT"
