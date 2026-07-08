#!/usr/bin/env bash
# TAIME v2.0 Format Script
# Formats code using ruff (backend) and prettier (frontend)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Formatting code..."

# Activate virtual environment if available
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Format backend with ruff
echo "Formatting backend with ruff..."
if command -v ruff &> /dev/null; then
    ruff format apps/api/src apps/api/tests
    ruff check --fix apps/api/src apps/api/tests || true
    echo "✅ Backend formatted"
else
    echo "⚠️  ruff not installed, skipping backend formatting"
fi

# Format frontend with prettier
echo "Formatting frontend with prettier..."
cd apps/web
if [ -f "node_modules/.bin/prettier" ]; then
    npm run format
    echo "✅ Frontend formatted"
else
    echo "⚠️  prettier not installed, skipping frontend formatting"
fi
cd ../..

echo
echo "Formatting complete!"
