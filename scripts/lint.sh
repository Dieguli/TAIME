#!/usr/bin/env bash
# TAIME v2.0 Lint Script
# Lints code using ruff (backend) and eslint (frontend)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Running linters..."

# Activate virtual environment if available
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

EXIT_CODE=0

# Lint backend with ruff
echo "Linting backend with ruff..."
if command -v ruff &> /dev/null; then
    ruff check apps/api/src apps/api/tests || EXIT_CODE=1
    ruff format --check apps/api/src apps/api/tests || EXIT_CODE=1
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Backend lint passed"
    fi
else
    echo "⚠️  ruff not installed, skipping backend linting"
fi

# Lint frontend with eslint
echo "Linting frontend with eslint..."
cd apps/web
if [ -f "node_modules/.bin/eslint" ]; then
    npm run lint || EXIT_CODE=1
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Frontend lint passed"
    fi
else
    echo "⚠️  eslint not installed, skipping frontend linting"
fi
cd ../..

if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "❌ Linting failed"
    exit 1
fi

echo
echo "All linting passed! ✅"
