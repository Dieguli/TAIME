#!/usr/bin/env bash
# TAIME v2.0 CI Script
# Runs linting and tests for both backend and frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================"
echo "  TAIME v2.0 CI"
echo "============================================"
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Run linting
echo "Running linting..."
./scripts/lint.sh

# Run backend tests
echo
echo "Running backend tests..."
cd apps/api
pytest tests/ -v --tb=short
cd ../..

# Run frontend type check
echo
echo "Running frontend type check..."
cd apps/web
npm run typecheck
cd ../..

# Optionally build the CPU Docker image (opt-in; requires the Docker daemon).
if command -v docker >/dev/null 2>&1 && [ "${CI_DOCKER_BUILD:-}" = "1" ]; then
    echo
    echo "Building CPU Docker image..."
    docker build -f docker/Dockerfile -t taime:cpu .
else
    echo
    echo "Skipping Docker image build (set CI_DOCKER_BUILD=1 with Docker available to enable)."
fi

echo
echo "============================================"
echo "  All checks passed! ✅"
echo "============================================"
