#!/usr/bin/env bash
# TAIME v2.0 Setup Script
# Installs Python and Node.js dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  TAIME v2.0 Setup"
echo "============================================"
echo

# Check Python version
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ Python 3.10+ is required. Found: Python $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION"

# Check Node.js version
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js 18+ is required. Found: $(node -v)"
    exit 1
fi
echo "✅ Node.js $(node -v)"

# Create Python virtual environment
echo
echo "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Created .venv"
else
    echo "✅ .venv already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Install backend dependencies
echo
echo "Installing backend dependencies..."
pip install --upgrade pip -q
pip install -e ./apps/api -q
echo "✅ Backend dependencies installed"

# Install frontend dependencies
echo
echo "Installing frontend dependencies..."
cd apps/web
npm install --silent
cd ../..
echo "✅ Frontend dependencies installed"

# Create data directories
echo
echo "Creating data directories..."
mkdir -p data/datasets data/models data/logs
echo "✅ Data directories created"

# Copy .env.example if .env doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
fi

echo
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo
echo "To start the platform:"
echo "  ./start.sh"
echo
echo "For development mode:"
echo "  ./dev.sh"
echo
