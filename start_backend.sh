#!/bin/bash
# ============================================================
# Effortless & Safe Backend Runner for macOS
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo " 🎙️ Starting Audio Semantic Segmentation Backend"
echo "============================================================"

# 1. Check virtual environment
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "❌ Error: Virtual environment (.venv) not found."
    echo "   Please create it by running:"
    echo "   uv venv --python 3.11 .venv && uv pip install -r requirements.txt"
    exit 1
fi

# 2. Check & free port 5001 if a stale process is holding it
PORT=5001
OCCUPYING_PID=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$OCCUPYING_PID" ]; then
    echo "⚠️  Port $PORT is currently held by PID $OCCUPYING_PID."
    echo "   Cleaning up stale process..."
    kill -9 $OCCUPYING_PID 2>/dev/null || true
    sleep 1
    echo "   ✅ Port $PORT is now clear."
fi

# 3. Load environment variables (.env) if present
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "🔑 Loading environment from .env"
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# 4. Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# 5. Export port
export PORT=5001

echo "🚀 Launching backend on http://localhost:$PORT..."
echo "   Press Ctrl+C to stop anytime."
echo "------------------------------------------------------------"

exec python backend/app.py

