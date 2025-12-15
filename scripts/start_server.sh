#!/bin/bash
# Start the chapkit model server

set -e

cd "$(dirname "$0")/../disease-model"

# Kill any existing servers
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 2

# Remove old database
rm -f data/chapkit.db

# Start server
echo "Starting server on port 8080..."
uv run uvicorn main:app --port 8080 &
SERVER_PID=$!

sleep 5

# Check health
if curl -s http://localhost:8080/health > /dev/null; then
    echo "Server started successfully (PID: $SERVER_PID)"
    curl -s http://localhost:8080/api/v1/info | python -m json.tool 2>/dev/null || echo ""
else
    echo "Server failed to start"
    exit 1
fi
