#!/bin/bash

echo "🚀 Smart Form Fill API - Comprehensive Testing Suite"
echo "=================================================="

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found in current directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3"
    exit 1
fi

# Install required packages if not present
echo "📦 Checking required packages..."
python3 -c "import aiohttp, ast, json" 2>/dev/null || {
    echo "📦 Installing required packages..."
    pip install aiohttp
}

# Check if server is running
echo "🔍 Checking if server is running on localhost:8000..."
curl -s http://localhost:8000/health > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ Server is running - starting tests..."
else
    echo "⚠️  Server not detected. Starting server in background..."
    
    # Start server in background
    python3 main.py &
    SERVER_PID=$!
    echo "🚀 Server started with PID: $SERVER_PID"
    
    # Wait for server to start
    echo "⏳ Waiting for server to start..."
    sleep 5
    
    # Check if server started successfully
    curl -s http://localhost:8000/health > /dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Server started successfully"
    else
        echo "❌ Failed to start server"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
fi

# Run the comprehensive test
echo "🧪 Running comprehensive test suite..."
python3 test_main_comprehensive.py

# If we started the server, clean it up
if [ ! -z "$SERVER_PID" ]; then
    echo "🧹 Cleaning up - stopping server..."
    kill $SERVER_PID 2>/dev/null
    echo "✅ Server stopped"
fi

echo "✅ Testing complete!" 