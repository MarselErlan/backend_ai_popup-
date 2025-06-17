#!/bin/bash

# AI Form Filler - PostgreSQL Startup Script
echo "🚀 Starting AI Form Filler with PostgreSQL backend..."

# Set PostgreSQL environment variable
export DATABASE_URL="postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup"

# Verify PostgreSQL connection
echo "📊 Testing PostgreSQL connection..."
psql -U ai_popup -d ai_popup -h localhost -c "SELECT 'PostgreSQL connection successful!' as status;" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL connection verified"
else
    echo "❌ PostgreSQL connection failed - please check your database"
    exit 1
fi

# Start the server with PostgreSQL
echo "🔧 Starting uvicorn server with PostgreSQL..."
echo "   Database: postgresql://ai_popup:***@localhost:5432/ai_popup"
echo "   Server: http://127.0.0.1:8000"
echo "   Docs: http://127.0.0.1:8000/docs"
echo ""

uvicorn main:app --reload --host 127.0.0.1 --port 8000 