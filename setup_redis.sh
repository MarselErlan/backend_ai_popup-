#!/bin/bash

# Redis Setup Script for Smart Form Fill API
# This script installs and configures Redis locally (no Docker required)

echo "🚀 Setting up Redis locally for Smart Form Fill API..."

# Detect operating system
OS=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "🔍 Detected OS: $OS"

# Install Redis based on OS
if [[ "$OS" == "macos" ]]; then
    echo "🍺 Installing Redis Stack using Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
        exit 1
    fi
    
    brew tap redis-stack/redis-stack
    brew install redis-stack/redis-stack/redis-stack
    redis-stack-server --daemonize yes
    echo "✅ Redis Stack installed and started with Homebrew"
    
elif [[ "$OS" == "linux" ]]; then
    echo "🐧 Installing Redis Stack on Linux..."
    curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
    sudo apt-get update
    sudo apt-get install -y redis-stack-server
    sudo systemctl start redis-stack-server
    sudo systemctl enable redis-stack-server
    echo "✅ Redis Stack installed and started as a service"
    
elif [[ "$OS" == "windows" ]]; then
    echo "🪟 Windows detected. Please install Redis Stack manually:"
    echo "   1. Download Redis Stack from: https://redis.io/download#redis-stack"
    echo "   2. Or use WSL with Linux instructions above"
    echo "   3. Or use Docker: docker run -d -p 6379:6379 redis/redis-stack:latest"
    exit 1
fi

# Test Redis connection
echo "🔬 Testing Redis connection..."
if redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis is running and responding!"
else
    echo "❌ Redis is not responding. Please check the installation."
    exit 1
fi

# Display Redis info
echo "📊 Redis Info:"
echo "   Port: 6379"
echo "   Host: localhost"
echo "   Connection URL: redis://localhost:6379"

# Test Python Redis connection
echo "🐍 Testing Python Redis connection..."
python3 -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print('✅ Python Redis connection successful!')
except Exception as e:
    print(f'❌ Python Redis connection failed: {e}')
    exit(1)
"

# Create a simple health check script
cat > redis_health_check.py << 'EOF'
#!/usr/bin/env python3
"""
Redis Health Check for Smart Form Fill API
Run this to verify Redis is working properly
"""

import redis
import sys

def check_redis():
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test basic operations
        r.ping()
        print("✅ Redis PING successful")
        
        # Test set/get
        r.set("test_key", "test_value")
        value = r.get("test_key")
        if value == "test_value":
            print("✅ Redis SET/GET successful")
        else:
            print("❌ Redis SET/GET failed")
            return False
            
        # Clean up
        r.delete("test_key")
        print("✅ Redis cleanup successful")
        
        # Test Redis info
        info = r.info()
        print(f"✅ Redis version: {info.get('redis_version', 'unknown')}")
        print(f"✅ Redis uptime: {info.get('uptime_in_seconds', 0)} seconds")
        
        return True
        
    except redis.ConnectionError:
        print("❌ Cannot connect to Redis. Is Redis running?")
        return False
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return False

if __name__ == "__main__":
    if check_redis():
        print("\n🎉 Redis is ready for Smart Form Fill API!")
        sys.exit(0)
    else:
        print("\n💡 To start Redis Stack manually:")
        print("   macOS: redis-stack-server --daemonize yes")
        print("   Linux: sudo systemctl start redis-stack-server")
        sys.exit(1)
EOF

chmod +x redis_health_check.py

echo ""
echo "🎉 Redis setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Install Python dependencies: pip install -r requirements.txt"
echo "   2. Set environment variables (OPENAI_API_KEY, DATABASE_URL)"
echo "   3. Run the API server: python main.py"
echo "   4. Test Redis anytime: python redis_health_check.py"
echo ""
echo "🔧 Redis Stack Management:"
echo "   Start:  redis-stack-server --daemonize yes  (macOS) | sudo systemctl start redis-stack-server  (Linux)"
echo "   Stop:   pkill redis-stack-server  (macOS) | sudo systemctl stop redis-stack-server  (Linux)"
echo "   Status: ps aux | grep redis-stack-server (macOS) | systemctl status redis-stack-server (Linux)"
echo ""
echo "📊 Redis is now running on: redis://localhost:6379" 