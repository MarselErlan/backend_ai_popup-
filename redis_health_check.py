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
        print("\n💡 To start Redis manually:")
        print("   macOS: brew services start redis")
        print("   Linux: sudo systemctl start redis")
        sys.exit(1) 