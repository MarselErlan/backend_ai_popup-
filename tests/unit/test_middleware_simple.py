#!/usr/bin/env python3
"""
🧪 Simple Middleware Test
Test if DeepTrackingMiddleware is working at all
"""

import requests
import time
import json

def test_middleware():
    """Test if middleware is working"""
    
    print("🧪 Testing DeepTrackingMiddleware")
    print("=" * 60)
    
    # Wait for server to start
    time.sleep(3)
    
    # Make a simple health check request
    try:
        print("📤 Making health check request...")
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"✅ Health check: {response.status_code}")
        
        # Wait a moment for any middleware processing
        time.sleep(2)
        
        # Check if any data was captured
        try:
            with open("tests/reports/integrated_analysis_current.json", "r") as f:
                report = json.load(f)
            
            execution_traces = report.get("execution_traces", [])
            detailed_calls = report.get("detailed_function_calls", [])
            
            print(f"📊 After health check:")
            print(f"   • Execution traces: {len(execution_traces)}")
            print(f"   • Detailed function calls: {len(detailed_calls)}")
            
            if execution_traces:
                print("✅ Middleware is working - execution traces found!")
                return True
            elif detailed_calls:
                print("⚠️  Middleware partially working - function calls found but no execution traces")
                return False
            else:
                print("❌ Middleware not working - no traces found")
                return False
                
        except FileNotFoundError:
            print("❌ No analysis report found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_middleware()
    if success:
        print("\n🎉 Middleware test passed!")
    else:
        print("\n❌ Middleware test failed!") 