#!/usr/bin/env python3
"""
🧪 Test Execution Traces
Verify that DeepTrackingMiddleware is capturing execution traces
"""

import requests
import json
import time
import uuid

def test_execution_traces():
    """Test that execution traces are being captured"""
    
    print("🧪 Testing Execution Traces with DeepTrackingMiddleware")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test if server is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server is not running! Start it with: uvicorn main:app --reload")
            return False
    except requests.exceptions.RequestException:
        print("❌ Server is not running! Start it with: uvicorn main:app --reload")
        return False
    
    print("✅ Server is running")
    
    # Create a test user and login
    test_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    test_password = "testpass123"
    
    print(f"📝 Creating test user: {test_email}")
    
    # Register user
    register_data = {
        "email": test_email,
        "password": test_password
    }
    
    register_response = requests.post(f"{base_url}/api/simple/register", json=register_data)
    if register_response.status_code != 200:
        print(f"❌ Registration failed: {register_response.text}")
        return False
    
    user_data = register_response.json()
    user_id = user_data["user_id"]
    print(f"✅ User registered with ID: {user_id}")
    
    # Create session
    session_data = {"user_id": user_id}
    session_response = requests.post(f"{base_url}/api/session/create", json=session_data)
    if session_response.status_code != 200:
        print(f"❌ Session creation failed: {session_response.text}")
        return False
    
    session_info = session_response.json()
    session_id = session_info["session_id"]
    print(f"✅ Session created: {session_id}")
    
    # Make API calls to generate execution traces
    headers = {"Authorization": session_id}
    
    print("🚀 Making API calls to generate execution traces...")
    
    test_requests = [
        {"label": "Full Name", "url": "https://example.com/form1"},
        {"label": "Email Address", "url": "https://example.com/form2"},
        {"label": "Phone Number", "url": "https://example.com/form3"}
    ]
    
    for i, request_data in enumerate(test_requests, 1):
        print(f"📤 Making API call {i}/3: {request_data['label']}")
        
        response = requests.post(
            f"{base_url}/api/generate-field-answer",
            json=request_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API call {i} successful: {result['answer'][:50]}...")
        else:
            print(f"❌ API call {i} failed: {response.status_code} - {response.text}")
    
    print("\n🔍 Checking for execution traces...")
    
    # Check the analysis report
    try:
        with open("tests/reports/integrated_analysis_current.json", "r") as f:
            report = json.load(f)
        
        execution_traces = report.get("execution_traces", [])
        detailed_calls = report.get("detailed_function_calls", [])
        
        print(f"📊 Analysis Results:")
        print(f"   • Execution traces: {len(execution_traces)}")
        print(f"   • Detailed function calls: {len(detailed_calls)}")
        
        if execution_traces:
            print(f"✅ SUCCESS! Found {len(execution_traces)} execution traces")
            for i, trace in enumerate(execution_traces[:3], 1):
                print(f"   🎯 Trace {i}: {trace.get('endpoint', 'Unknown')} ({trace.get('total_time', 0):.3f}s)")
        else:
            print("❌ No execution traces found")
        
        if detailed_calls:
            print(f"✅ Found {len(detailed_calls)} detailed function calls")
            for i, call in enumerate(detailed_calls[:5], 1):
                print(f"   🔧 Call {i}: {call.get('function_name', 'Unknown')} ({call.get('execution_time', 0):.3f}s)")
        
        return len(execution_traces) > 0
        
    except FileNotFoundError:
        print("❌ Analysis report not found")
        return False
    except Exception as e:
        print(f"❌ Error reading analysis report: {e}")
        return False

if __name__ == "__main__":
    success = test_execution_traces()
    if success:
        print("\n🎉 Test completed successfully! Check the HTML report for execution traces.")
    else:
        print("\n❌ Test failed. Check server logs for issues.") 