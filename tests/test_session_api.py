#!/usr/bin/env python3
"""
🔑 Test Simple Session Management API
Demonstrates the ultra-simple authentication flow for browser extensions
"""
import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_simple_session_flow():
    """Test the complete simple session flow"""
    print("🔑 Testing Simple Session Management API")
    print("=" * 60)
    
    # Step 1: Register a user (or login existing)
    print("\n📝 Step 1: Register User")
    register_data = {
        "email": f"test_session_{datetime.now().strftime('%H%M%S')}@example.com",
        "password": "password123"
    }
    
    response = requests.post(f"{BASE_URL}/api/simple/register", json=register_data)
    if response.status_code == 200:
        user_data = response.json()
        print(f"✅ User registered successfully!")
        print(f"   📧 Email: {user_data['email']}")
        print(f"   🆔 User ID: {user_data['user_id']}")
        user_id = user_data['user_id']
    else:
        print(f"❌ Registration failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return
    
    # Step 2: Create a session
    print("\n🔑 Step 2: Create Session")
    session_data = {
        "user_id": user_id,
        "device_info": "Chrome Extension v1.0 - Test Browser"
    }
    
    response = requests.post(f"{BASE_URL}/api/session/create", json=session_data)
    if response.status_code == 200:
        session_response = response.json()
        print(f"✅ Session created successfully!")
        print(f"   🎫 Session ID: {session_response['session_id']}")
        print(f"   💾 Store this session_id in browser extension storage")
        session_id = session_response['session_id']
    else:
        print(f"❌ Session creation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return
    
    # Step 3: Get current user by session_id
    print("\n👤 Step 3: Get Current User Info")
    response = requests.get(f"{BASE_URL}/api/session/current/{session_id}")
    if response.status_code == 200:
        current_user = response.json()
        print(f"✅ Current user retrieved successfully!")
        print(f"   📧 Email: {current_user['email']}")
        print(f"   🆔 User ID: {current_user['user_id']}")
        print(f"   🎫 Session ID: {current_user['session_id']}")
        print(f"   📱 Device: {current_user['device_info']}")
        print(f"   🕐 Created: {current_user['created_at']}")
        print(f"   🕐 Last Used: {current_user['last_used_at']}")
    else:
        print(f"❌ Get current user failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return
    
    # Step 4: Use the user_id for form filling
    print("\n🎯 Step 4: Test Form Filling with User ID")
    field_data = {
        "label": "Your Full Name",
        "url": "https://example.com/job-application",
        "user_id": user_id
    }
    
    response = requests.post(f"{BASE_URL}/api/generate-field-answer", json=field_data)
    if response.status_code == 200:
        field_response = response.json()
        print(f"✅ Field answer generated successfully!")
        print(f"   💬 Answer: {field_response['answer']}")
        print(f"   📊 Source: {field_response['data_source']}")
        print(f"   🔍 Reasoning: {field_response['reasoning']}")
    else:
        print(f"❌ Field answer failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    # Step 5: Logout (deactivate session)
    print("\n🚪 Step 5: Logout Session")
    response = requests.delete(f"{BASE_URL}/api/session/{session_id}")
    if response.status_code == 200:
        logout_response = response.json()
        print(f"✅ Session logged out successfully!")
        print(f"   📝 Status: {logout_response['status']}")
        print(f"   💬 Message: {logout_response['message']}")
    else:
        print(f"❌ Logout failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    # Step 6: Try to use deactivated session (should fail)
    print("\n🚫 Step 6: Test Deactivated Session")
    response = requests.get(f"{BASE_URL}/api/session/current/{session_id}")
    if response.status_code == 404:
        print(f"✅ Deactivated session correctly rejected!")
        print(f"   📝 Status: {response.status_code}")
    else:
        print(f"❌ Deactivated session still works (unexpected)")
        print(f"   Status: {response.status_code}")

def test_browser_extension_flow():
    """Simulate browser extension usage"""
    print("\n" + "=" * 60)
    print("🌐 Browser Extension Simulation")
    print("=" * 60)
    
    print("""
📋 Browser Extension Implementation:

// 1. One-time setup (when extension is installed)
async function setupUser() {
    // Register user
    const registerResponse = await fetch('/api/simple/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            email: 'user@example.com',
            password: 'password123'
        })
    });
    const userData = await registerResponse.json();
    
    // Create session
    const sessionResponse = await fetch('/api/session/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_id: userData.user_id,
            device_info: 'Chrome Extension v1.0'
        })
    });
    const sessionData = await sessionResponse.json();
    
    // Store session_id permanently
    chrome.storage.local.set({
        session_id: sessionData.session_id,
        user_id: userData.user_id
    });
}

// 2. Get current user (when extension starts)
async function getCurrentUser() {
    const { session_id } = await chrome.storage.local.get(['session_id']);
    const response = await fetch(`/api/session/current/${session_id}`);
    const user = await response.json();
    return user;
}

// 3. Fill form fields (main functionality)
async function fillField(inputElement) {
    const { user_id } = await chrome.storage.local.get(['user_id']);
    
    const response = await fetch('/api/generate-field-answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            label: inputElement.placeholder,
            url: window.location.href,
            user_id: user_id  // ← Simple!
        })
    });
    
    const result = await response.json();
    inputElement.value = result.answer;
}
    """)

if __name__ == "__main__":
    print("🚀 Starting Simple Session Management Tests...")
    print(f"🔗 Testing against: {BASE_URL}")
    
    try:
        # Test the session flow
        test_simple_session_flow()
        
        # Show browser extension example
        test_browser_extension_flow()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed!")
        print("💡 Your simple session management API is ready!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {BASE_URL}")
        print("💡 Make sure your FastAPI server is running:")
        print("   python -m uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Test failed: {e}") 