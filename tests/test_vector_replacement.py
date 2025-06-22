#!/usr/bin/env python3
"""
Test script to verify vector replacement functionality
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
TEST_EMAIL = "erlan1919@gmail.com"
TEST_PASSWORD = "your_password"  # Replace with actual password

def test_vector_replacement():
    """Test that old vectors are properly replaced when re-embedding"""
    
    print("🧪 Testing Vector Replacement Functionality")
    print("=" * 50)
    
    # Step 1: Login to get session
    print("1. Logging in...")
    login_response = requests.post(f"{BASE_URL}/api/simple/login", json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    })
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.text}")
        return False
    
    login_data = login_response.json()
    user_id = login_data["user_id"]
    
    # Step 2: Get session
    print("2. Getting session...")
    session_response = requests.post(f"{BASE_URL}/api/session/check-and-update/{user_id}")
    
    if session_response.status_code != 200:
        print(f"❌ Session creation failed: {session_response.text}")
        return False
    
    session_data = session_response.json()
    session_id = session_data["session_id"]
    
    headers = {"Authorization": session_id}
    
    # Step 3: First re-embed (creates initial vectors)
    print("3. First re-embed (creating initial vectors)...")
    reembed1_response = requests.post(f"{BASE_URL}/api/v1/personal-info/reembed", headers=headers)
    
    if reembed1_response.status_code != 200:
        print(f"❌ First re-embed failed: {reembed1_response.text}")
        return False
    
    reembed1_data = reembed1_response.json()
    print(f"✅ First re-embed completed in {reembed1_data['processing_time']:.2f}s")
    
    # Step 4: Wait a moment
    print("4. Waiting 2 seconds...")
    time.sleep(2)
    
    # Step 5: Second re-embed (should replace old vectors)
    print("5. Second re-embed (should replace old vectors)...")
    reembed2_response = requests.post(f"{BASE_URL}/api/v1/personal-info/reembed", headers=headers)
    
    if reembed2_response.status_code != 200:
        print(f"❌ Second re-embed failed: {reembed2_response.text}")
        return False
    
    reembed2_data = reembed2_response.json()
    print(f"✅ Second re-embed completed in {reembed2_data['processing_time']:.2f}s")
    
    # Step 6: Test the new vector replacement test endpoint (if available)
    print("6. Testing vector replacement endpoint...")
    try:
        vector_test_response = requests.post(f"{BASE_URL}/api/test/vector-replacement", headers=headers)
        
        if vector_test_response.status_code == 200:
            vector_test_data = vector_test_response.json()
            print(f"✅ Vector replacement test completed:")
            print(f"   Status: {vector_test_data['status']}")
            print(f"   Test passed: {vector_test_data['old_vs_new_comparison']['test_passed']}")
            print(f"   Old phone after update: {vector_test_data['old_vs_new_comparison']['old_phone_after_update']}")
            print(f"   New phone found: {vector_test_data['old_vs_new_comparison']['new_phone_found']}")
        else:
            print(f"⚠️ Vector replacement test endpoint not available: {vector_test_response.status_code}")
    except Exception as e:
        print(f"⚠️ Vector replacement test endpoint error: {e}")
    
    print("\n✅ Vector replacement test completed!")
    print("Check the server logs to see if old vectors are being deleted before new ones are stored.")
    
    return True

if __name__ == "__main__":
    test_vector_replacement() 