#!/usr/bin/env python3
"""
Test script for the new field-by-field API
This simulates how a React frontend would interact with the backend
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_field_answer(label: str, url: str, user_id: str = "test-user"):
    """Test the field answer generation endpoint"""
    
    print(f"\n🎯 Testing field: '{label}'")
    print(f"   URL: {url}")
    print(f"   User ID: {user_id}")
    
    # Prepare request
    payload = {
        "label": label,
        "url": url,
        "user_id": user_id
    }
    
    try:
        # Make API call
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/generate-field-answer",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        end_time = time.time()
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success ({(end_time - start_time):.2f}s)")
            print(f"   📝 Answer: '{data.get('answer', 'N/A')}'")
            print(f"   🔗 Source: {data.get('data_source', 'N/A')}")
            print(f"   💭 Reasoning: {data.get('reasoning', 'N/A')[:100]}...")
            return data
        else:
            print(f"   ❌ Error: HTTP {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return None

def test_health_check():
    """Test the health check endpoint"""
    print("\n🏥 Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   📊 Services: {data.get('services')}")
            return True
        else:
            print(f"   ❌ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

def main():
    """Run comprehensive API tests"""
    print("🚀 Smart Form Fill API v4.0 - Field-by-Field Test Suite")
    print("=" * 60)
    
    # Test health check first
    if not test_health_check():
        print("\n❌ Health check failed. Make sure the server is running.")
        return
    
    # Test various field types
    test_cases = [
        {
            "label": "What is your full name?",
            "url": "https://example.com/job-application",
            "expected_source": "resume_vectordb"
        },
        {
            "label": "What is your current occupation?", 
            "url": "https://jobs.google.com/apply",
            "expected_source": "resume_vectordb"
        },
        {
            "label": "What is your email address?",
            "url": "https://careers.microsoft.com/apply",
            "expected_source": "personal_info_vectordb"
        },
        {
            "label": "What programming languages do you know?",
            "url": "https://careers.amazon.com/apply", 
            "expected_source": "resume_vectordb"
        },
        {
            "label": "Do you require visa sponsorship?",
            "url": "https://jobs.netflix.com/apply",
            "expected_source": "personal_info_vectordb"
        },
        {
            "label": "Upload your resume",
            "url": "https://jobs.apple.com/apply",
            "expected_source": "skipped"
        }
    ]
    
    print(f"\n📋 Running {len(test_cases)} test cases...")
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}/{len(test_cases)} ---")
        
        result = test_field_answer(
            label=test_case["label"],
            url=test_case["url"],
            user_id="test-user"
        )
        
        if result:
            success_count += 1
            # Validate expected behavior
            actual_source = result.get("data_source", "")
            expected_source = test_case["expected_source"]
            
            if expected_source in actual_source or actual_source == expected_source:
                print(f"   🎉 Data source validation: PASSED")
            else:
                print(f"   ⚠️  Data source validation: Expected '{expected_source}', got '{actual_source}'")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{len(test_cases)} passed")
    
    if success_count == len(test_cases):
        print("🎉 All tests passed! The API is ready for React integration.")
    else:
        print("⚠️  Some tests failed. Please check the server logs.")
    
    # React integration example
    print("\n🔧 React Integration Example:")
    print("""
    const handleFieldFill = async (fieldElement) => {
      const label = getFieldLabel(fieldElement);
      const url = window.location.href;
      
      try {
        const response = await fetch('http://localhost:8000/api/generate-field-answer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label, url, user_id: 'your-user-id' })
        });
        
        const data = await response.json();
        fieldElement.value = data.answer || '⚠️ No answer';
        
      } catch (error) {
        console.error('API Error:', error);
        fieldElement.value = '⚠️ Backend error';
      }
    };
    """)

if __name__ == "__main__":
    main() 