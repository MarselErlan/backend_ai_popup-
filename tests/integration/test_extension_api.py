#!/usr/bin/env python3
"""
🧪 Test Extension API Endpoints
Tests all the API endpoints that the browser extension uses
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

class ExtensionAPITest:
    """Test class for extension API endpoints"""
    
    def __init__(self):
        self.session_id = None
        self.user_id = None
        self.test_email = f"test_extension_{datetime.now().strftime('%H%M%S')}@example.com"
        self.test_password = "password123"

    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def test_api_health(self) -> bool:
        """Test API health endpoint"""
        try:
            self.log("Testing API health...")
            response = requests.get(f"{BASE_URL}/health")
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ API is healthy: {data}", "SUCCESS")
                return True
            else:
                self.log(f"❌ API health check failed: {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ API connection error: {e}", "ERROR")
            return False

    def test_user_registration(self) -> bool:
        """Test user registration (extension signup)"""
        try:
            self.log(f"Testing user registration with email: {self.test_email}")
            
            response = requests.post(f"{BASE_URL}/api/simple/register", json={
                "email": self.test_email,
                "password": self.test_password
            })
            
            if response.status_code == 200:
                data = response.json()
                self.user_id = data.get("user_id")
                self.log(f"✅ User registered successfully: {self.user_id}", "SUCCESS")
                return True
            else:
                self.log(f"❌ Registration failed: {response.status_code} - {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Registration error: {e}", "ERROR")
            return False

    def test_session_creation(self) -> bool:
        """Test session creation (extension session management)"""
        try:
            self.log("Testing session creation...")
            
            if not self.user_id:
                self.log("❌ No user_id available for session creation", "ERROR")
                return False
            
            response = requests.post(f"{BASE_URL}/api/session/check-and-update/{self.user_id}")
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                self.log(f"✅ Session created: {self.session_id}", "SUCCESS")
                return True
            else:
                self.log(f"❌ Session creation failed: {response.status_code} - {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Session creation error: {e}", "ERROR")
            return False

    def test_save_current_page(self) -> bool:
        """Test saving current page URL (Save Current Page button)"""
        try:
            self.log("Testing save current page...")
            
            if not self.session_id:
                self.log("❌ No session_id available for saving page", "ERROR")
                return False
            
            test_url_data = {
                "url": "https://example.com/test-job-application",
                "title": "Test Job Application - Example Company"
            }
            
            response = requests.post(
                f"{BASE_URL}/api/urls/save",
                json=test_url_data,
                headers={"Authorization": self.session_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Page saved successfully: {data.get('message')}", "SUCCESS")
                return True
            else:
                self.log(f"❌ Save page failed: {response.status_code} - {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Save page error: {e}", "ERROR")
            return False

    def test_url_stats(self) -> bool:
        """Test URL statistics (URL stats display)"""
        try:
            self.log("Testing URL statistics...")
            
            if not self.session_id:
                self.log("❌ No session_id available for URL stats", "ERROR")
                return False
            
            response = requests.get(
                f"{BASE_URL}/api/urls/stats/summary",
                headers={"Authorization": self.session_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                stats = data.get("stats", {})
                self.log(f"✅ URL stats retrieved:", "SUCCESS")
                self.log(f"   Total URLs: {stats.get('total_urls', 0)}")
                self.log(f"   Applied: {stats.get('applied', 0)}")
                self.log(f"   In Progress: {stats.get('in_progress', 0)}")
                self.log(f"   Not Applied: {stats.get('not_applied', 0)}")
                return True
            else:
                self.log(f"❌ URL stats failed: {response.status_code} - {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ URL stats error: {e}", "ERROR")
            return False

    def test_form_filling(self) -> bool:
        """Test AI form filling (main extension feature)"""
        try:
            self.log("Testing AI form filling...")
            
            if not self.user_id:
                self.log("❌ No user_id available for form filling", "ERROR")
                return False
            
            test_field_data = {
                "label": "Full Name",
                "url": "https://example.com/job-application",
                "user_id": self.user_id
            }
            
            response = requests.post(
                f"{BASE_URL}/api/generate-field-answer",
                json=test_field_data
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Form filling successful:", "SUCCESS")
                self.log(f"   Answer: {data.get('answer')}")
                self.log(f"   Source: {data.get('data_source')}")
                return True
            else:
                self.log(f"❌ Form filling failed: {response.status_code} - {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Form filling error: {e}", "ERROR")
            return False

    def run_all_tests(self) -> bool:
        """Run all extension API tests"""
        self.log("🚀 Starting Extension API Tests...")
        print("=" * 60)
        
        tests = [
            ("API Health", self.test_api_health),
            ("User Registration", self.test_user_registration),
            ("Session Creation", self.test_session_creation),
            ("Save Current Page", self.test_save_current_page),
            ("URL Statistics", self.test_url_stats),
            ("AI Form Filling", self.test_form_filling),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 Running test: {test_name}")
            try:
                if test_func():
                    self.log(f"✅ {test_name}: PASSED", "SUCCESS")
                    passed += 1
                else:
                    self.log(f"❌ {test_name}: FAILED", "ERROR")
            except Exception as e:
                self.log(f"💥 {test_name}: ERROR - {e}", "ERROR")
        
        print("\n" + "=" * 60)
        self.log(f"🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            self.log("🎉 All extension API tests passed!", "SUCCESS")
            return True
        else:
            self.log("💥 Some tests failed!", "ERROR")
            return False


def main():
    """Main test function"""
    print("🧪 Extension API Test Suite")
    print("=" * 60)
    
    # Test API connection first
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API server is running")
        else:
            print("❌ API server responded with error")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        print("💡 Make sure your backend is running:")
        print("   cd backend_ai_popup")
        print("   python -m uvicorn main:app --reload")
        return
    
    # Run extension tests
    tester = ExtensionAPITest()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎊 Extension API is ready!")
        print("📋 Your extension can now:")
        print("   • ✅ Register and login users")
        print("   • ✅ Create and manage sessions")
        print("   • ✅ Save URLs from browser")
        print("   • ✅ Display URL statistics")
        print("   • ✅ Fill forms with AI")
        print("\n🔧 Next steps:")
        print("   1. Load extension in Chrome (chrome://extensions/)")
        print("   2. Open test-extension.html in browser")
        print("   3. Test extension functionality")
    else:
        print("\n💥 Some API endpoints are not working!")
        print("🔧 Check your backend logs for errors")


if __name__ == "__main__":
    main()
