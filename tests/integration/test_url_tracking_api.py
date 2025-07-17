#!/usr/bin/env python3
"""
Test URL Tracking API - Browser Extension URL Management
"""

import requests
import json
from typing import Dict, Any
from loguru import logger

# API base URL
BASE_URL = "http://localhost:8000"

class URLTrackingAPITest:
    """Test class for URL tracking API"""
    
    def __init__(self):
        self.session_id = None
        self.user_id = None
        self.test_urls = [
            {
                "url": "https://jobs.google.com/apply/123",
                "title": "Software Engineer - Google",
                "notes": "Great company, good benefits"
            },
            {
                "url": "https://careers.microsoft.com/job/456",
                "title": "Full Stack Developer - Microsoft",
                "notes": "Remote work available"
            },
            {
                "url": "https://www.linkedin.com/jobs/view/789",
                "title": "Senior Developer - Startup",
                "notes": "Equity package included"
            }
        ]
    
    def register_and_login(self) -> bool:
        """Register a test user and get session"""
        try:
            # Register user
            register_data = {
                "email": "urltest@example.com",
                "password": "testpass123"
            }
            
            logger.info("🔐 Registering test user...")
            response = requests.post(f"{BASE_URL}/api/simple/register", json=register_data)
            
            if response.status_code == 409:
                # User already exists, try login
                logger.info("👤 User exists, trying login...")
                response = requests.post(f"{BASE_URL}/api/simple/login", json=register_data)
            
            if response.status_code == 200:
                data = response.json()
                self.user_id = data["user_id"]
                logger.info(f"✅ User authenticated: {data['email']} (ID: {self.user_id})")
                
                # Create session
                session_data = {"user_id": self.user_id}
                session_response = requests.post(f"{BASE_URL}/api/session/create", json=session_data)
                
                if session_response.status_code == 200:
                    session_info = session_response.json()
                    self.session_id = session_info["session_id"]
                    logger.info(f"🎫 Session created: {self.session_id}")
                    return True
                else:
                    logger.error(f"❌ Session creation failed: {session_response.text}")
                    return False
            else:
                logger.error(f"❌ Authentication failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Registration/login failed: {e}")
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers with session authentication"""
        return {
            "Authorization": self.session_id,
            "Content-Type": "application/json"
        }
    
    def test_save_urls(self) -> bool:
        """Test saving URLs from browser extension"""
        try:
            logger.info("💾 Testing URL saving...")
            saved_urls = []
            
            for i, url_data in enumerate(self.test_urls):
                logger.info(f"   📝 Saving URL {i+1}: {url_data['title']}")
                
                response = requests.post(
                    f"{BASE_URL}/api/urls/save",
                    json=url_data,
                    headers=self.get_headers()
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"   ✅ Saved: {result['message']}")
                    if result["status"] in ["success", "exists"]:
                        saved_urls.append(result["url"])
                else:
                    logger.error(f"   ❌ Failed to save URL: {response.text}")
                    return False
            
            logger.info(f"✅ Successfully saved {len(saved_urls)} URLs")
            return True
            
        except Exception as e:
            logger.error(f"❌ URL saving test failed: {e}")
            return False
    
    def test_get_urls(self) -> bool:
        """Test retrieving user URLs"""
        try:
            logger.info("📋 Testing URL retrieval...")
            
            # Get all URLs
            response = requests.get(
                f"{BASE_URL}/api/urls/list",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   ✅ Retrieved {data['total']} URLs")
                
                for url in data['urls']:
                    logger.info(f"      📌 {url['title']} - Status: {url['status']}")
                
                # Test filtering by status
                status_response = requests.get(
                    f"{BASE_URL}/api/urls/list?status=not_applied",
                    headers=self.get_headers()
                )
                
                if status_response.status_code == 200:
                    filtered_data = status_response.json()
                    logger.info(f"   🔍 Filtered 'not_applied': {filtered_data['total']} URLs")
                    return True
                else:
                    logger.error(f"   ❌ Status filtering failed: {status_response.text}")
                    return False
            else:
                logger.error(f"❌ URL retrieval failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ URL retrieval test failed: {e}")
            return False
    
    def test_update_status(self) -> bool:
        """Test updating URL application status"""
        try:
            logger.info("🔄 Testing status updates...")
            
            # Get URLs first
            response = requests.get(
                f"{BASE_URL}/api/urls/list",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['urls']:
                    url_id = data['urls'][0]['id']
                    url_title = data['urls'][0]['title']
                    
                    # Update status to "applied"
                    update_data = {
                        "status": "applied",
                        "notes": "Applied successfully via company website"
                    }
                    
                    logger.info(f"   🎯 Updating status for: {url_title}")
                    
                    update_response = requests.put(
                        f"{BASE_URL}/api/urls/{url_id}/status",
                        json=update_data,
                        headers=self.get_headers()
                    )
                    
                    if update_response.status_code == 200:
                        result = update_response.json()
                        logger.info(f"   ✅ Status updated: {result['url']['status']}")
                        logger.info(f"   📝 Applied at: {result['url']['applied_at']}")
                        return True
                    else:
                        logger.error(f"   ❌ Status update failed: {update_response.text}")
                        return False
                else:
                    logger.warning("   ⚠️ No URLs found to update")
                    return True
            else:
                logger.error(f"❌ Failed to get URLs for status update: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Status update test failed: {e}")
            return False
    
    def test_get_stats(self) -> bool:
        """Test getting URL statistics"""
        try:
            logger.info("📊 Testing URL statistics...")
            
            response = requests.get(
                f"{BASE_URL}/api/urls/stats/summary",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                stats = data['stats']
                
                logger.info("   📈 URL Statistics:")
                logger.info(f"      Total URLs: {stats['total_urls']}")
                logger.info(f"      Not Applied: {stats['not_applied']}")
                logger.info(f"      Applied: {stats['applied']}")
                logger.info(f"      In Progress: {stats['in_progress']}")
                logger.info(f"      Application Rate: {stats['application_rate']}%")
                logger.info(f"      Recent Activity: {stats['recent_activity']}")
                
                return True
            else:
                logger.error(f"❌ Stats retrieval failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Stats test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all URL tracking tests"""
        logger.info("🚀 Starting URL Tracking API Tests...")
        
        tests = [
            ("Authentication", self.register_and_login),
            ("Save URLs", self.test_save_urls),
            ("Get URLs", self.test_get_urls),
            ("Update Status", self.test_update_status),
            ("Get Statistics", self.test_get_stats),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running test: {test_name}")
            try:
                if test_func():
                    logger.info(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
        
        logger.info(f"\n🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All URL tracking tests passed!")
            return True
        else:
            logger.error("💥 Some tests failed!")
            return False

def main():
    """Main test function"""
    # Test health check first
    try:
        logger.info("🏥 Testing API health...")
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code == 200:
            logger.info("✅ API is healthy")
        else:
            logger.error("❌ API health check failed")
            return
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        logger.info("💡 Make sure the API server is running: python main.py")
        return
    
    # Run URL tracking tests
    tester = URLTrackingAPITest()
    success = tester.run_all_tests()
    
    if success:
        logger.info("\n🎊 URL Tracking API is ready for browser extension integration!")
        logger.info("📋 Available endpoints:")
        logger.info("   • POST /api/urls/save - Save URL from extension")
        logger.info("   • GET /api/urls/list - Get user's tracked URLs")
        logger.info("   • PUT /api/urls/{id}/status - Update application status")
        logger.info("   • GET /api/urls/stats/summary - Get application statistics")
    else:
        logger.error("\n💥 URL Tracking API tests failed!")

if __name__ == "__main__":
    main() 