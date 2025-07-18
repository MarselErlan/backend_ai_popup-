#!/usr/bin/env python3
"""
Comprehensive test script for Pinecone integration features
Tests all new endpoints and functionality
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword123"

class PineconeFeaturesTester:
    def __init__(self):
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.user_id = None
        self.session_id = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log test messages"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{timestamp} | {level} | {message}")
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                self.log("✅ API connection successful")
                return True
            else:
                self.log(f"❌ API connection failed: {response.status_code}")
                return False
        except Exception as e:
            self.log(f"❌ API connection error: {e}")
            return False
    
    def register_and_login(self) -> bool:
        """Register and login test user"""
        try:
            # Register user
            register_data = {
                "email": TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD
            }
            
            response = self.session.post(f"{self.base_url}/api/simple/register", json=register_data)
            if response.status_code in [200, 409]:  # 409 if user already exists
                self.log("✅ User registration/exists check successful")
            else:
                self.log(f"❌ User registration failed: {response.status_code} - {response.text}")
                return False
            
            # Login user
            response = self.session.post(f"{self.base_url}/api/simple/login", json=register_data)
            if response.status_code == 200:
                user_data = response.json()
                self.user_id = user_data["user_id"]
                self.log(f"✅ User login successful - User ID: {self.user_id}")
            else:
                self.log(f"❌ User login failed: {response.status_code} - {response.text}")
                return False
            
            # Create session
            session_data = {
                "user_id": self.user_id,
                "device_info": "PineconeFeaturesTester"
            }
            
            response = self.session.post(f"{self.base_url}/api/session/create", json=session_data)
            if response.status_code == 200:
                session_data = response.json()
                self.session_id = session_data["session_id"]
                self.log(f"✅ Session created - Session ID: {self.session_id}")
                
                # Set session header for authenticated requests
                self.session.headers.update({"Authorization": self.session_id})
                return True
            else:
                self.log(f"❌ Session creation failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.log(f"❌ Authentication error: {e}")
            return False
    
    def test_document_upload_with_embedding(self) -> bool:
        """Test document upload with automatic embedding"""
        try:
            # Test resume upload
            resume_content = """
            John Doe
            Software Engineer
            
            Experience:
            - 5 years Python development
            - Machine Learning expertise
            - FastAPI and React experience
            
            Education:
            - BS Computer Science, MIT
            
            Skills:
            - Python, JavaScript, SQL
            - AWS, Docker, Kubernetes
            - Machine Learning, AI
            """
            
            # Create a test file
            files = {
                'file': ('test_resume.txt', resume_content.encode(), 'text/plain')
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/resume/upload", files=files)
            if response.status_code == 200:
                result = response.json()
                self.log(f"✅ Resume upload successful - Document ID: {result['document_id']}")
                self.log(f"   Processing time: {result['processing_time']}s")
            else:
                self.log(f"❌ Resume upload failed: {response.status_code} - {response.text}")
                return False
            
            # Test personal info upload
            personal_info_content = """
            Personal Information:
            
            Name: John Doe
            Email: john.doe@example.com
            Phone: (555) 123-4567
            Address: 123 Main St, San Francisco, CA 94105
            
            LinkedIn: linkedin.com/in/johndoe
            GitHub: github.com/johndoe
            
            Preferences:
            - Remote work preferred
            - Available for travel
            - Salary range: $120k-$150k
            """
            
            files = {
                'file': ('test_personal_info.txt', personal_info_content.encode(), 'text/plain')
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/personal-info/upload", files=files)
            if response.status_code == 200:
                result = response.json()
                self.log(f"✅ Personal info upload successful - Document ID: {result['document_id']}")
                self.log(f"   Processing time: {result['processing_time']}s")
                return True
            else:
                self.log(f"❌ Personal info upload failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.log(f"❌ Document upload error: {e}")
            return False
    
    def test_vector_store_stats(self) -> bool:
        """Test vector store statistics"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/vector-store/stats")
            if response.status_code == 200:
                result = response.json()
                self.log(f"✅ Vector store stats retrieved")
                self.log(f"   Vector store type: {result['vector_store_type']}")
                self.log(f"   Stats: {json.dumps(result['stats'], indent=2)}")
                return True
            else:
                self.log(f"❌ Vector store stats failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.log(f"❌ Vector store stats error: {e}")
            return False
    
    def test_form_filling_with_vectors(self) -> bool:
        """Test form filling using vector search"""
        try:
            # Test various field types
            test_fields = [
                "Full Name",
                "Email Address", 
                "Phone Number",
                "Current Job Title",
                "Years of Experience",
                "Programming Languages",
                "Education",
                "Address"
            ]
            
            all_passed = True
            
            for field in test_fields:
                field_data = {
                    "label": field,
                    "url": "https://example.com/job-application"
                }
                
                response = self.session.post(f"{self.base_url}/api/generate-field-answer", json=field_data)
                if response.status_code == 200:
                    result = response.json()
                    self.log(f"✅ Field '{field}': {result['answer']}")
                    self.log(f"   Source: {result['data_source']}")
                else:
                    self.log(f"❌ Field '{field}' failed: {response.status_code}")
                    all_passed = False
                
                # Small delay between requests
                time.sleep(0.5)
            
            return all_passed
            
        except Exception as e:
            self.log(f"❌ Form filling test error: {e}")
            return False
    
    def test_reembedding(self) -> bool:
        """Test re-embedding functionality"""
        try:
            # Test resume re-embedding
            response = self.session.post(f"{self.base_url}/api/v1/resume/reembed")
            if response.status_code == 200:
                result = response.json()
                self.log(f"✅ Resume re-embedding successful")
                self.log(f"   Document ID: {result.get('document_id', 'N/A')}")
            else:
                self.log(f"❌ Resume re-embedding failed: {response.status_code} - {response.text}")
                return False
            
            # Test personal info re-embedding
            response = self.session.post(f"{self.base_url}/api/v1/personal-info/reembed")
            if response.status_code == 200:
                result = response.json()
                self.log(f"✅ Personal info re-embedding successful")
                self.log(f"   Document ID: {result.get('document_id', 'N/A')}")
                return True
            else:
                self.log(f"❌ Personal info re-embedding failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.log(f"❌ Re-embedding test error: {e}")
            return False
    
    def test_vector_search_direct(self) -> bool:
        """Test direct vector search functionality"""
        try:
            # Test resume search
            search_params = {
                "query": "Python programming experience",
                "top_k": 3,
                "min_score": 0.1
            }
            
            response = self.session.get(f"{self.base_url}/api/v1/resume/search", params=search_params)
            if response.status_code == 200:
                results = response.json()
                self.log(f"✅ Resume search successful - Found {len(results)} results")
                for i, result in enumerate(results):
                    self.log(f"   Result {i+1}: Score {result['score']:.3f} - {result['text'][:100]}...")
            else:
                self.log(f"❌ Resume search failed: {response.status_code} - {response.text}")
                return False
            
            return True
            
        except Exception as e:
            self.log(f"❌ Vector search test error: {e}")
            return False
    
    def test_vector_cleanup(self) -> bool:
        """Test vector cleanup functionality"""
        try:
            # Test clearing vectors
            response = self.session.delete(f"{self.base_url}/api/v1/vector-store/clear")
            if response.status_code == 200:
                result = response.json()
                self.log(f"✅ Vector cleanup successful")
                self.log(f"   Message: {result['message']}")
                return True
            else:
                self.log(f"❌ Vector cleanup failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.log(f"❌ Vector cleanup test error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        self.log("🚀 Starting Pinecone Features Test Suite")
        
        results = {}
        
        # Test 1: Connection
        results["connection"] = self.test_connection()
        if not results["connection"]:
            self.log("❌ Connection test failed - stopping tests")
            return results
        
        # Test 2: Authentication
        results["authentication"] = self.register_and_login()
        if not results["authentication"]:
            self.log("❌ Authentication test failed - stopping tests")
            return results
        
        # Test 3: Document Upload with Embedding
        results["document_upload"] = self.test_document_upload_with_embedding()
        
        # Test 4: Vector Store Stats
        results["vector_stats"] = self.test_vector_store_stats()
        
        # Test 5: Form Filling with Vectors
        results["form_filling"] = self.test_form_filling_with_vectors()
        
        # Test 6: Re-embedding
        results["reembedding"] = self.test_reembedding()
        
        # Test 7: Direct Vector Search
        results["vector_search"] = self.test_vector_search_direct()
        
        # Test 8: Vector Cleanup
        results["vector_cleanup"] = self.test_vector_cleanup()
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Print test summary"""
        self.log("📊 Test Summary")
        self.log("=" * 50)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            self.log(f"{test_name.replace('_', ' ').title()}: {status}")
        
        self.log("=" * 50)
        self.log(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.log("🎉 All tests passed! Pinecone integration is working correctly.")
        else:
            self.log(f"⚠️ {total_tests - passed_tests} tests failed. Check the logs above.")

def main():
    """Main test runner"""
    tester = PineconeFeaturesTester()
    results = tester.run_all_tests()
    tester.print_summary(results)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 