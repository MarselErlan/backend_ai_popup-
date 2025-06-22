#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE END-TO-END TESTING SUITE
Tests the complete user workflow from registration to intelligent form filling

This test simulates a real user's journey:
1. Registration & Authentication
2. Document Upload & Processing
3. Vector Database Operations
4. URL Tracking & Management
5. AI-Powered Form Filling
6. Performance & Health Monitoring
"""

import asyncio
import aiohttp
import json
import time
import tempfile
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Test Configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class EndToEndTestSuite:
    """Comprehensive end-to-end test suite"""
    
    def __init__(self):
        self.session = None
        self.user_id = None
        self.session_id = None
        self.test_email = f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}@example.com"
        self.test_password = "E2ETestPassword123!"
        
        # Test results tracking
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "performance_metrics": {},
            "errors": []
        }
        
        # Test data
        self.sample_resume = self._create_sample_resume()
        self.sample_personal_info = self._create_sample_personal_info()
        self.test_urls = [
            "https://jobs.google.com/apply/12345",
            "https://careers.microsoft.com/jobs/67890",
            "https://jobs.netflix.com/jobs/software-engineer"
        ]
    
    def _create_sample_resume(self) -> str:
        """Create a comprehensive sample resume for testing"""
        return """
ALEX JOHNSON
Senior Software Engineer

CONTACT INFORMATION:
Email: alex.johnson@email.com
Phone: (555) 123-4567
Location: San Francisco, CA
LinkedIn: linkedin.com/in/alexjohnson
GitHub: github.com/alexjohnson

PROFESSIONAL SUMMARY:
Experienced full-stack software engineer with 6+ years building scalable web applications,
microservices, and AI-powered solutions. Expertise in Python, JavaScript, React, and cloud technologies.
Proven track record of leading development teams and delivering high-impact products.

TECHNICAL SKILLS:
• Programming Languages: Python, JavaScript, TypeScript, Java, Go
• Frontend: React, Vue.js, HTML5, CSS3, Tailwind CSS
• Backend: FastAPI, Django, Node.js, Express, Spring Boot
• Databases: PostgreSQL, MongoDB, Redis, MySQL, Elasticsearch
• Cloud & DevOps: AWS, GCP, Docker, Kubernetes, Terraform, CI/CD
• AI/ML: OpenAI API, LangChain, TensorFlow, scikit-learn, Vector Databases
• Tools: Git, Jira, Figma, Postman, DataDog, Prometheus

PROFESSIONAL EXPERIENCE:

Senior Software Engineer | TechCorp Inc. | Jan 2022 - Present
• Led a team of 5 engineers developing a real-time analytics platform serving 10M+ users
• Built microservices architecture using FastAPI and Docker, reducing latency by 40%
• Implemented AI-powered recommendation system increasing user engagement by 25%
• Designed and deployed ML pipelines for automated data processing and insights
• Mentored junior developers and established best practices for code quality

Full Stack Developer | StartupXYZ | Mar 2020 - Dec 2021
• Developed responsive web applications using React and Node.js for e-commerce platform
• Built RESTful APIs and GraphQL endpoints handling 1M+ requests per day
• Implemented payment processing, user authentication, and order management systems
• Optimized database queries and caching, improving response times by 60%
• Collaborated with product team to deliver features on tight deadlines

Software Engineer | InnovateTech | Jun 2018 - Feb 2020
• Created automated testing frameworks reducing bug reports by 35%
• Developed internal tools for deployment and monitoring using Python and Bash
• Contributed to open-source projects and technical documentation
• Participated in code reviews and agile development processes

EDUCATION:
Bachelor of Science in Computer Science | University of California, Berkeley | 2018
• Relevant Coursework: Data Structures, Algorithms, Database Systems, Machine Learning
• GPA: 3.8/4.0
• Senior Project: AI-powered chatbot for customer service automation

PROJECTS:
• Smart Form Filler: AI-powered browser extension for automatic form completion
• Weather Analytics Dashboard: Real-time weather data visualization using D3.js
• Task Management API: RESTful service with authentication and real-time updates

CERTIFICATIONS:
• AWS Certified Solutions Architect - Associate (2023)
• Google Cloud Professional Developer (2022)
• MongoDB Certified Developer (2021)

ACHIEVEMENTS:
• Published 3 technical articles on Medium with 10K+ views
• Speaker at PyBay 2023 conference on "Building Scalable AI Applications"
• Hackathon winner - Best AI Application (2022)
• Open source contributor with 500+ GitHub stars across projects
        """
    
    def _create_sample_personal_info(self) -> str:
        """Create comprehensive personal information for testing"""
        return """
PERSONAL INFORMATION FOR JOB APPLICATIONS

CONTACT DETAILS:
Full Name: Alex Johnson
Email: alex.johnson@email.com
Phone: (555) 123-4567
Current Address: 123 Tech Street, Apt 4B, San Francisco, CA 94105
LinkedIn: https://linkedin.com/in/alexjohnson
GitHub: https://github.com/alexjohnson
Portfolio: https://alexjohnson.dev

WORK AUTHORIZATION:
• US Citizen - Authorized to work in the United States
• No visa sponsorship required
• Available for relocation nationwide
• Valid driver's license and passport

SALARY & COMPENSATION:
• Current Salary: $145,000 base + equity + benefits
• Desired Salary Range: $160,000 - $200,000 (negotiable)
• Open to equity compensation and comprehensive benefits
• Interested in performance bonuses and professional development opportunities

WORK PREFERENCES:
• Preferred Work Model: Hybrid (3 days in office, 2 remote)
• Open to full remote for exceptional opportunities
• Available start date: 2 weeks notice required
• Willing to travel up to 20% for business needs
• Preferred company size: 50-500 employees (growth stage)

CAREER OBJECTIVES:
• Seeking Senior/Staff Software Engineer or Technical Lead roles
• Interested in AI/ML, fintech, healthcare tech, or climate tech sectors
• Goal to lead technical initiatives and mentor engineering teams
• Looking for companies with strong engineering culture and learning opportunities

EDUCATION DETAILS:
• University of California, Berkeley
• Bachelor of Science in Computer Science (2018)
• GPA: 3.8/4.0
• Relevant coursework: Machine Learning, Database Systems, Software Engineering
• Active in Computer Science student organizations

REFERENCES:
• Sarah Chen, Engineering Manager at TechCorp Inc.
  Email: sarah.chen@techcorp.com, Phone: (555) 987-6543
• Mike Rodriguez, CTO at StartupXYZ
  Email: mike@startupxyz.com, Phone: (555) 456-7890
• Dr. Jennifer Lee, CS Professor at UC Berkeley
  Email: jlee@berkeley.edu, Phone: (510) 642-1234

ADDITIONAL INFORMATION:
• Bilingual: English (native), Spanish (conversational)
• Security Clearance: None (eligible for clearance if required)
• Professional Memberships: ACM, IEEE Computer Society
• Volunteer Work: Code for America volunteer, teaching coding to underserved youth
• Hobbies: Rock climbing, photography, contributing to open source projects

COVER LETTER TEMPLATES:
• Template 1: Technical focus for engineering roles
• Template 2: Leadership focus for senior positions
• Template 3: Startup focus for high-growth companies
• Template 4: Enterprise focus for large corporations

INTERVIEW PREPARATION:
• Available for phone, video, or in-person interviews
• Preferred interview times: Weekday afternoons (Pacific Time)
• Portfolio projects ready for technical discussions
• References available upon request
        """
    
    async def setup(self):
        """Initialize test session"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT))
        self.log("✅ Test session initialized")
    
    async def cleanup(self):
        """Clean up test session and resources"""
        if self.session:
            await self.session.close()
        self.log("🧹 Test session cleaned up")
    
    def log(self, message: str, test_name: str = None, success: bool = None):
        """Log test messages with formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if success is not None:
            icon = "✅" if success else "❌"
            if test_name:
                print(f"[{timestamp}] {icon} {test_name}: {message}")
                
                # Track results
                self.test_results["total_tests"] += 1
                if success:
                    self.test_results["passed_tests"] += 1
                else:
                    self.test_results["failed_tests"] += 1
                    self.test_results["errors"].append(f"{test_name}: {message}")
                
                self.test_results["test_details"].append({
                    "test_name": test_name,
                    "message": message,
                    "success": success,
                    "timestamp": timestamp
                })
            else:
                print(f"[{timestamp}] {icon} {message}")
        else:
            print(f"[{timestamp}] 📝 {message}")
    
    async def test_api_health(self) -> bool:
        """Test 1: API Health and Connectivity"""
        try:
            start_time = time.time()
            async with self.session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    response_time = time.time() - start_time
                    self.test_results["performance_metrics"]["health_check_time"] = response_time
                    
                    self.log(f"API healthy in {response_time:.3f}s - Services: {data.get('services', {})}", 
                            "API Health Check", True)
                    return True
                else:
                    self.log(f"Health check failed: HTTP {response.status}", "API Health Check", False)
                    return False
        except Exception as e:
            self.log(f"Connection error: {str(e)}", "API Health Check", False)
            return False
    
    async def test_user_registration(self) -> bool:
        """Test 2: User Registration"""
        try:
            start_time = time.time()
            
            payload = {
                "email": self.test_email,
                "password": self.test_password
            }
            
            async with self.session.post(f"{BASE_URL}/api/simple/register", json=payload) as response:
                response_time = time.time() - start_time
                
                if response.status in [200, 409]:  # 409 if user already exists
                    data = await response.json()
                    
                    if response.status == 409:
                        # User exists, try login instead
                        return await self.test_user_login()
                    else:
                        self.user_id = data.get("user_id")
                        self.log(f"User registered successfully in {response_time:.3f}s - ID: {self.user_id}", 
                                "User Registration", True)
                        return True
                else:
                    error_data = await response.text()
                    self.log(f"Registration failed: HTTP {response.status} - {error_data}", 
                            "User Registration", False)
                    return False
        except Exception as e:
            self.log(f"Registration error: {str(e)}", "User Registration", False)
            return False
    
    async def test_user_login(self) -> bool:
        """Test 3: User Login"""
        try:
            start_time = time.time()
            
            payload = {
                "email": self.test_email,
                "password": self.test_password
            }
            
            async with self.session.post(f"{BASE_URL}/api/simple/login", json=payload) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    self.user_id = data.get("user_id")
                    self.log(f"User logged in successfully in {response_time:.3f}s - ID: {self.user_id}", 
                            "User Login", True)
                    return True
                else:
                    error_data = await response.text()
                    self.log(f"Login failed: HTTP {response.status} - {error_data}", 
                            "User Login", False)
                    return False
        except Exception as e:
            self.log(f"Login error: {str(e)}", "User Login", False)
            return False
    
    async def test_session_creation(self) -> bool:
        """Test 4: Session Creation and Management"""
        try:
            if not self.user_id:
                self.log("No user_id available for session creation", "Session Creation", False)
                return False
            
            start_time = time.time()
            
            async with self.session.post(f"{BASE_URL}/api/session/check-and-update/{self.user_id}") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    self.session_id = data.get("session_id")
                    self.log(f"Session created in {response_time:.3f}s - Session ID: {self.session_id[:8]}...", 
                            "Session Creation", True)
                    return True
                else:
                    error_data = await response.text()
                    self.log(f"Session creation failed: HTTP {response.status} - {error_data}", 
                            "Session Creation", False)
                    return False
        except Exception as e:
            self.log(f"Session creation error: {str(e)}", "Session Creation", False)
            return False
    
    async def test_demo_form_filling(self) -> bool:
        """Test 5: Demo Form Filling (No Auth Required)"""
        try:
            start_time = time.time()
            
            # Test demo form filling
            demo_payload = {
                "label": "What is your name?",
                "url": "https://example.com/demo-form"
            }
            
            async with self.session.post(f"{BASE_URL}/api/demo/generate-field-answer", 
                                       json=demo_payload) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    answer = data.get("answer", "")
                    
                    self.log(f"Demo form filling working in {response_time:.3f}s - Answer: '{answer[:50]}...'", 
                            "Demo Form Filling", True)
                    return True
                else:
                    error_data = await response.text()
                    self.log(f"Demo form filling failed: HTTP {response.status} - {error_data}", 
                            "Demo Form Filling", False)
                    return False
                    
        except Exception as e:
            self.log(f"Demo form filling error: {str(e)}", "Demo Form Filling", False)
            return False
    
    async def test_authenticated_form_filling(self) -> bool:
        """Test 6: Authenticated Form Filling"""
        try:
            if not self.session_id:
                self.log("No session_id available for authenticated form filling", "Authenticated Form Filling", False)
                return False
            
            start_time = time.time()
            headers = {"Authorization": self.session_id}
            
            # Test authenticated form filling
            payload = {
                "label": "What is your email address?",
                "url": "https://jobs.example.com/apply"
            }
            
            async with self.session.post(f"{BASE_URL}/api/generate-field-answer", 
                                       json=payload, headers=headers) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    answer = data.get("answer", "")
                    data_source = data.get("data_source", "")
                    
                    self.log(f"Authenticated form filling working in {response_time:.3f}s - Answer: '{answer[:30]}...' (Source: {data_source})", 
                            "Authenticated Form Filling", True)
                    return True
                else:
                    error_data = await response.text()
                    self.log(f"Authenticated form filling failed: HTTP {response.status} - {error_data}", 
                            "Authenticated Form Filling", False)
                    return False
                    
        except Exception as e:
            self.log(f"Authenticated form filling error: {str(e)}", "Authenticated Form Filling", False)
            return False
    
    async def test_url_tracking(self) -> bool:
        """Test 7: URL Tracking"""
        try:
            if not self.session_id:
                self.log("No session_id available for URL tracking", "URL Tracking", False)
                return False
            
            start_time = time.time()
            headers = {"Authorization": self.session_id}
            
            # Test saving a URL
            url_data = {
                "url": "https://jobs.google.com/test-application",
                "title": "Software Engineer - Test Company"
            }
            
            async with self.session.post(f"{BASE_URL}/api/urls/save", 
                                       json=url_data, headers=headers) as response:
                if response.status == 200:
                    # Test getting URL stats
                    async with self.session.get(f"{BASE_URL}/api/urls/stats/summary", 
                                               headers=headers) as stats_response:
                        response_time = time.time() - start_time
                        
                        if stats_response.status == 200:
                            stats_data = await stats_response.json()
                            total_urls = stats_data.get("stats", {}).get("total_urls", 0)
                            
                            self.log(f"URL tracking working in {response_time:.3f}s - Total URLs: {total_urls}", 
                                    "URL Tracking", True)
                            return True
                        else:
                            self.log(f"URL stats failed: HTTP {stats_response.status}", "URL Tracking", False)
                            return False
                else:
                    error_data = await response.text()
                    self.log(f"URL saving failed: HTTP {response.status} - {error_data}", 
                            "URL Tracking", False)
                    return False
                    
        except Exception as e:
            self.log(f"URL tracking error: {str(e)}", "URL Tracking", False)
            return False
    
    async def test_session_cleanup(self) -> bool:
        """Test 8: Session Cleanup"""
        try:
            if not self.session_id:
                self.log("No session to clean up", "Session Cleanup", True)
                return True
            
            start_time = time.time()
            
            async with self.session.delete(f"{BASE_URL}/api/session/{self.session_id}") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    self.log(f"Session cleaned up in {response_time:.3f}s", "Session Cleanup", True)
                    return True
                else:
                    error_data = await response.text()
                    self.log(f"Session cleanup failed: HTTP {response.status} - {error_data}", 
                            "Session Cleanup", False)
                    return False
                    
        except Exception as e:
            self.log(f"Session cleanup error: {str(e)}", "Session Cleanup", False)
            return False
    
    async def run_comprehensive_test_suite(self):
        """Run the complete end-to-end test suite"""
        print("🚀 COMPREHENSIVE END-TO-END TEST SUITE")
        print("=" * 80)
        print(f"Testing against: {BASE_URL}")
        print(f"Test User: {self.test_email}")
        print("=" * 80)
        
        await self.setup()
        
        try:
            # Run all tests in sequence
            test_sequence = [
                self.test_api_health,
                self.test_user_registration,
                self.test_session_creation,
                self.test_demo_form_filling,
                self.test_authenticated_form_filling,
                self.test_url_tracking,
                self.test_session_cleanup
            ]
            
            print(f"\n📋 Running {len(test_sequence)} comprehensive tests...\n")
            
            overall_start_time = time.time()
            
            for test_func in test_sequence:
                await test_func()
                await asyncio.sleep(0.5)  # Small delay between tests
            
            total_test_time = time.time() - overall_start_time
            
            # Generate final report
            self.generate_final_report(total_test_time)
            
        finally:
            await self.cleanup()
    
    def generate_final_report(self, total_time: float):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("📊 END-TO-END TEST RESULTS SUMMARY")
        print("=" * 80)
        
        # Basic stats
        total_tests = self.test_results["total_tests"]
        passed_tests = self.test_results["passed_tests"]
        failed_tests = self.test_results["failed_tests"]
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📈 TEST STATISTICS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ✅")
        print(f"   Failed: {failed_tests} ❌")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Time: {total_time:.2f}s")
        
        # Performance metrics
        metrics = self.test_results["performance_metrics"]
        if metrics:
            print(f"\n⚡ PERFORMANCE METRICS:")
            
            if "health_check_time" in metrics:
                print(f"   Health Check: {metrics['health_check_time']*1000:.1f}ms")
        
        # Test details
        print(f"\n📋 TEST DETAILS:")
        for detail in self.test_results["test_details"]:
            icon = "✅" if detail["success"] else "❌"
            print(f"   {icon} {detail['test_name']}")
        
        # Errors
        if self.test_results["errors"]:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for error in self.test_results["errors"]:
                print(f"   • {error}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if success_rate >= 90:
            print("   🏆 EXCELLENT - System is production-ready!")
        elif success_rate >= 75:
            print("   ✅ GOOD - System is mostly functional with minor issues")
        elif success_rate >= 50:
            print("   ⚠️  FAIR - System has significant issues that need attention")
        else:
            print("   ❌ POOR - System has major problems requiring immediate fixes")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"e2e_test_report_{timestamp}.json"
        
        detailed_report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_time": total_time
            },
            "performance_metrics": metrics,
            "test_details": self.test_results["test_details"],
            "errors": self.test_results["errors"],
            "test_configuration": {
                "base_url": BASE_URL,
                "test_email": self.test_email,
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print("=" * 80)

async def main():
    """Main test runner"""
    print("🎯 END-TO-END TESTING FOR MAIN.PY")
    print("=" * 80)
    print("This test validates the complete user workflow:")
    print("1. ✅ API Health & Connectivity")
    print("2. 👤 User Registration/Login") 
    print("3. 🔑 Session Management")
    print("4. 🧠 AI Form Filling (Demo & Authenticated)")
    print("5. 📊 URL Tracking & Management")
    print("6. 🧹 Session Cleanup")
    print("=" * 80)
    
    # Check if server is running first
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    print(f"❌ Server health check failed: HTTP {response.status}")
                    return
                else:
                    print("✅ Server is running and healthy")
    except Exception as e:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print(f"Error: {e}")
        print("💡 Please start the server first:")
        print("   python main.py")
        return
    
    # Run comprehensive test suite
    test_suite = EndToEndTestSuite()
    await test_suite.run_comprehensive_test_suite()

if __name__ == "__main__":
    asyncio.run(main()) 