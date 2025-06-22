#!/usr/bin/env python3
"""
Final Comprehensive Test Summary for main.py
Validates functionality and provides actionable insights
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_USER_EMAIL = "final_test@example.com"
TEST_USER_PASSWORD = "finaltest123"

class FinalTestSuite:
    """Final comprehensive test suite"""
    
    def __init__(self):
        self.session = None
        self.test_results = {
            "critical_endpoints": [],
            "working_endpoints": [],
            "broken_endpoints": [],
            "performance_issues": [],
            "recommendations": []
        }
    
    async def setup(self):
        """Setup test session"""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Cleanup test session"""
        if self.session:
            await self.session.close()
    
    async def test_endpoint(self, name: str, method: str, url: str, data: dict = None, headers: dict = None):
        """Test a single endpoint and categorize results"""
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url, headers=headers) as response:
                    result = await response.json()
                    status = response.status
            elif method.upper() == "POST":
                async with self.session.post(url, json=data, headers=headers) as response:
                    result = await response.json()
                    status = response.status
            elif method.upper() == "DELETE":
                async with self.session.delete(url, json=data, headers=headers) as response:
                    result = await response.json()
                    status = response.status
            
            processing_time = time.time() - start_time
            
            # Categorize result
            test_info = {
                "name": name,
                "method": method,
                "url": url,
                "status": status,
                "processing_time": processing_time,
                "response": result
            }
            
            if 200 <= status < 300:
                if processing_time > 5.0:  # Flag slow endpoints
                    self.test_results["performance_issues"].append({
                        **test_info,
                        "issue": f"Slow response time: {processing_time:.2f}s"
                    })
                
                if name in ["health_check", "root", "demo_field_answer"]:
                    self.test_results["critical_endpoints"].append(test_info)
                else:
                    self.test_results["working_endpoints"].append(test_info)
            else:
                self.test_results["broken_endpoints"].append(test_info)
            
            return test_info
            
        except Exception as e:
            error_info = {
                "name": name,
                "method": method,
                "url": url,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            self.test_results["broken_endpoints"].append(error_info)
            return error_info
    
    async def run_critical_tests(self):
        """Test critical endpoints that should always work"""
        print("🔥 Testing Critical Endpoints...")
        
        # Health check - most important
        await self.test_endpoint("health_check", "GET", f"{BASE_URL}/health")
        
        # Root endpoint
        await self.test_endpoint("root", "GET", f"{BASE_URL}/")
        
        # Demo endpoint (main functionality)
        demo_data = {
            "label": "What is your name?",
            "url": "https://example.com/test-form"
        }
        await self.test_endpoint("demo_field_answer", "POST", f"{BASE_URL}/api/demo/generate-field-answer", demo_data)
    
    async def run_auth_tests(self):
        """Test authentication flow"""
        print("🔐 Testing Authentication Flow...")
        
        # Register
        register_data = {
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
        register_result = await self.test_endpoint("register", "POST", f"{BASE_URL}/api/simple/register", register_data)
        
        # Login
        login_data = {
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
        login_result = await self.test_endpoint("login", "POST", f"{BASE_URL}/api/simple/login", login_data)
        
        # Try to get user_id from login
        user_id = None
        if login_result.get("status") == 200:
            user_id = login_result.get("response", {}).get("user_id")
        
        return user_id
    
    async def run_document_tests(self, user_id: str = None):
        """Test document management"""
        print("📄 Testing Document Management...")
        
        if user_id:
            # Test session creation
            session_data = {"user_id": user_id}
            session_result = await self.test_endpoint("create_session", "POST", f"{BASE_URL}/api/session/create", session_data)
            
            # Test with session
            if session_result.get("status") == 200:
                session_id = session_result.get("response", {}).get("session_id")
                headers = {"Authorization": session_id}
                
                # Test document status
                await self.test_endpoint("document_status", "GET", f"{BASE_URL}/api/v1/documents/status", headers=headers)
                
                # Test resume get (should fail - no upload)
                await self.test_endpoint("get_resume", "GET", f"{BASE_URL}/api/v1/resume", headers=headers)
    
    async def run_comprehensive_test(self):
        """Run all tests"""
        print("🚀 Starting Final Comprehensive Test Suite...")
        print("=" * 60)
        
        await self.setup()
        
        try:
            # Critical tests
            await self.run_critical_tests()
            
            # Authentication tests
            user_id = await self.run_auth_tests()
            
            # Document tests
            await self.run_document_tests(user_id)
            
            # Generate recommendations
            self._generate_recommendations()
            
        finally:
            await self.cleanup()
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check critical endpoints
        critical_working = len([ep for ep in self.test_results["critical_endpoints"] if ep.get("status") == 200])
        if critical_working < 3:
            recommendations.append("❌ CRITICAL: Some core endpoints are not working - fix immediately!")
        else:
            recommendations.append("✅ All critical endpoints are working")
        
        # Check performance
        if self.test_results["performance_issues"]:
            recommendations.append(f"⚠️  PERFORMANCE: {len(self.test_results['performance_issues'])} endpoints are slow (>5s)")
            for issue in self.test_results["performance_issues"]:
                recommendations.append(f"   • {issue['name']}: {issue['issue']}")
        
        # Check broken endpoints
        if self.test_results["broken_endpoints"]:
            recommendations.append(f"🔧 FIX NEEDED: {len(self.test_results['broken_endpoints'])} endpoints are broken")
            for broken in self.test_results["broken_endpoints"][:3]:  # Show first 3
                if "error" in broken:
                    recommendations.append(f"   • {broken['name']}: {broken['error']}")
                else:
                    recommendations.append(f"   • {broken['name']}: HTTP {broken.get('status', 'unknown')}")
        
        # Overall health
        total_tested = len(self.test_results["critical_endpoints"]) + len(self.test_results["working_endpoints"]) + len(self.test_results["broken_endpoints"])
        working_count = len(self.test_results["critical_endpoints"]) + len(self.test_results["working_endpoints"])
        
        if total_tested > 0:
            success_rate = (working_count / total_tested) * 100
            if success_rate >= 80:
                recommendations.append(f"🎉 EXCELLENT: {success_rate:.1f}% of endpoints are working")
            elif success_rate >= 60:
                recommendations.append(f"⚠️  GOOD: {success_rate:.1f}% of endpoints are working - room for improvement")
            else:
                recommendations.append(f"❌ POOR: Only {success_rate:.1f}% of endpoints are working - needs attention")
        
        self.test_results["recommendations"] = recommendations
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 FINAL TEST SUMMARY")
        print("=" * 60)
        
        # Critical endpoints
        print(f"\n🔥 CRITICAL ENDPOINTS ({len(self.test_results['critical_endpoints'])}):")
        for ep in self.test_results["critical_endpoints"]:
            status_icon = "✅" if ep.get("status") == 200 else "❌"
            time_str = f"({ep.get('processing_time', 0):.3f}s)"
            print(f"   {status_icon} {ep['name']} {time_str}")
        
        # Working endpoints
        if self.test_results["working_endpoints"]:
            print(f"\n✅ WORKING ENDPOINTS ({len(self.test_results['working_endpoints'])}):")
            for ep in self.test_results["working_endpoints"][:5]:  # Show first 5
                time_str = f"({ep.get('processing_time', 0):.3f}s)"
                print(f"   • {ep['name']} {time_str}")
            if len(self.test_results["working_endpoints"]) > 5:
                print(f"   ... and {len(self.test_results['working_endpoints']) - 5} more")
        
        # Broken endpoints
        if self.test_results["broken_endpoints"]:
            print(f"\n❌ BROKEN ENDPOINTS ({len(self.test_results['broken_endpoints'])}):")
            for ep in self.test_results["broken_endpoints"][:5]:  # Show first 5
                if "error" in ep:
                    print(f"   • {ep['name']}: {ep['error']}")
                else:
                    print(f"   • {ep['name']}: HTTP {ep.get('status', 'unknown')}")
            if len(self.test_results["broken_endpoints"]) > 5:
                print(f"   ... and {len(self.test_results['broken_endpoints']) - 5} more")
        
        # Performance issues
        if self.test_results["performance_issues"]:
            print(f"\n⚠️  PERFORMANCE ISSUES ({len(self.test_results['performance_issues'])}):")
            for issue in self.test_results["performance_issues"]:
                print(f"   • {issue['name']}: {issue['issue']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in self.test_results["recommendations"]:
            print(f"   {rec}")
        
        print("\n" + "=" * 60)

async def main():
    """Main test runner"""
    print("🎯 FINAL COMPREHENSIVE TEST FOR MAIN.PY")
    print("=" * 60)
    
    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    print("✅ Server is running - proceeding with final tests")
                else:
                    print("❌ Server health check failed")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Start the server first: python main.py")
        return
    
    # Run tests
    test_suite = FinalTestSuite()
    await test_suite.run_comprehensive_test()
    test_suite.print_summary()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(test_suite.test_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print("✅ Final testing complete!")

if __name__ == "__main__":
    asyncio.run(main()) 