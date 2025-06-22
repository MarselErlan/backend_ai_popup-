#!/usr/bin/env python3
"""
Comprehensive Test Suite for main.py
Tests all endpoints, analyzes function usage, and provides detailed reports
"""

import asyncio
import aiohttp
import json
import time
import os
import ast
import inspect
from typing import Dict, List, Set, Any
from datetime import datetime
import sys

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword123"

class MainPyAnalyzer:
    """Analyzes main.py for function usage and code health"""
    
    def __init__(self, file_path: str = "main.py"):
        self.file_path = file_path
        self.functions_defined = set()
        self.functions_called = set()
        self.classes_defined = set()
        self.endpoints_defined = set()
        self.imports_used = set()
        
    def analyze_code(self):
        """Parse and analyze the main.py file"""
        try:
            with open(self.file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Find all function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.functions_defined.add(node.name)
                    
                    # Check if it's an endpoint
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if hasattr(decorator.func, 'attr'):
                                if decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                                    if len(decorator.args) > 0:
                                        endpoint = ast.literal_eval(decorator.args[0])
                                        self.endpoints_defined.add(f"{decorator.func.attr.upper()} {endpoint}")
                
                elif isinstance(node, ast.ClassDef):
                    self.classes_defined.add(node.name)
                
                elif isinstance(node, ast.Call):
                    # Find function calls
                    if isinstance(node.func, ast.Name):
                        self.functions_called.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        self.functions_called.add(node.func.attr)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports_used.add(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            self.imports_used.add(f"{node.module}.{alias.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing code: {e}")
            return False
    
    def get_unused_functions(self) -> Set[str]:
        """Get functions that are defined but never called"""
        # Exclude special methods and endpoints
        exclude_patterns = ['__', 'lifespan', 'get_', 'create_', 'delete_', 'update_', 'validate_']
        
        unused = set()
        for func in self.functions_defined:
            if func not in self.functions_called:
                # Check if it's not a special function
                if not any(pattern in func for pattern in exclude_patterns):
                    unused.add(func)
        
        return unused

class ComprehensiveApiTester:
    """Comprehensive API testing suite"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        self.user_id = None
        self.session_id = None
        self.test_results = {
            "endpoints_tested": 0,
            "endpoints_passed": 0,
            "endpoints_failed": 0,
            "errors": [],
            "warnings": [],
            "performance": {}
        }
    
    async def setup_session(self):
        """Create aiohttp session"""
        self.session = aiohttp.ClientSession()
    
    async def cleanup_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def test_endpoint(self, method: str, endpoint: str, data: dict = None, 
                          headers: dict = None, files=None) -> Dict[str, Any]:
        """Test a single endpoint"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                async with self.session.get(url, headers=headers) as response:
                    result = await response.json()
                    status = response.status
            elif method.upper() == "POST":
                if files:
                    async with self.session.post(url, data=data, headers=headers) as response:
                        result = await response.json()
                        status = response.status
                else:
                    async with self.session.post(url, json=data, headers=headers) as response:
                        result = await response.json()
                        status = response.status
            elif method.upper() == "DELETE":
                async with self.session.delete(url, json=data, headers=headers) as response:
                    result = await response.json()
                    status = response.status
            else:
                return {"error": f"Method {method} not implemented in test"}
            
            processing_time = time.time() - start_time
            
            self.test_results["endpoints_tested"] += 1
            
            if 200 <= status < 300:
                self.test_results["endpoints_passed"] += 1
                return {
                    "status": "PASS",
                    "http_status": status,
                    "response": result,
                    "processing_time": processing_time
                }
            else:
                self.test_results["endpoints_failed"] += 1
                return {
                    "status": "FAIL",
                    "http_status": status,
                    "response": result,
                    "processing_time": processing_time
                }
                
        except Exception as e:
            self.test_results["endpoints_failed"] += 1
            self.test_results["errors"].append(f"{method} {endpoint}: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🚀 Starting Comprehensive API Test Suite...")
        
        await self.setup_session()
        
        try:
            # Test basic endpoints
            basic_tests = await self.test_basic_endpoints()
            
            # Test authentication
            auth_tests = await self.test_authentication_flow()
            
            # Test documents
            doc_tests = await self.test_document_endpoints()
            
            # Test field answers
            field_tests = await self.test_field_answer_endpoints()
            
            results = {
                "test_start_time": datetime.now().isoformat(),
                "basic_endpoints": basic_tests,
                "authentication": auth_tests,
                "document_management": doc_tests,
                "field_answers": field_tests,
                "test_summary": self.test_results
            }
            
            results["test_end_time"] = datetime.now().isoformat()
            
            return results
            
        finally:
            await self.cleanup_session()
    
    async def test_basic_endpoints(self) -> Dict[str, Any]:
        """Test basic endpoints that don't require authentication"""
        print("🧪 Testing Basic Endpoints...")
        
        basic_tests = {}
        
        # Health check
        basic_tests["health_check"] = await self.test_endpoint("GET", "/health")
        
        # Root endpoint
        basic_tests["root"] = await self.test_endpoint("GET", "/")
        
        return basic_tests
    
    async def test_authentication_flow(self) -> Dict[str, Any]:
        """Test complete authentication flow"""
        print("🔐 Testing Authentication Flow...")
        
        auth_tests = {}
        
        # Register user
        register_data = {
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
        auth_tests["register"] = await self.test_endpoint("POST", "/api/simple/register", register_data)
        
        # Login user
        login_data = {
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
        auth_tests["login"] = await self.test_endpoint("POST", "/api/simple/login", login_data)
        
        return auth_tests
    
    async def test_document_endpoints(self) -> Dict[str, Any]:
        """Test document endpoints"""
        print("📄 Testing Document Endpoints...")
        
        doc_tests = {}
        
        # Test without auth first
        doc_tests["status_no_auth"] = await self.test_endpoint("GET", "/api/v1/documents/status")
        
        return doc_tests
    
    async def test_field_answer_endpoints(self) -> Dict[str, Any]:
        """Test field answer generation endpoints"""
        print("🧠 Testing Field Answer Endpoints...")
        
        field_tests = {}
        
        # Test demo endpoint (no auth required)
        demo_data = {
            "label": "What is your name?",
            "url": "https://example.com/form"
        }
        field_tests["demo_field_answer"] = await self.test_endpoint("POST", "/api/demo/generate-field-answer", demo_data)
        
        return field_tests

async def main():
    """Main test runner"""
    print("=" * 80)
    print("🧪 COMPREHENSIVE MAIN.PY ANALYSIS AND TESTING SUITE")
    print("=" * 80)
    
    # 1. Code Analysis
    print("\n📊 STEP 1: CODE ANALYSIS")
    print("-" * 40)
    
    analyzer = MainPyAnalyzer()
    if analyzer.analyze_code():
        print(f"✅ Functions Defined: {len(analyzer.functions_defined)}")
        print(f"✅ Classes Defined: {len(analyzer.classes_defined)}")
        print(f"✅ Endpoints Defined: {len(analyzer.endpoints_defined)}")
        
        unused = analyzer.get_unused_functions()
        print(f"⚠️  Potentially Unused Functions: {len(unused)}")
        
        if unused:
            print("🔍 Potentially Unused Functions:")
            for func in unused:
                print(f"   • {func}")
        
        print("\n📍 Defined Endpoints:")
        for endpoint in sorted(analyzer.endpoints_defined):
            print(f"   • {endpoint}")
    
    # 2. API Testing
    print("\n🧪 STEP 2: API ENDPOINT TESTING")
    print("-" * 40)
    
    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    print("✅ Server is running - proceeding with tests")
                else:
                    print("❌ Server health check failed")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print("💡 Please start the server first: python main.py")
        return
    
    # Run comprehensive API tests
    tester = ComprehensiveApiTester(BASE_URL)
    test_results = await tester.run_comprehensive_test()
    
    # 3. Generate Reports
    print("\n📋 STEP 3: TEST RESULTS SUMMARY")
    print("-" * 40)
    
    summary = test_results["test_summary"]
    print(f"✅ Endpoints Tested: {summary['endpoints_tested']}")
    print(f"✅ Endpoints Passed: {summary['endpoints_passed']}")
    print(f"❌ Endpoints Failed: {summary['endpoints_failed']}")
    
    if summary['errors']:
        print("\n❌ ERRORS ENCOUNTERED:")
        for error in summary['errors']:
            print(f"   • {error}")
    
    print("\n📊 DETAILED TEST RESULTS:")
    print("-" * 40)
    
    for category, tests in test_results.items():
        if category in ['test_start_time', 'test_end_time', 'test_summary']:
            continue
            
        print(f"\n{category.upper().replace('_', ' ')}:")
        if isinstance(tests, dict):
            for test_name, result in tests.items():
                if isinstance(result, dict) and 'status' in result:
                    status_icon = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⚠️"
                    time_info = f" ({result.get('processing_time', 0):.3f}s)" if 'processing_time' in result else ""
                    print(f"   {status_icon} {test_name}{time_info}")
                    
                    if result['status'] in ["FAIL", "ERROR"] and 'response' in result:
                        print(f"      └─ Error: {result['response']}")
    
    # 4. Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"main_py_test_report_{timestamp}.json"
    
    full_report = {
        "code_analysis": {
            "functions_defined": list(analyzer.functions_defined),
            "classes_defined": list(analyzer.classes_defined),
            "endpoints_defined": list(analyzer.endpoints_defined),
            "unused_functions": list(analyzer.get_unused_functions())
        },
        "api_test_results": test_results,
        "generated_at": datetime.now().isoformat()
    }
    
    with open(report_file, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    
    success_rate = (summary['endpoints_passed'] / summary['endpoints_tested'] * 100) if summary['endpoints_tested'] > 0 else 0
    print(f"\n🎯 API Success Rate: {success_rate:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 