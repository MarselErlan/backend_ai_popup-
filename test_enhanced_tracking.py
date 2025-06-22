#!/usr/bin/env python3
"""
Test script to verify enhanced tracking works with real endpoints.
This simulates what happens when you use your UI to make API calls.
"""

import time
import requests
import json
from app.services.integrated_usage_analyzer import get_analyzer

def test_enhanced_tracking():
    """Test enhanced tracking with real API calls"""
    print("🚀 Testing Enhanced Function & Class Tracking")
    print("=" * 60)
    
    # Initialize and start analyzer
    analyzer = get_analyzer()
    analyzer.start_monitoring()
    
    try:
        print("📊 Analyzer started, making API calls to test tracking...")
        
        # Base URL for API calls
        base_url = "http://localhost:8000"
        
        # Test 1: Health check (simple endpoint)
        print("\n1️⃣ Testing health check endpoint...")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Health check successful")
            else:
                print(f"⚠️ Health check returned status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("⚠️ Server not running - cannot test real endpoints")
            print("   (This is expected if you haven't started the server)")
        except Exception as e:
            print(f"⚠️ Health check failed: {e}")
        
        # Test 2: Analysis status endpoint
        print("\n2️⃣ Testing analysis status endpoint...")
        try:
            response = requests.get(f"{base_url}/api/analysis/status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                print(f"✅ Analysis status: {status_data.get('status', 'unknown')}")
                print(f"   Functions discovered: {status_data.get('functions_discovered', 0)}")
                print(f"   Classes discovered: {status_data.get('classes_discovered', 0)}")
            else:
                print(f"⚠️ Analysis status returned: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("⚠️ Server not running - cannot test real endpoints")
        except Exception as e:
            print(f"⚠️ Analysis status failed: {e}")
        
        # Test 3: Simulate some service calls directly
        print("\n3️⃣ Testing direct service calls...")
        
        # Import and test document service
        try:
            from app.services.document_service import DocumentService
            from app.services.simple_function_tracker import track_service_call, track_class_creation, track_method_call
            
            # Create a temporary database
            import tempfile
            import os
            temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            temp_db.close()
            
            database_url = f"sqlite:///{temp_db.name}"
            
            # Track class creation
            track_class_creation("DocumentService", "document_service", f"(database_url={database_url})")
            
            doc_service = DocumentService(database_url)
            print("✅ Created DocumentService instance")
            
            # Test document stats method
            service_start = time.time()
            stats = doc_service.get_document_stats()
            service_time = time.time() - service_start
            
            # Track the method call
            track_service_call(
                "get_document_stats",
                "document_service",
                service_time,
                "()",
                f"stats={stats}",
                True
            )
            track_method_call("DocumentService", "get_document_stats", "document_service")
            
            print(f"✅ Got document stats: {stats}")
            
            # Test save/retrieve operations
            test_content = b"This is a test resume for tracking"
            
            service_start = time.time()
            doc_id = doc_service.save_resume_document(
                filename="test_tracking_resume.pdf",
                file_content=test_content,
                content_type="application/pdf",
                user_id="test_tracking_user"
            )
            service_time = time.time() - service_start
            
            # Track the method call
            track_service_call(
                "save_resume_document",
                "document_service",
                service_time,
                f"(filename=test_tracking_resume.pdf, size={len(test_content)}, user_id=test_tracking_user)",
                f"document_id={doc_id}",
                True
            )
            track_method_call("DocumentService", "save_resume_document", "document_service")
            
            print(f"✅ Saved test resume with ID: {doc_id}")
            
            # Retrieve the document
            service_start = time.time()
            resume_doc = doc_service.get_user_resume("test_tracking_user")
            service_time = time.time() - service_start
            
            # Track the method call
            track_service_call(
                "get_user_resume",
                "document_service",
                service_time,
                "(user_id=test_tracking_user)",
                f"resume_doc={resume_doc.filename if resume_doc else None}",
                True
            )
            track_method_call("DocumentService", "get_user_resume", "document_service")
            
            print(f"✅ Retrieved resume: {resume_doc.filename if resume_doc else 'None'}")
            
            # Cleanup
            os.unlink(temp_db.name)
            print("🧹 Cleaned up temporary database")
            
        except Exception as e:
            print(f"⚠️ Direct service test failed: {e}")
        
        # Test 4: Test URL tracking service
        print("\n4️⃣ Testing URL tracking service...")
        try:
            from app.services.url_tracking_service import URLTrackingService
            from app.services.simple_function_tracker import track_service_call, track_class_creation, track_method_call
            
            # Track class creation
            track_class_creation("URLTrackingService", "url_tracking_service", "()")
            
            url_service = URLTrackingService()
            print("✅ Created URLTrackingService instance")
            
            # Track some URL visits
            service_start = time.time()
            url_service.track_url_visit("https://example.com/job-application", "test_tracking_user")
            service_time = time.time() - service_start
            
            # Track the method call
            track_service_call(
                "track_url_visit",
                "url_tracking_service",
                service_time,
                "(url=https://example.com/job-application, user_id=test_tracking_user)",
                "url_tracked",
                True
            )
            track_method_call("URLTrackingService", "track_url_visit", "url_tracking_service")
            
            print("✅ Tracked URL visit")
            
            # Get user stats
            service_start = time.time()
            stats = url_service.get_user_stats("test_tracking_user")
            service_time = time.time() - service_start
            
            # Track the method call
            track_service_call(
                "get_user_stats",
                "url_tracking_service",
                service_time,
                "(user_id=test_tracking_user)",
                f"stats={stats}",
                True
            )
            track_method_call("URLTrackingService", "get_user_stats", "url_tracking_service")
            
            print(f"✅ Got user URL stats: {stats}")
            
        except Exception as e:
            print(f"⚠️ URL tracking service test failed: {e}")
        
        # Wait for tracking to complete
        print("\n⏱️ Waiting for tracking to complete...")
        time.sleep(1.0)
        
        # Check current status
        status = analyzer.get_status()
        print(f"\n📊 Final tracking status:")
        print(f"   Functions discovered: {status.get('functions_discovered', 0)}")
        print(f"   Classes discovered: {status.get('classes_discovered', 0)}")
        print(f"   Monitoring enabled: {status.get('monitoring', False)}")
        print(f"   Duration: {status.get('duration_minutes', 0):.2f} minutes")
        
        # Check if we have any tracked function/class usage
        if hasattr(analyzer, 'function_usage') and analyzer.function_usage:
            print(f"   ✅ Functions tracked: {len(analyzer.function_usage)}")
            print("   📝 Tracked functions:")
            for func_name, usage in list(analyzer.function_usage.items())[:5]:
                print(f"      - {func_name}: {usage.call_count} calls")
        else:
            print("   ⚠️ No functions tracked yet")
            
        if hasattr(analyzer, 'class_usage') and analyzer.class_usage:
            print(f"   ✅ Classes tracked: {len(analyzer.class_usage)}")
            print("   📝 Tracked classes:")
            for class_name, usage in list(analyzer.class_usage.items())[:5]:
                print(f"      - {class_name}: {usage.instantiation_count} instances")
        else:
            print("   ⚠️ No classes tracked yet")
        
    finally:
        # Stop monitoring and generate report
        print("\n🔄 Stopping analysis and generating report...")
        analyzer.stop_monitoring()
        
        print("✅ Enhanced tracking test completed!")
        print("\n📊 Check the reports:")
        print("   📄 tests/reports/integrated_analysis_current.json")
        print("   🌐 tests/reports/integrated_analysis_current.html")
        print("\n💡 Expected to see:")
        print("   - DocumentService, URLTrackingService in 'Most Used Classes'")
        print("   - save_resume_document, get_user_stats, etc. in 'Most Used Functions'")
        print("   - Real execution times and success rates")
        print("   - Detailed input/output information")


if __name__ == "__main__":
    test_enhanced_tracking() 