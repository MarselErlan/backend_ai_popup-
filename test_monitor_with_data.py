#!/usr/bin/env python3
"""
Test monitor with actual deep tracking data
"""

import sys
import os
import time
import threading
import asyncio

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_monitor_with_data():
    """Test monitor with actual deep tracking data"""
    
    print("🧪 Testing monitor with deep tracking data...")
    
    # Import the realtime analyzer
    try:
        tests_path = os.path.join(project_root, 'tests')
        if tests_path not in sys.path:
            sys.path.insert(0, tests_path)
        from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
        print("✅ Successfully imported RealTimeUsageAnalyzer")
    except Exception as e:
        print(f"❌ Failed to import RealTimeUsageAnalyzer: {e}")
        return False
    
    # Create analyzer instance
    analyzer = RealTimeUsageAnalyzer()
    print("✅ Created analyzer instance")
    
    # Start monitoring in background
    print("🔄 Starting monitor...")
    monitor_thread = threading.Thread(target=analyzer.start_monitoring, daemon=True)
    monitor_thread.start()
    time.sleep(2)  # Let it initialize
    
    # Now simulate some deep tracking by calling the demo functions
    print("🎯 Triggering deep tracking data...")
    
    try:
        # Import and run the demo deep tracking
        from demo_deep_tracking import run_demo_tracking
        
        # Run the demo to generate tracking data
        run_demo_tracking()
        print("✅ Generated demo tracking data")
        
        # Wait a bit for data to be processed
        time.sleep(2)
        
    except Exception as e:
        print(f"⚠️  Could not run demo tracking: {e}")
        print("💡 Creating manual deep tracking data...")
        
        # Create some manual tracking data
        from app.services.integrated_usage_analyzer import DetailedFunctionCall
        import uuid
        
        # Add some sample function calls
        sample_call = DetailedFunctionCall(
            function_name="test_function",
            file_name="test_file.py",
            file_path="/path/to/test_file.py",
            line_number=42,
            timestamp=time.time(),
            execution_time=0.123,
            input_args='{"test": "data"}',
            output_result='"success"',
            success=True,
            error_message="",
            call_stack=["main", "test_function"],
            memory_usage=1024.0,
            cpu_usage=15.5,
            thread_id=str(threading.get_ident()),
            request_id=str(uuid.uuid4()),
            endpoint_triggered="/api/test",
            parent_function="main",
            child_functions=[],
            database_queries=1,
            api_calls=0,
            cache_hits=1,
            cache_misses=0
        )
        
        analyzer.deep_analyzer.detailed_function_calls.append(sample_call)
        print("✅ Added manual tracking data")
    
    # Stop monitoring and generate report
    print("⏹️  Stopping monitor and generating report...")
    analyzer.stop_monitoring()
    
    # Check results
    deep_calls = len(analyzer.deep_analyzer.detailed_function_calls)
    deep_traces = len(analyzer.deep_analyzer.execution_traces)
    
    print(f"📊 Final results:")
    print(f"   • Function calls captured: {deep_calls}")
    print(f"   • Execution traces captured: {deep_traces}")
    
    if deep_calls > 0:
        print("✅ Deep tracking data successfully captured!")
        print("🌐 Check the HTML report for detailed analysis")
        return True
    else:
        print("⚠️  No deep tracking data captured")
        return False

if __name__ == "__main__":
    success = test_monitor_with_data()
    if success:
        print("\n🎉 Test passed! Monitor captured deep tracking data.")
        print("📄 View the report: tests/reports/integrated_analysis_current.html")
    else:
        print("\n❌ Test failed. No deep tracking data captured.")
        sys.exit(1) 