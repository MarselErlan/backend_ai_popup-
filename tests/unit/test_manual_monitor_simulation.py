#!/usr/bin/env python3
"""
Manual Monitor Simulation Test

This simulates exactly what happens when you run:
python tests/analysis/realtime_usage_analyzer.py --monitor

And explains why you don't see deep tracking data.
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def simulate_manual_monitor():
    """Simulate the manual monitor scenario"""
    
    print("🧪 Simulating Manual Monitor Scenario")
    print("=" * 50)
    print("This simulates: python tests/analysis/realtime_usage_analyzer.py --monitor")
    print()
    
    # Import the analyzer
    try:
        from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
        print("✅ Successfully imported RealTimeUsageAnalyzer")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Create analyzer (this is what happens when you run --monitor)
    print("🔄 Creating analyzer instance...")
    analyzer = RealTimeUsageAnalyzer()
    
    print("📊 Initial state:")
    print(f"   • Deep analyzer functions: {len(analyzer.deep_analyzer.detailed_function_calls)}")
    print(f"   • Deep analyzer traces: {len(analyzer.deep_analyzer.execution_traces)}")
    print()
    
    # Start monitoring (this is what happens when --monitor starts)
    print("🚀 Starting monitoring (like --monitor does)...")
    
    # Start in background thread
    monitor_thread = threading.Thread(target=analyzer.start_monitoring, daemon=True)
    monitor_thread.start()
    
    # Wait for initialization
    time.sleep(3)
    
    print("✅ Monitor started!")
    print(f"   • Realtime analyzer monitoring: {analyzer.monitoring}")
    print(f"   • Deep analyzer monitoring: {analyzer.deep_analyzer.monitoring}")
    print()
    
    # This is the key issue: The monitor makes HTTP requests to localhost:8000
    # but there's no FastAPI server running, so no @deep_track_function decorators are triggered
    print("🔍 What the monitor is doing:")
    print("   • Making HTTP requests to http://localhost:8000")
    print("   • Trying endpoints like /health, /docs, /api/v1/upload, etc.")
    print("   • These requests FAIL because no server is running")
    print("   • No @deep_track_function decorators get triggered")
    print("   • Therefore: NO deep tracking data is captured")
    print()
    
    # Let it run for a bit to simulate what you experience
    print("⏳ Letting monitor run for 8 seconds (like when you wait before Ctrl+C)...")
    time.sleep(8)
    
    # Stop monitoring (this is what happens when you press Ctrl+C)
    print("🛑 Stopping monitoring (simulating Ctrl+C)...")
    analyzer.stop_monitoring()
    
    # Check results (this is what you see in the terminal)
    print("\n" + "=" * 50)
    print("📊 RESULTS (what you see after Ctrl+C):")
    print("=" * 50)
    
    deep_calls = len(analyzer.deep_analyzer.detailed_function_calls)
    deep_traces = len(analyzer.deep_analyzer.execution_traces)
    
    print(f"📈 Deep Tracking Results:")
    print(f"   • Detailed function calls captured: {deep_calls}")
    print(f"   • Execution traces captured: {deep_traces}")
    
    if deep_calls == 0:
        print("\n❌ NO DEEP TRACKING DATA CAPTURED!")
        print("🔍 Why this happens:")
        print("   1. Monitor tries to call http://localhost:8000 endpoints")
        print("   2. No FastAPI server is running on port 8000")
        print("   3. HTTP requests fail (connection refused)")
        print("   4. No actual Python functions with @deep_track_function get called")
        print("   5. Deep tracking system has nothing to track")
        print()
        print("💡 SOLUTION:")
        print("   Option 1: Start your FastAPI server first, then run monitor")
        print("   Option 2: Use the test scripts that generate mock data")
        print("   Option 3: Add mock function calls to the monitor itself")
    else:
        print("✅ Deep tracking data was captured!")
    
    print(f"\n📄 HTML Report: tests/reports/integrated_analysis_current.html")
    print("   This report will show 'No Deep Tracking Data Available'")
    print("   because no @deep_track_function decorators were triggered")

def test_with_mock_functions():
    """Test what happens when we actually trigger deep tracking functions"""
    
    print("\n" + "=" * 50)
    print("🧪 TESTING WITH MOCK FUNCTIONS")
    print("=" * 50)
    
    from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
    from app.services.integrated_usage_analyzer import deep_track_function
    
    # Create mock functions with deep tracking
    @deep_track_function
    def mock_api_call():
        """Mock API call"""
        time.sleep(0.1)
        return "api_result"
    
    @deep_track_function
    def mock_data_processing():
        """Mock data processing"""
        time.sleep(0.2)
        return "processed_data"
    
    # Create analyzer
    analyzer = RealTimeUsageAnalyzer()
    
    # Start monitoring
    monitor_thread = threading.Thread(target=analyzer.start_monitoring, daemon=True)
    monitor_thread.start()
    time.sleep(2)
    
    print("🔄 Calling mock functions (this WILL generate deep tracking data)...")
    
    # Call the mock functions - this WILL be tracked
    for i in range(3):
        print(f"   📡 Mock call {i+1}")
        mock_api_call()
        mock_data_processing()
    
    time.sleep(1)
    
    # Stop monitoring
    analyzer.stop_monitoring()
    
    # Check results
    deep_calls = len(analyzer.deep_analyzer.detailed_function_calls)
    print(f"\n📊 Results with mock functions:")
    print(f"   • Detailed function calls captured: {deep_calls}")
    
    if deep_calls > 0:
        print("✅ SUCCESS! Deep tracking captured the mock function calls")
        print("🔍 Recent functions:")
        for call in analyzer.deep_analyzer.detailed_function_calls[-3:]:
            print(f"   • {call.function_name} ({call.execution_time:.3f}s)")
    else:
        print("❌ Still no data captured")

if __name__ == "__main__":
    simulate_manual_monitor()
    test_with_mock_functions()
    
    print("\n" + "=" * 70)
    print("🎯 CONCLUSION:")
    print("=" * 70)
    print("The manual monitor doesn't show deep tracking data because:")
    print("1. It makes HTTP requests to endpoints that don't exist")
    print("2. No @deep_track_function decorated functions get called")
    print("3. The deep tracking system needs actual Python function calls to work")
    print()
    print("To see deep tracking data, you need to:")
    print("• Start your FastAPI server first")
    print("• Or use test scripts that call @deep_track_function functions")
    print("• Or modify the monitor to include mock function calls") 