#!/usr/bin/env python3
"""
Test Script: Realtime Monitoring with Deep Tracking

This script simulates the scenario where you:
1. Start the realtime analyzer monitoring
2. Generate some function calls
3. Stop monitoring with Ctrl+C
4. Check if deep tracking data is captured
"""

import time
import threading
import signal
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try multiple import methods to work with different environments
try:
    from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
except ImportError:
    try:
        sys.path.insert(0, str(project_root / "tests" / "analysis"))
        from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
    except ImportError:
        # Direct path import as fallback
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "realtime_usage_analyzer", 
            project_root / "tests" / "analysis" / "realtime_usage_analyzer.py"
        )
        realtime_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(realtime_module)
        RealTimeUsageAnalyzer = realtime_module.RealTimeUsageAnalyzer

from app.services.integrated_usage_analyzer import deep_track_function

# Create some functions to track
@deep_track_function
def test_function_1():
    """Test function 1"""
    time.sleep(0.1)
    return "result_1"

@deep_track_function
def test_function_2():
    """Test function 2 - slower"""
    time.sleep(0.5)
    return "result_2"

@deep_track_function
def test_function_3():
    """Test function 3 - with error"""
    time.sleep(0.2)
    if True:  # Simulate conditional error
        raise ValueError("Test error for tracking")

def simulate_activity():
    """Simulate some activity to generate tracking data"""
    print("🔄 Generating function call activity...")
    
    # Call tracked functions
    for i in range(3):
        print(f"   📡 Calling test functions (iteration {i+1})")
        
        # Successful calls
        result1 = test_function_1()
        result2 = test_function_2()
        
        # Error call (caught)
        try:
            test_function_3()
        except ValueError as e:
            print(f"   ⚠️  Expected error caught: {e}")
        
        time.sleep(0.1)
    
    print("✅ Activity simulation complete")

def main():
    """Main test function"""
    print("🧪 Realtime Monitoring Test")
    print("=" * 40)
    
    # Create analyzer
    analyzer = RealTimeUsageAnalyzer()
    
    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\n🛑 Ctrl+C detected - stopping monitoring...")
        analyzer.stop_monitoring()
        
        # Check if we have deep tracking data
        deep_analyzer = analyzer.deep_analyzer
        print(f"\n📊 Deep Tracking Results:")
        print(f"   • Detailed function calls: {len(deep_analyzer.detailed_function_calls)}")
        print(f"   • Execution traces: {len(deep_analyzer.execution_traces)}")
        print(f"   • Active requests: {len(deep_analyzer.active_requests)}")
        
        if deep_analyzer.detailed_function_calls:
            print(f"\n🔍 Recent function calls:")
            for call in deep_analyzer.detailed_function_calls[-5:]:
                status = "✅" if call.success else "❌"
                print(f"   {status} {call.function_name} ({call.execution_time:.3f}s)")
                if call.error_message:
                    print(f"      Error: {call.error_message}")
        
        print(f"\n📄 Check the HTML report at: tests/reports/integrated_analysis_current.html")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("1. 🚀 Starting monitoring...")
    print("2. 🔄 Will simulate activity for 10 seconds")
    print("3. ⏹️  Press Ctrl+C to stop and see results")
    print()
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=analyzer.start_monitoring, daemon=True)
    monitor_thread.start()
    
    # Wait a bit for monitoring to start
    time.sleep(2)
    
    # Simulate activity
    activity_thread = threading.Thread(target=simulate_activity, daemon=True)
    activity_thread.start()
    
    # Keep running until Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main() 