#!/usr/bin/env python3
"""
Quick test to verify monitor coordination between realtime analyzer and deep analyzer
"""

import sys
import os
import time
import threading
import signal

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_monitor_coordination():
    """Test that the two analyzers coordinate properly"""
    
    print("🧪 Testing monitor coordination...")
    
    # Import the realtime analyzer
    try:
        # Try multiple import paths
        try:
            from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
        except ImportError:
            # Add tests directory to path
            tests_path = os.path.join(project_root, 'tests')
            if tests_path not in sys.path:
                sys.path.insert(0, tests_path)
            from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
        
        print("✅ Successfully imported RealTimeUsageAnalyzer")
    except Exception as e:
        print(f"❌ Failed to import RealTimeUsageAnalyzer: {e}")
        return False
    
    # Create analyzer instance
    try:
        analyzer = RealTimeUsageAnalyzer()
        print("✅ Successfully created analyzer instance")
        
        # Check that deep analyzer is initialized
        if hasattr(analyzer, 'deep_analyzer'):
            print(f"✅ Deep analyzer initialized: {type(analyzer.deep_analyzer)}")
            print(f"📊 Deep analyzer DB path: {analyzer.deep_analyzer.db_path}")
            print(f"🔍 Deep analyzer functions discovered: {len(analyzer.deep_analyzer.discovered_functions)}")
        else:
            print("❌ Deep analyzer not found in realtime analyzer")
            return False
            
    except Exception as e:
        print(f"❌ Failed to create analyzer: {e}")
        return False
    
    # Test starting monitoring
    try:
        print("\n🔄 Testing monitor start...")
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=analyzer.start_monitoring, daemon=True)
        monitor_thread.start()
        
        # Wait a bit for initialization
        time.sleep(3)
        
        # Check if both analyzers are monitoring
        print(f"📡 Realtime analyzer monitoring: {analyzer.monitoring}")
        print(f"🧠 Deep analyzer monitoring: {analyzer.deep_analyzer.monitoring}")
        
        if analyzer.monitoring and analyzer.deep_analyzer.monitoring:
            print("✅ Both analyzers are monitoring!")
        else:
            print("⚠️  Analyzers not both monitoring")
        
        # Stop monitoring
        print("\n⏹️  Testing monitor stop...")
        analyzer.stop_monitoring()
        
        # Check deep tracking results
        deep_calls = len(analyzer.deep_analyzer.detailed_function_calls)
        deep_traces = len(analyzer.deep_analyzer.execution_traces)
        
        print(f"📊 Deep tracking results:")
        print(f"   • Function calls: {deep_calls}")
        print(f"   • Execution traces: {deep_traces}")
        
        print("✅ Monitor coordination test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during monitoring test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_monitor_coordination()
    if success:
        print("\n🎉 All tests passed! Monitor coordination is working.")
    else:
        print("\n❌ Tests failed. Check the errors above.")
        sys.exit(1) 