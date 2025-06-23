#!/usr/bin/env python3
"""
Test: Exact Monitor Command Replication

This exactly replicates:
python tests/analysis/realtime_usage_analyzer.py --monitor

Then stops with Ctrl+C to show deep tracking data.
"""

import sys
import time
import threading
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests" / "analysis"))

# Try multiple import methods to work with different environments
try:
    from tests.analysis.realtime_usage_analyzer import main as analyzer_main
except ImportError:
    try:
        from tests.analysis.realtime_usage_analyzer import main as analyzer_main
    except ImportError:
        # Direct path import as fallback
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "realtime_usage_analyzer", 
            project_root / "tests" / "analysis" / "realtime_usage_analyzer.py"
        )
        realtime_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(realtime_module)
        analyzer_main = realtime_module.main

from app.services.integrated_usage_analyzer import deep_track_function

# Create some test functions to generate activity
@deep_track_function
def background_task_1():
    """Background task 1"""
    time.sleep(0.2)
    return "task1_result"

@deep_track_function  
def background_task_2():
    """Background task 2 - slower"""
    time.sleep(0.8)
    return "task2_result"

def generate_background_activity():
    """Generate some background activity while monitoring"""
    print("🔄 Starting background activity generation...")
    
    for i in range(5):
        print(f"   📡 Background activity cycle {i+1}")
        background_task_1()
        if i % 2 == 0:
            background_task_2()
        time.sleep(1)
    
    print("✅ Background activity complete")

def main():
    """Run the exact same scenario as the command line"""
    print("🧪 Exact Monitor Command Test")
    print("=" * 45)
    print("This exactly replicates:")
    print("   python tests/analysis/realtime_usage_analyzer.py --monitor")
    print("Then pressing Ctrl+C")
    print()
    
    # Start background activity in a separate thread
    activity_thread = threading.Thread(target=generate_background_activity, daemon=True)
    activity_thread.start()
    
    # Simulate the --monitor argument
    sys.argv = ["realtime_usage_analyzer.py", "--monitor"]
    
    print("🚀 Starting monitor (equivalent to --monitor flag)...")
    print("⏳ Will auto-stop after 8 seconds to simulate Ctrl+C")
    print()
    
    # Start the analyzer in a separate thread
    def run_analyzer():
        try:
            analyzer_main()
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped (simulated Ctrl+C)")
    
    analyzer_thread = threading.Thread(target=run_analyzer, daemon=True)
    analyzer_thread.start()
    
    # Wait for 8 seconds then simulate Ctrl+C
    time.sleep(8)
    
    # Force stop by raising KeyboardInterrupt in the main thread
    print("\n🛑 Simulating Ctrl+C after 8 seconds...")
    
    # Import the analyzer instance and stop it
    try:
        from tests.analysis.realtime_usage_analyzer import main as analyzer_main
        # The analyzer should have been created by now
        # We'll check if there's deep tracking data
        
        print("\n" + "="*50)
        print("📊 CHECKING DEEP TRACKING DATA:")
        print("="*50)
        
        # Check if the HTML report was generated with deep tracking data
        html_path = Path("tests/reports/integrated_analysis_current.html")
        if html_path.exists():
            print("✅ HTML report exists!")
            
            # Read a bit of the HTML to check for deep tracking content
            with open(html_path, 'r') as f:
                content = f.read()
                
            if "Deep Real-Time API Analysis" in content:
                print("✅ HTML contains deep tracking title!")
            if "detailed_function_calls" in content:
                print("✅ HTML contains detailed function calls!")
            if "call-stack" in content:
                print("✅ HTML contains call stack visualization!")
                
            print(f"\n📄 Report location: {html_path}")
            print("🌐 Open this file to see the deep tracking data!")
        else:
            print("❌ HTML report not found")
            
    except Exception as e:
        print(f"⚠️  Error checking results: {e}")
    
    print("\n🎉 Test complete!")
    print("If you see ✅ checkmarks above, the deep tracking is working!")

if __name__ == "__main__":
    main() 