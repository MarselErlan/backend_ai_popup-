#!/usr/bin/env python3
"""
Quick Test: Verify Monitor Works

1. Run this script
2. It will start monitoring automatically
3. Generate some activity
4. Stop and show results
"""

import time
import subprocess
import threading
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app.services.integrated_usage_analyzer import deep_track_function

@deep_track_function
def test_function_slow():
    """A slow function to create a bottleneck"""
    print("🐌 Running slow function...")
    time.sleep(1.2)  # This should be detected as slow
    return "slow_result"

@deep_track_function
def test_function_fast():
    """A fast function"""
    print("⚡ Running fast function...")
    time.sleep(0.1)
    return "fast_result"

def generate_test_activity():
    """Generate test activity"""
    print("🔄 Generating test activity...")
    
    for i in range(3):
        print(f"   Cycle {i+1}:")
        test_function_fast()
        if i == 1:  # Add slow function in middle
            test_function_slow()
        test_function_fast()
        time.sleep(0.5)
    
    print("✅ Test activity complete")

def main():
    print("🧪 Quick Monitor Test")
    print("=" * 30)
    print("This will:")
    print("1. Start monitoring")
    print("2. Generate function calls")
    print("3. Stop and show results")
    print()
    
    # Start the monitor process
    print("🚀 Starting monitor...")
    monitor_process = subprocess.Popen([
        sys.executable, 
        "tests/analysis/realtime_usage_analyzer.py", 
        "--monitor"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for monitor to start
    time.sleep(3)
    
    # Generate activity
    generate_test_activity()
    
    # Stop the monitor
    print("\n🛑 Stopping monitor...")
    monitor_process.send_signal(signal.SIGINT)
    
    # Wait for it to finish
    stdout, stderr = monitor_process.communicate(timeout=10)
    
    print("\n📊 Monitor Output:")
    if stdout:
        print(stdout.decode())
    if stderr:
        print("Errors:", stderr.decode())
    
    # Check results
    html_path = Path("tests/reports/integrated_analysis_current.html")
    if html_path.exists():
        print("✅ SUCCESS: HTML report generated!")
        print(f"📄 Location: {html_path}")
        
        # Check content
        with open(html_path, 'r') as f:
            content = f.read()
        
        if "test_function_slow" in content:
            print("✅ Deep tracking captured slow function!")
        if "test_function_fast" in content:  
            print("✅ Deep tracking captured fast function!")
        if "call-stack" in content:
            print("✅ Call stack visualization included!")
            
        print("\n🌐 Open the HTML file to see detailed tracking!")
    else:
        print("❌ No HTML report found")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 