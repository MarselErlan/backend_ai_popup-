#!/usr/bin/env python3
"""
Simple Verification: Deep Tracking Works

This script verifies that the deep tracking system is working correctly
by running the monitor and generating activity, then checking the results.
"""

import time
import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🔍 Deep Tracking Verification")
    print("=" * 40)
    print("This will verify that deep tracking works when you:")
    print("1. Run the monitor with --monitor")
    print("2. Generate some function calls")
    print("3. Stop with Ctrl+C")
    print()
    
    # Check if the realtime analyzer exists
    analyzer_path = Path("tests/analysis/realtime_usage_analyzer.py")
    if not analyzer_path.exists():
        print("❌ Realtime analyzer not found!")
        return
    
    print("✅ Realtime analyzer found")
    
    # Check if the demo script exists
    demo_path = Path("demo_deep_tracking.py")
    if not demo_path.exists():
        print("❌ Demo script not found!")
        return
        
    print("✅ Demo script found")
    
    # Test 1: Run the demo to generate tracking data
    print("\n📊 Test 1: Generating tracking data...")
    try:
        result = subprocess.run([
            sys.executable, "demo_deep_tracking.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Demo ran successfully")
            if "Deep Tracking Results" in result.stdout:
                print("✅ Deep tracking data was generated")
            else:
                print("⚠️  Demo ran but no tracking data found")
        else:
            print(f"❌ Demo failed: {result.stderr}")
            return
            
    except subprocess.TimeoutExpired:
        print("❌ Demo timed out")
        return
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return
    
    # Test 2: Check if HTML report was generated
    print("\n📄 Test 2: Checking HTML report...")
    html_path = Path("tests/reports/integrated_analysis_current.html")
    
    if html_path.exists():
        print("✅ HTML report exists")
        
        # Read and check content
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            checks = [
                ("Deep Real-Time API Analysis", "Deep tracking title"),
                ("detailed_function_calls", "Detailed function calls"),
                ("call-stack", "Call stack visualization"),
                ("execution_traces", "Execution traces"),
                ("performance_bottlenecks", "Performance bottlenecks"),
                ("memory_usage", "Memory usage tracking"),
                ("api_endpoint_handler", "Demo functions tracked")
            ]
            
            for check_text, description in checks:
                if check_text in content:
                    print(f"✅ {description} found")
                else:
                    print(f"⚠️  {description} not found")
                    
        except Exception as e:
            print(f"❌ Error reading HTML: {e}")
            return
            
    else:
        print("❌ HTML report not found")
        return
    
    # Test 3: Check if monitoring command works
    print("\n🔍 Test 3: Testing monitor command...")
    print("Starting monitor for 5 seconds...")
    
    try:
        # Start monitor
        monitor_proc = subprocess.Popen([
            sys.executable, 
            "tests/analysis/realtime_usage_analyzer.py", 
            "--monitor"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for it to start
        time.sleep(2)
        
        # Run demo while monitor is running
        demo_proc = subprocess.run([
            sys.executable, "demo_deep_tracking.py"
        ], capture_output=True, text=True, timeout=15)
        
        # Stop monitor
        monitor_proc.terminate()
        monitor_stdout, monitor_stderr = monitor_proc.communicate(timeout=10)
        
        if "Deep function call tracking enabled" in monitor_stdout.decode():
            print("✅ Monitor started with deep tracking")
        else:
            print("⚠️  Monitor output unclear")
            
        if demo_proc.returncode == 0:
            print("✅ Demo ran while monitor was active")
        else:
            print("⚠️  Demo had issues while monitor was active")
            
    except Exception as e:
        print(f"⚠️  Monitor test had issues: {e}")
        try:
            monitor_proc.terminate()
        except:
            pass
    
    # Final verification
    print("\n🎉 Final Verification:")
    
    # Check the latest HTML report
    if html_path.exists():
        stat = html_path.stat()
        age_seconds = time.time() - stat.st_mtime
        
        if age_seconds < 300:  # Less than 5 minutes old
            print("✅ HTML report is recent (generated in last 5 minutes)")
            print(f"📄 Report location: {html_path.absolute()}")
            print("🌐 Open this file in your browser to see deep tracking!")
            
            # Show file size as indicator of content
            size_kb = stat.st_size / 1024
            print(f"📊 Report size: {size_kb:.1f} KB")
            
            if size_kb > 50:  # Substantial content
                print("✅ Report has substantial content")
            else:
                print("⚠️  Report seems small - may not have much data")
                
        else:
            print("⚠️  HTML report is old - may not reflect recent changes")
    
    print("\n" + "="*50)
    print("🎯 SUMMARY:")
    print("If you see mostly ✅ above, deep tracking is working!")
    print("If you see ❌ or ⚠️, there may be issues to investigate.")
    print()
    print("💡 To use deep tracking:")
    print("1. Run: python tests/analysis/realtime_usage_analyzer.py --monitor")
    print("2. In another terminal: python demo_deep_tracking.py")
    print("3. Press Ctrl+C in the monitor terminal")
    print("4. Open: tests/reports/integrated_analysis_current.html")

if __name__ == "__main__":
    main() 