#!/usr/bin/env python3
"""
Debug Manual Usage

This script helps debug why manual usage doesn't work by showing
exactly what happens step by step.
"""

import subprocess
import time
import signal
import sys
from pathlib import Path

def debug_monitor_process():
    """Debug the monitor process step by step"""
    
    print("🔍 Debugging Manual Monitor Usage")
    print("=" * 50)
    
    # Step 1: Check if monitor can start
    print("Step 1: Testing monitor startup...")
    try:
        # Start monitor process
        monitor_proc = subprocess.Popen([
            sys.executable, 
            "tests/analysis/realtime_usage_analyzer.py", 
            "--monitor"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("✅ Monitor process started")
        
        # Let it run for a few seconds to see startup output
        time.sleep(3)
        
        # Check if it's still running
        if monitor_proc.poll() is None:
            print("✅ Monitor is running")
        else:
            print("❌ Monitor stopped unexpectedly")
            stdout, stderr = monitor_proc.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return
            
        # Step 2: Check what the monitor is doing
        print("\nStep 2: Checking monitor status...")
        
        # Send SIGTERM to stop gracefully
        monitor_proc.terminate()
        
        # Wait for output
        try:
            stdout, stderr = monitor_proc.communicate(timeout=10)
            print("✅ Monitor stopped gracefully")
            
            # Analyze output
            if "Deep function call tracking enabled" in stdout:
                print("✅ Deep tracking was enabled")
            else:
                print("❌ Deep tracking not enabled")
                
            if "Enhanced HTML report available" in stdout:
                print("✅ HTML report was generated")
            else:
                print("❌ No HTML report mentioned")
                
            # Show last few lines of output
            lines = stdout.strip().split('\n')
            if lines:
                print("\n📄 Last few lines of monitor output:")
                for line in lines[-5:]:
                    if line.strip():
                        print(f"   {line}")
                        
        except subprocess.TimeoutExpired:
            print("⚠️  Monitor didn't stop gracefully, killing...")
            monitor_proc.kill()
            
    except Exception as e:
        print(f"❌ Error starting monitor: {e}")
        return
    
    # Step 3: Check what happens with activity
    print("\nStep 3: Testing with activity generation...")
    
    try:
        # Start monitor again
        monitor_proc = subprocess.Popen([
            sys.executable, 
            "tests/analysis/realtime_usage_analyzer.py", 
            "--monitor"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("✅ Monitor restarted")
        
        # Wait for startup
        time.sleep(2)
        
        # Run demo while monitor is active
        print("🔄 Running demo while monitor is active...")
        demo_result = subprocess.run([
            sys.executable, "demo_deep_tracking.py"
        ], capture_output=True, text=True, timeout=30)
        
        if demo_result.returncode == 0:
            print("✅ Demo ran successfully while monitor was active")
            
            # Check if demo generated tracking data
            if "Deep Tracking Results" in demo_result.stdout:
                print("✅ Demo generated tracking data")
                
                # Extract tracking stats
                lines = demo_result.stdout.split('\n')
                for line in lines:
                    if "Total detailed function calls:" in line:
                        print(f"📊 {line.strip()}")
                    elif "Execution traces captured:" in line:
                        print(f"📊 {line.strip()}")
                        
        else:
            print("❌ Demo failed while monitor was active")
            print(f"Demo error: {demo_result.stderr}")
        
        # Stop monitor
        print("\n🛑 Stopping monitor...")
        monitor_proc.terminate()
        
        stdout, stderr = monitor_proc.communicate(timeout=10)
        
        # Check final output
        if "Enhanced HTML report available" in stdout:
            print("✅ Monitor generated HTML report")
        else:
            print("❌ Monitor didn't mention HTML report")
            
        # Check if report actually exists and has content
        html_path = Path("tests/reports/integrated_analysis_current.html")
        if html_path.exists():
            print("✅ HTML report file exists")
            
            # Check content
            with open(html_path, 'r') as f:
                content = f.read()
            
            if "api_endpoint_handler" in content:
                print("✅ Report contains demo functions")
            else:
                print("❌ Report doesn't contain demo functions")
                
            if "detailed_function_calls" in content:
                print("✅ Report has detailed function calls section")
            else:
                print("❌ Report missing detailed function calls")
                
            size_kb = html_path.stat().st_size / 1024
            print(f"📊 Report size: {size_kb:.1f} KB")
            
        else:
            print("❌ HTML report file not found")
            
    except Exception as e:
        print(f"❌ Error in activity test: {e}")
        try:
            monitor_proc.kill()
        except:
            pass
    
    print("\n" + "="*50)
    print("🎯 DIAGNOSIS:")
    print("="*50)
    
    # Final diagnosis
    html_path = Path("tests/reports/integrated_analysis_current.html")
    if html_path.exists():
        with open(html_path, 'r') as f:
            content = f.read()
            
        if "api_endpoint_handler" in content and "detailed_function_calls" in content:
            print("✅ WORKING: Deep tracking is capturing data correctly")
            print("💡 The issue might be:")
            print("   • You're not generating activity while monitor runs")
            print("   • You're stopping monitor too quickly")
            print("   • You're not checking the right HTML file")
        else:
            print("❌ ISSUE: Deep tracking not capturing data properly")
            print("💡 Possible causes:")
            print("   • Monitor and demo running in isolation")
            print("   • Timing issues")
            print("   • Configuration problems")
    else:
        print("❌ MAJOR ISSUE: No HTML report generated")

def main():
    debug_monitor_process()

if __name__ == "__main__":
    main() 