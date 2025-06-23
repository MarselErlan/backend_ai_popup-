#!/usr/bin/env python3
"""
Start FastAPI Server and Monitor

This script:
1. Starts your FastAPI server in the background
2. Waits for it to be ready
3. Runs the realtime monitor
4. When you press Ctrl+C, it shows deep tracking data from actual API calls
"""

import subprocess
import time
import signal
import sys
import requests
from pathlib import Path

def check_server_running(port=8000, max_attempts=10):
    """Check if the FastAPI server is running"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"⏳ Waiting for server... (attempt {attempt + 1}/{max_attempts})")
        time.sleep(2)
    
    return False

def main():
    print("🚀 Starting FastAPI Server and Monitor")
    print("=" * 40)
    
    # Start the FastAPI server
    print("1. 🌐 Starting FastAPI server...")
    
    # You'll need to adjust this command based on how you start your server
    # Common patterns:
    server_commands = [
        ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"],
        ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"],
        ["python", "app/api.py"],
        ["fastapi", "run", "app/api.py"],
    ]
    
    server_process = None
    
    for cmd in server_commands:
        try:
            print(f"   Trying: {' '.join(cmd)}")
            server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent
            )
            
            # Give it a moment to start
            time.sleep(3)
            
            # Check if it's running
            if server_process.poll() is None:  # Still running
                print("   ✅ Server process started")
                break
            else:
                print("   ❌ Server process exited")
                server_process = None
        except FileNotFoundError:
            print(f"   ❌ Command not found: {cmd[0]}")
            continue
    
    if not server_process:
        print("❌ Could not start FastAPI server")
        print("💡 Please start your server manually first, then run:")
        print("   python tests/analysis/realtime_usage_analyzer.py --monitor")
        return
    
    # Wait for server to be ready
    print("2. ⏳ Waiting for server to be ready...")
    if not check_server_running():
        print("❌ Server did not start properly")
        server_process.terminate()
        return
    
    print("✅ Server is ready!")
    
    # Set up signal handler for cleanup
    def cleanup(sig, frame):
        print("\n🛑 Cleaning up...")
        if server_process:
            server_process.terminate()
            server_process.wait()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    
    # Start the monitor
    print("3. 📊 Starting realtime monitor...")
    print("   The monitor will now make requests to your running server")
    print("   This WILL trigger @deep_track_function decorators")
    print("   Press Ctrl+C to stop and see deep tracking results")
    print()
    
    try:
        # Run the monitor
        monitor_process = subprocess.run([
            "python", "tests/analysis/realtime_usage_analyzer.py", "--monitor"
        ], cwd=Path(__file__).parent)
        
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(None, None)

if __name__ == "__main__":
    main() 