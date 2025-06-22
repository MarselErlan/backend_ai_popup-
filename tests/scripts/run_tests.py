#!/usr/bin/env python3
"""
Cross-platform test runner for comprehensive main.py testing
"""

import subprocess
import sys
import time
import os
import signal
import requests
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = ['aiohttp', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)

def is_server_running(url="http://localhost:8000/health", timeout=5):
    """Check if the server is running"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Start the server in background"""
    if not Path("main.py").exists():
        print("❌ Error: main.py not found in current directory")
        return None
    
    print("🚀 Starting server in background...")
    process = subprocess.Popen([sys.executable, "main.py"], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(10):  # Wait up to 10 seconds
        time.sleep(1)
        if is_server_running():
            print("✅ Server started successfully")
            return process
        print(f"   Waiting... ({i+1}/10)")
    
    print("❌ Server failed to start within 10 seconds")
    process.terminate()
    return None

def stop_server(process):
    """Stop the server process"""
    if process:
        print("🧹 Stopping server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("✅ Server stopped")

def main():
    """Main test runner"""
    print("🚀 Smart Form Fill API - Comprehensive Testing Suite")
    print("=" * 60)
    
    # Check requirements
    print("📦 Checking requirements...")
    check_requirements()
    
    # Check if server is already running
    server_process = None
    if is_server_running():
        print("✅ Server is already running - proceeding with tests...")
    else:
        print("⚠️  Server not detected. Starting server...")
        server_process = start_server()
        if not server_process:
            print("❌ Failed to start server. Please start it manually and try again.")
            return 1
    
    try:
        # Run the comprehensive test
        print("\n🧪 Running comprehensive test suite...")
        result = subprocess.run([sys.executable, "test_main_comprehensive.py"], 
                              capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ All tests completed successfully!")
        else:
            print(f"\n⚠️  Tests completed with return code: {result.returncode}")
            
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1
        
    finally:
        # Clean up server if we started it
        if server_process:
            stop_server(server_process)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 