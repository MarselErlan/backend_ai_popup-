#!/usr/bin/env python3
"""
Demo: Integrated Usage Analysis

This script demonstrates the integrated usage analysis feature.
"""

import time
import requests
import subprocess
import os
from pathlib import Path

def main():
    print("🎯 Integrated Usage Analysis Demo")
    print("=" * 50)
    
    # Check if analysis is enabled
    print("\n1️⃣ Checking current analysis status...")
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if 'ENABLE_USAGE_ANALYSIS=true' in content:
                print("✅ Analysis is ENABLED")
            else:
                print("❌ Analysis is DISABLED")
                print("💡 Run: python toggle_analysis.py on")
                return
    else:
        print("⚠️  No .env file found")
        print("💡 Run: python toggle_analysis.py on")
        return
    
    print("\n2️⃣ Starting FastAPI server...")
    print("🚀 uvicorn main:app --reload")
    print("⏳ Waiting for server to start...")
    
    # Note: In a real demo, you'd start the server in background
    # For this demo, we assume the server is already running
    
    print("\n3️⃣ Making test requests...")
    base_url = "http://localhost:8000"
    
    try:
        # Test endpoints
        endpoints = [
            "/",
            "/health", 
            "/docs",
            "/api/analysis/status"
        ]
        
        for endpoint in endpoints:
            print(f"📡 GET {endpoint}")
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                status = "✅" if response.status_code == 200 else "❌"
                print(f"   {status} {response.status_code} ({len(response.content)} bytes)")
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Failed: {e}")
                print("   💡 Make sure your server is running: uvicorn main:app --reload")
                return
        
        print("\n4️⃣ Checking analysis status...")
        try:
            response = requests.get(f"{base_url}/api/analysis/status")
            if response.status_code == 200:
                data = response.json()
                analysis = data['analysis']
                print(f"✅ Analysis Status:")
                print(f"   📊 Enabled: {analysis['enabled']}")
                print(f"   🔄 Monitoring: {analysis['monitoring']}")
                print(f"   ⏱️  Duration: {analysis['duration_minutes']:.1f} minutes")
                print(f"   🌐 Endpoints tracked: {analysis['endpoints_tracked']}")
                print(f"   📈 Total calls: {analysis['total_calls']}")
                print(f"   🔍 Functions discovered: {analysis['functions_discovered']}")
                print(f"   📦 Classes discovered: {analysis['classes_discovered']}")
            else:
                print(f"❌ Status check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Status check failed: {e}")
        
        print("\n5️⃣ Demo complete!")
        print("🎯 To see full analysis:")
        print("   1. Stop server with Ctrl+C")
        print("   2. Open tests/reports/integrated_analysis_current.html")
        print("   3. View detailed JSON: tests/reports/integrated_analysis_current.json")
        
        print("\n💡 Toggle analysis:")
        print("   Enable:  python toggle_analysis.py on")
        print("   Disable: python toggle_analysis.py off")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main() 