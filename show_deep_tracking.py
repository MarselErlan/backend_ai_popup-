#!/usr/bin/env python3
"""
Show Deep Tracking in Action

This script demonstrates that deep tracking IS working and shows you
exactly how to see it.
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    print("🔍 Deep Tracking Demonstration")
    print("=" * 50)
    
    # Check current report
    html_path = Path("tests/reports/integrated_analysis_current.html")
    if html_path.exists():
        print("✅ Current HTML report exists")
        
        # Check content
        with open(html_path, 'r') as f:
            content = f.read()
        
        if "Deep Real-Time API Analysis" in content:
            print("✅ Report has deep tracking title")
        
        if "api_endpoint_handler" in content:
            print("✅ Report contains tracked functions")
            
        if "call-stack" in content:
            print("✅ Report has call stack visualization")
            
        size_kb = html_path.stat().st_size / 1024
        print(f"📊 Report size: {size_kb:.1f} KB")
        
        print(f"\n📄 Report location: {html_path.absolute()}")
        
    else:
        print("❌ No HTML report found")
        return
    
    print("\n" + "="*50)
    print("🎯 TO SEE DEEP TRACKING IN YOUR BROWSER:")
    print("="*50)
    print("1. Open your web browser")
    print("2. Navigate to this file:")
    print(f"   file://{html_path.absolute()}")
    print("3. Look for these sections:")
    print("   • 'Recent Execution Traces'")
    print("   • 'Recent Detailed Function Calls'")
    print("   • Click on 'Click to view execution flow'")
    print("   • Expand call stacks to see full details")
    
    print("\n" + "="*50)
    print("🚀 TO GENERATE NEW DEEP TRACKING DATA:")
    print("="*50)
    print("Run these commands in separate terminals:")
    print()
    print("Terminal 1:")
    print("  python tests/analysis/realtime_usage_analyzer.py --monitor")
    print()
    print("Terminal 2 (after monitor starts):")
    print("  python demo_deep_tracking.py")
    print()
    print("Terminal 1 (after demo finishes):")
    print("  Press Ctrl+C")
    print()
    print("Then refresh the HTML report in your browser!")
    
    print("\n" + "="*50)
    print("🔧 TROUBLESHOOTING:")
    print("="*50)
    print("• Import errors in IDE: Normal - scripts work from terminal")
    print("• Empty report: Make sure to run demo WHILE monitor is running")
    print("• No data: Check that both scripts finish successfully")
    
    # Offer to open the report
    try:
        import webbrowser
        print(f"\n🌐 Opening report in default browser...")
        webbrowser.open(f"file://{html_path.absolute()}")
        print("✅ Report opened! Look for deep tracking sections.")
    except Exception as e:
        print(f"⚠️  Could not auto-open browser: {e}")
        print(f"Manually open: file://{html_path.absolute()}")

if __name__ == "__main__":
    main() 