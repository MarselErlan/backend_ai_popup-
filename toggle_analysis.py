#!/usr/bin/env python3
"""
Toggle Usage Analysis Script

This script helps you easily enable/disable the integrated usage analysis.
"""

import os
import sys
from pathlib import Path

def toggle_analysis(enable: bool):
    """Toggle the usage analysis on/off"""
    
    # Create or update .env file
    env_file = Path('.env')
    env_lines = []
    
    # Read existing .env if it exists
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_lines = f.readlines()
    
    # Remove existing ENABLE_USAGE_ANALYSIS line
    env_lines = [line for line in env_lines if not line.startswith('ENABLE_USAGE_ANALYSIS')]
    
    # Add new setting
    env_lines.append(f'ENABLE_USAGE_ANALYSIS={"true" if enable else "false"}\n')
    
    # Write back to .env
    with open(env_file, 'w') as f:
        f.writelines(env_lines)
    
    status = "enabled" if enable else "disabled"
    print(f"✅ Usage analysis {status}")
    print(f"📝 Updated .env file: ENABLE_USAGE_ANALYSIS={'true' if enable else 'false'}")
    
    if enable:
        print("\n📊 Analysis Features:")
        print("  • Real-time API endpoint monitoring")
        print("  • Code discovery (functions, classes, endpoints)")
        print("  • Performance metrics tracking")
        print("  • Report generation on shutdown (Ctrl+C)")
        print("  • Reports saved to tests/reports/")
        print("\n🚀 Start your app: uvicorn main:app --reload")
        print("⏹️  Stop with Ctrl+C to generate reports")
    else:
        print("\n❌ Analysis disabled for production use")
        print("🚀 Start your app: uvicorn main:app --reload")

def main():
    """Main CLI interface"""
    if len(sys.argv) != 2 or sys.argv[1] not in ['on', 'off', 'enable', 'disable', 'true', 'false']:
        print("Usage: python toggle_analysis.py [on|off|enable|disable|true|false]")
        print("\nExamples:")
        print("  python toggle_analysis.py on      # Enable analysis")
        print("  python toggle_analysis.py off     # Disable analysis")
        print("  python toggle_analysis.py enable  # Enable analysis")
        print("  python toggle_analysis.py disable # Disable analysis")
        sys.exit(1)
    
    arg = sys.argv[1].lower()
    enable = arg in ['on', 'enable', 'true']
    
    toggle_analysis(enable)

if __name__ == "__main__":
    main() 