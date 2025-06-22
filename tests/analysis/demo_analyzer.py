#!/usr/bin/env python3
"""
Demo script showing how to use the Real-Time Usage Analyzer

This script demonstrates:
1. How to discover code structure
2. How to generate a sample report
3. How to use the analyzer programmatically
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from realtime_usage_analyzer import RealTimeUsageAnalyzer

def demo_code_discovery():
    """Demonstrate code discovery capabilities"""
    print("🔍 Demo: Code Discovery")
    print("=" * 50)
    
    analyzer = RealTimeUsageAnalyzer()
    
    print(f"📁 Discovered {len(analyzer.discovered_functions)} functions")
    print(f"🏗️  Discovered {len(analyzer.discovered_classes)} classes")
    print(f"🌐 Discovered {len(analyzer.discovered_endpoints)} endpoints")
    
    print("\n📋 Sample Functions:")
    for func in list(analyzer.discovered_functions)[:10]:
        print(f"   ✓ {func}")
    
    print("\n🏗️  Sample Classes:")
    for cls in list(analyzer.discovered_classes)[:10]:
        print(f"   ✓ {cls}")
    
    print("\n🌐 Sample Endpoints:")
    for endpoint in list(analyzer.discovered_endpoints)[:10]:
        print(f"   ✓ {endpoint}")
    
    print()

def demo_report_generation():
    """Demonstrate report generation"""
    print("📊 Demo: Report Generation")
    print("=" * 50)
    
    analyzer = RealTimeUsageAnalyzer()
    
    # Simulate some endpoint usage
    print("📡 Simulating endpoint usage...")
    
    # Add some fake endpoint usage data
    from realtime_usage_analyzer import EndpointUsage
    
    analyzer.endpoint_usage["GET /"] = EndpointUsage(
        endpoint="GET /",
        method="GET",
        path="/",
        call_count=5,
        response_times=[0.123, 0.145, 0.098, 0.156, 0.134],
        status_codes={200: 5},
        first_called=time.time() - 300,
        last_called=time.time()
    )
    
    analyzer.endpoint_usage["GET /health"] = EndpointUsage(
        endpoint="GET /health",
        method="GET",
        path="/health",
        call_count=3,
        response_times=[0.023, 0.019, 0.025],
        status_codes={200: 3},
        first_called=time.time() - 200,
        last_called=time.time() - 50
    )
    
    analyzer.endpoint_usage["POST /api/v1/upload"] = EndpointUsage(
        endpoint="POST /api/v1/upload",
        method="POST",
        path="/api/v1/upload",
        call_count=2,
        response_times=[1.234, 0.987],
        status_codes={200: 1, 422: 1},
        first_called=time.time() - 100,
        last_called=time.time() - 30
    )
    
    # Generate report
    print("📈 Generating analysis report...")
    report = analyzer.generate_report()
    
    print(f"⏱️  Analysis duration: {report['total_duration_minutes']:.2f} minutes")
    print(f"🌐 Endpoints tested: {len(report['endpoint_usage'])}")
    print(f"📊 Total API calls: {report['performance_metrics']['total_endpoint_calls']}")
    print(f"📈 Endpoint coverage: {report['code_coverage']['endpoint_coverage_percent']:.1f}%")
    print(f"🚫 Unused endpoints: {len(report['unused_code']['endpoints'])}")
    
    # Save report
    print("💾 Saving demo report...")
    analyzer.save_report(report)
    
    print("✅ Demo report generated successfully!")
    print()

def demo_mini_report():
    """Demonstrate mini report functionality"""
    print("📋 Demo: Mini Report")
    print("=" * 50)
    
    analyzer = RealTimeUsageAnalyzer()
    
    # Add some test data
    from realtime_usage_analyzer import EndpointUsage
    
    analyzer.endpoint_usage["GET /docs"] = EndpointUsage(
        endpoint="GET /docs",
        method="GET",
        path="/docs",
        call_count=1,
        response_times=[0.234],
        status_codes={200: 1},
        first_called=time.time(),
        last_called=time.time()
    )
    
    mini_report = analyzer.generate_mini_report()
    
    print("📊 Current Status:")
    for key, value in mini_report.items():
        print(f"   {key}: {value}")
    
    print()

def main():
    """Run all demos"""
    print("🎬 Real-Time Usage Analyzer Demo")
    print("=" * 60)
    print()
    
    # Run demos
    demo_code_discovery()
    demo_mini_report()
    demo_report_generation()
    
    print("🎉 Demo completed!")
    print()
    print("💡 To use the real analyzer:")
    print("   1. Start your FastAPI app: uvicorn main:app --reload")
    print("   2. Run: python tests/analysis/realtime_usage_analyzer.py --monitor")
    print("   3. Use your application")
    print("   4. Stop with Ctrl+C to get the report")

if __name__ == "__main__":
    main() 