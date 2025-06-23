#!/usr/bin/env python3
"""
Test: Ctrl+C Scenario Simulation

This simulates exactly what happens when you:
1. Run: python tests/analysis/realtime_usage_analyzer.py --monitor
2. Generate some activity
3. Press Ctrl+C
4. Check if deep tracking data is shown
"""

import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests" / "analysis"))

# Try multiple import methods to work with different environments
try:
    from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
except ImportError:
    try:
        from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
    except ImportError:
        # Direct path import as fallback
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "realtime_usage_analyzer", 
            project_root / "tests" / "analysis" / "realtime_usage_analyzer.py"
        )
        realtime_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(realtime_module)
        RealTimeUsageAnalyzer = realtime_module.RealTimeUsageAnalyzer

from app.services.integrated_usage_analyzer import deep_track_function

@deep_track_function
def simulate_api_call(endpoint: str, data: dict):
    """Simulate an API call"""
    print(f"📡 API Call: {endpoint}")
    time.sleep(0.3)  # Simulate processing time
    return {"status": "success", "data_processed": len(str(data))}

@deep_track_function
def process_data(data: dict):
    """Simulate data processing"""
    print(f"⚙️  Processing data: {list(data.keys())}")
    time.sleep(0.2)
    return {"processed": True, "items": len(data)}

@deep_track_function
def slow_operation():
    """Simulate a slow operation"""
    print("🐌 Running slow operation...")
    time.sleep(1.0)  # This should be detected as a bottleneck
    return "slow_result"

def main():
    """Main test that simulates the Ctrl+C scenario"""
    print("🧪 Simulating Ctrl+C Scenario")
    print("=" * 40)
    print("This simulates what happens when you:")
    print("1. Run: python tests/analysis/realtime_usage_analyzer.py --monitor")
    print("2. Generate some activity")
    print("3. Press Ctrl+C")
    print()
    
    # Create the realtime analyzer
    analyzer = RealTimeUsageAnalyzer()
    
    print("🚀 Starting monitoring (like when you run --monitor)...")
    
    # Start monitoring (this is what happens when you run --monitor)
    analyzer.monitoring = True
    analyzer.start_time = time.time()
    
    # Start the deep tracking
    from app.services.integrated_usage_analyzer import start_analysis
    start_analysis()
    
    print("✅ Monitoring started!")
    print()
    
    # Simulate some activity (like when your app receives requests)
    print("🔄 Simulating activity...")
    
    # Generate some function calls
    for i in range(3):
        print(f"\n📡 Simulating request batch {i+1}:")
        
        # Simulate API calls
        result1 = simulate_api_call("/api/generate-field-answer", {"field": "name", "context": "resume"})
        result2 = process_data({"user_id": 123, "field": "email", "data": "test@example.com"})
        
        if i == 1:  # Add a slow operation in the middle
            slow_result = slow_operation()
    
    print("\n🔄 Activity complete")
    print()
    
    # Simulate Ctrl+C (this is what happens when you press Ctrl+C)
    print("🛑 Simulating Ctrl+C - stopping monitoring...")
    analyzer.stop_monitoring()
    
    print("\n" + "="*50)
    print("📊 RESULTS (what you should see after Ctrl+C):")
    print("="*50)
    
    # Check the deep tracking data
    deep_analyzer = analyzer.deep_analyzer
    
    print(f"📈 Deep Tracking Summary:")
    print(f"   • Total detailed function calls: {len(deep_analyzer.detailed_function_calls)}")
    print(f"   • Execution traces: {len(deep_analyzer.execution_traces)}")
    print(f"   • Active requests: {len(deep_analyzer.active_requests)}")
    
    if deep_analyzer.detailed_function_calls:
        print(f"\n🔍 Recent Function Calls:")
        for call in deep_analyzer.detailed_function_calls[-10:]:
            status = "✅" if call.success else "❌"
            memory_mb = call.memory_usage / (1024 * 1024)
            print(f"   {status} {call.function_name}")
            print(f"      ⏱️  Time: {call.execution_time:.3f}s")
            print(f"      💾 Memory: {memory_mb:.1f}MB")
            print(f"      📁 File: {call.file_name}:{call.line_number}")
            if call.error_message:
                print(f"      ❌ Error: {call.error_message}")
            print()
    else:
        print("\n❌ NO DEEP TRACKING DATA FOUND!")
        print("   This is the problem you were experiencing.")
    
    print("📄 HTML Report Location: tests/reports/integrated_analysis_current.html")
    print("\n🎉 Test Complete!")
    
    if deep_analyzer.detailed_function_calls:
        print("✅ SUCCESS: Deep tracking data was captured!")
        print("   The HTML report should now show detailed function calls.")
    else:
        print("❌ PROBLEM: No deep tracking data was captured.")
        print("   This indicates the integration needs fixing.")

if __name__ == "__main__":
    main() 