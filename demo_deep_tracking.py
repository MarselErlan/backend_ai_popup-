#!/usr/bin/env python3
"""
Demo: Deep Function Tracking with Real-Time Analysis

This script demonstrates the enhanced deep tracking capabilities:
- Call stack capture
- Execution flow analysis  
- Performance bottleneck detection
- Memory and CPU usage tracking
- Request-level tracing
"""

import time
import threading
from app.services.integrated_usage_analyzer import (
    get_analyzer, 
    start_analysis,
    stop_analysis,
    deep_track_function
)

# Initialize the analyzer with deep tracking enabled
analyzer = get_analyzer()
start_analysis()

print("🔍 Deep Function Tracking Demo")
print("=" * 50)

@deep_track_function
def slow_database_query(query: str, timeout: int = 2):
    """Simulate a slow database query"""
    print(f"📊 Executing query: {query}")
    time.sleep(timeout)  # Simulate slow query
    return {"rows": 42, "execution_time": timeout}

@deep_track_function
def process_user_data(user_id: int, data: dict):
    """Process user data with multiple function calls"""
    print(f"👤 Processing user {user_id}")
    
    # Call other tracked functions
    query_result = slow_database_query(f"SELECT * FROM users WHERE id = {user_id}")
    
    # Simulate some processing
    time.sleep(0.1)
    
    return {
        "user_id": user_id,
        "processed": True,
        "query_result": query_result,
        "timestamp": time.time()
    }

@deep_track_function
def api_endpoint_handler(request_data: dict):
    """Simulate an API endpoint that processes requests"""
    print(f"🎯 API Handler called with: {list(request_data.keys())}")
    
    # Start request trace
    request_id = f"req_{int(time.time())}"
    analyzer.start_request_trace(request_id, "/api/process-user")
    
    # Set thread-local data for tracking
    threading.current_thread().request_id = request_id
    threading.current_thread().endpoint = "/api/process-user"
    
    try:
        # Process multiple users
        results = []
        for user_id in request_data.get("user_ids", [1, 2, 3]):
            result = process_user_data(user_id, {"name": f"User {user_id}"})
            results.append(result)
            
        return {"status": "success", "results": results}
        
    except Exception as e:
        analyzer.end_request_trace(request_id, 500, str(e))
        raise
    else:
        analyzer.end_request_trace(request_id, 200)

@deep_track_function  
def memory_intensive_function():
    """Function that uses significant memory"""
    print("🧠 Running memory-intensive operation")
    # Simulate memory usage
    large_data = [i for i in range(100000)]  # Create large list
    time.sleep(0.5)
    return len(large_data)

def run_demo():
    """Run the deep tracking demo"""
    
    print("\n1. 🚀 Starting Deep Function Tracking...")
    
    # Simulate API calls with different patterns
    print("\n2. 📡 Simulating API Calls...")
    
    # Fast API call
    api_endpoint_handler({"user_ids": [1, 2]})
    
    # Slow API call with more data
    api_endpoint_handler({"user_ids": [3, 4, 5, 6]})
    
    # Memory intensive operation
    memory_intensive_function()
    
    # Another fast call
    api_endpoint_handler({"user_ids": [7]})
    
    print("\n3. 📊 Generating Deep Analysis Report...")
    
    # The report is automatically generated when monitoring stops
    # Access the detailed tracking data directly
    
    print(f"\n4. ✅ Deep Tracking Results:")
    print(f"   • Total detailed function calls: {len(analyzer.detailed_function_calls)}")
    print(f"   • Execution traces captured: {len(analyzer.execution_traces)}")
    print(f"   • Active requests tracked: {len(analyzer.active_requests)}")
    
    # Show some detailed call information
    if analyzer.detailed_function_calls:
        print(f"\n5. 🔍 Recent Function Calls:")
        for call in analyzer.detailed_function_calls[-5:]:
            print(f"   • {call.function_name} ({call.execution_time:.3f}s) - {call.file_name}:{call.line_number}")
            if call.error_message:
                print(f"     ❌ Error: {call.error_message}")
            print(f"     💾 Memory: {call.memory_usage / (1024*1024):.1f}MB, CPU: {call.cpu_usage:.1f}%")
    
    # Show execution traces
    if analyzer.execution_traces:
        print(f"\n6. 🎯 Execution Traces:")
        for trace in analyzer.execution_traces:
            print(f"   • {trace.endpoint} ({trace.total_execution_time:.3f}s)")
            print(f"     Functions called: {len(trace.function_calls)}")
            print(f"     Performance bottlenecks: {len(trace.performance_bottlenecks)}")
            if trace.performance_bottlenecks:
                for bottleneck in trace.performance_bottlenecks:
                    print(f"       ⚠️  {bottleneck['type']}: {bottleneck.get('function', 'unknown')}")
    
    print(f"\n7. 📄 HTML Report Generated:")
    print(f"   📂 Location: tests/reports/integrated_analysis_current.html")
    print(f"   🌐 Open in browser to see deep tracking visualization")
    
    print(f"\n8. 🎉 Deep Tracking Demo Complete!")
    print(f"   The HTML report now shows:")
    print(f"   • 📚 Complete call stacks for each function")
    print(f"   • 🔄 Step-by-step execution flow")
    print(f"   • ⚡ Performance bottleneck analysis")
    print(f"   • 📊 Memory and CPU usage per function")
    print(f"   • 🎯 Request-level execution traces")
    print(f"   • 🧬 Parent-child function relationships")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_analysis()
        print("\n🔍 Deep tracking stopped") 