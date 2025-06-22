#!/usr/bin/env python3
"""
Performance Monitor for Smart Form Fill API
Tracks optimization metrics and performance improvements
"""

import time
import requests
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    endpoint: str
    response_time: float
    status_code: int
    cache_hits: int = 0
    database_queries: int = 0
    optimization_enabled: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PerformanceMonitor:
    """Monitor API performance and optimization metrics"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.metrics: List[PerformanceMetric] = []
        self.session = requests.Session()
        
    def measure_endpoint(self, endpoint: str, method: str = "POST", payload: Dict = None) -> PerformanceMetric:
        """Measure performance of a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        start_time = time.time()
        try:
            if method.upper() == "POST":
                response = self.session.post(url, json=payload or {})
            else:
                response = self.session.get(url, params=payload or {})
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract performance metrics from response
            cache_hits = 0
            database_queries = 0
            optimization_enabled = False
            
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    data = response.json()
                    if 'performance_metrics' in data:
                        metrics = data['performance_metrics']
                        cache_hits = metrics.get('cache_hits', 0)
                        database_queries = metrics.get('database_queries', 0)
                        optimization_enabled = metrics.get('optimization_enabled', False)
                except:
                    pass
            
            metric = PerformanceMetric(
                endpoint=endpoint,
                response_time=response_time,
                status_code=response.status_code,
                cache_hits=cache_hits,
                database_queries=database_queries,
                optimization_enabled=optimization_enabled
            )
            
            self.metrics.append(metric)
            
            logger.info(f"📊 {endpoint}: {response_time:.2f}s (Status: {response.status_code})")
            if optimization_enabled:
                logger.info(f"   ⚡ Cache hits: {cache_hits}, DB queries: {database_queries}")
            
            return metric
            
        except Exception as e:
            logger.error(f"❌ Error measuring {endpoint}: {e}")
            return PerformanceMetric(
                endpoint=endpoint,
                response_time=999.0,
                status_code=500
            )
    
    def run_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test"""
        logger.info("🚀 Starting performance test suite...")
        
        # Test field answer generation
        field_answer_payload = {
            "label": "Full name",
            "url": "https://example.com/job-application",
            "user_id": "default"
        }
        
        # Test multiple field requests to measure caching
        logger.info("📝 Testing field answer generation...")
        for i in range(3):
            metric = self.measure_endpoint("/api/generate-field-answer", "POST", field_answer_payload)
            time.sleep(0.5)  # Small delay between requests
        
        # Test document status
        logger.info("📋 Testing document status...")
        self.measure_endpoint("/api/v1/documents/status", "GET", {"user_id": "default"})
        
        # Test health check
        logger.info("❤️ Testing health check...")
        self.measure_endpoint("/health", "GET")
        
        # Test root endpoint
        logger.info("🏠 Testing root endpoint...")
        self.measure_endpoint("/", "GET")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        # Group metrics by endpoint
        endpoint_metrics = {}
        for metric in self.metrics:
            if metric.endpoint not in endpoint_metrics:
                endpoint_metrics[metric.endpoint] = []
            endpoint_metrics[metric.endpoint].append(metric)
        
        report = {
            "test_summary": {
                "total_requests": len(self.metrics),
                "test_duration": f"{(self.metrics[-1].timestamp - self.metrics[0].timestamp).total_seconds():.2f}s",
                "successful_requests": len([m for m in self.metrics if m.status_code < 400]),
                "failed_requests": len([m for m in self.metrics if m.status_code >= 400]),
                "average_response_time": f"{sum(m.response_time for m in self.metrics) / len(self.metrics):.2f}s"
            },
            "endpoint_performance": {},
            "optimization_metrics": {
                "total_cache_hits": sum(m.cache_hits for m in self.metrics),
                "total_database_queries": sum(m.database_queries for m in self.metrics),
                "optimization_enabled_requests": len([m for m in self.metrics if m.optimization_enabled]),
                "cache_hit_ratio": 0.0
            },
            "recommendations": []
        }
        
        # Calculate cache hit ratio
        total_operations = report["optimization_metrics"]["total_cache_hits"] + report["optimization_metrics"]["total_database_queries"]
        if total_operations > 0:
            report["optimization_metrics"]["cache_hit_ratio"] = report["optimization_metrics"]["total_cache_hits"] / total_operations
        
        # Analyze each endpoint
        for endpoint, metrics in endpoint_metrics.items():
            response_times = [m.response_time for m in metrics]
            
            endpoint_report = {
                "total_requests": len(metrics),
                "average_response_time": f"{sum(response_times) / len(response_times):.2f}s",
                "min_response_time": f"{min(response_times):.2f}s",
                "max_response_time": f"{max(response_times):.2f}s",
                "success_rate": f"{(len([m for m in metrics if m.status_code < 400]) / len(metrics) * 100):.1f}%",
                "cache_hits": sum(m.cache_hits for m in metrics),
                "database_queries": sum(m.database_queries for m in metrics)
            }
            
            report["endpoint_performance"][endpoint] = endpoint_report
            
            # Generate recommendations
            avg_time = sum(response_times) / len(response_times)
            if avg_time > 2.0:
                report["recommendations"].append(f"⚠️ {endpoint} is slow (avg: {avg_time:.2f}s) - consider further optimization")
            elif avg_time < 0.5:
                report["recommendations"].append(f"✅ {endpoint} has excellent performance (avg: {avg_time:.2f}s)")
        
        # General recommendations
        if report["optimization_metrics"]["cache_hit_ratio"] > 0.7:
            report["recommendations"].append("✅ Excellent cache performance - optimization working well")
        elif report["optimization_metrics"]["cache_hit_ratio"] > 0.3:
            report["recommendations"].append("⚡ Good cache performance - room for improvement")
        else:
            report["recommendations"].append("❌ Low cache hit ratio - check caching implementation")
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted performance report"""
        print("\n" + "="*80)
        print("🎯 SMART FORM FILL API - PERFORMANCE REPORT")
        print("="*80)
        
        # Test Summary
        print(f"\n📊 TEST SUMMARY:")
        summary = report["test_summary"]
        print(f"   • Total Requests: {summary['total_requests']}")
        print(f"   • Test Duration: {summary['test_duration']}")
        print(f"   • Success Rate: {summary['successful_requests']}/{summary['total_requests']}")
        print(f"   • Average Response Time: {summary['average_response_time']}")
        
        # Optimization Metrics
        print(f"\n⚡ OPTIMIZATION METRICS:")
        opt = report["optimization_metrics"]
        print(f"   • Cache Hits: {opt['total_cache_hits']}")
        print(f"   • Database Queries: {opt['total_database_queries']}")
        print(f"   • Cache Hit Ratio: {opt['cache_hit_ratio']:.2%}")
        print(f"   • Optimized Requests: {opt['optimization_enabled_requests']}")
        
        # Endpoint Performance
        print(f"\n🎯 ENDPOINT PERFORMANCE:")
        for endpoint, perf in report["endpoint_performance"].items():
            print(f"   {endpoint}:")
            print(f"     ⏱️  Avg Response: {perf['average_response_time']}")
            print(f"     📈 Success Rate: {perf['success_rate']}")
            print(f"     🔄 Cache Hits: {perf['cache_hits']}")
            print(f"     💾 DB Queries: {perf['database_queries']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   {rec}")
        
        print("\n" + "="*80)
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save performance report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Performance report saved: {filename}")


async def main():
    """Run performance monitoring"""
    print("🚀 Smart Form Fill API - Performance Monitor")
    print("=" * 50)
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Check if API is running
    try:
        response = requests.get(f"{monitor.base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding properly")
            return
    except requests.exceptions.RequestException:
        print("❌ API is not running. Please start the API first with: uvicorn main:app --reload")
        return
    
    print("✅ API is running - starting performance tests...")
    
    # Run performance test
    report = monitor.run_performance_test()
    
    # Print results
    monitor.print_report(report)
    
    # Save report
    monitor.save_report(report)
    
    print("\n🎉 Performance testing completed!")


if __name__ == "__main__":
    asyncio.run(main()) 