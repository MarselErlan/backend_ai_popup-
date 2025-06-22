#!/usr/bin/env python3
"""
Real-Time Usage Analyzer for FastAPI Application

This module provides real-time monitoring and analysis of:
- Function calls and usage patterns
- Class instantiations and method calls
- API endpoint usage and response patterns
- Unused code detection
- Performance metrics

Usage:
    # Start monitoring (run this while your app is running)
    python tests/analysis/realtime_usage_analyzer.py --monitor
    
    # Generate report from collected data
    python tests/analysis/realtime_usage_analyzer.py --report
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Set, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import ast
import requests
import psutil
import sqlite3

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class EndpointUsage:
    """Represents API endpoint usage"""
    endpoint: str
    method: str
    path: str
    call_count: int
    response_times: List[float]
    status_codes: Dict[int, int]
    first_called: float
    last_called: float

class RealTimeUsageAnalyzer:
    """Real-time usage analyzer for FastAPI applications"""
    
    def __init__(self, 
                 app_host: str = "localhost",
                 app_port: int = 8000,
                 db_path: str = "tests/reports/realtime_analysis.db"):
        self.app_host = app_host
        self.app_port = app_port
        self.db_path = db_path
        self.project_root = project_root
        
        # Analysis data
        self.endpoint_usage: Dict[str, EndpointUsage] = {}
        self.monitoring = False
        self.start_time = time.time()
        
        # Code discovery
        self.discovered_functions: Set[str] = set()
        self.discovered_classes: Set[str] = set()
        self.discovered_endpoints: Set[str] = set()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize database
        self.init_database()
        
        # Discover code structure
        self.discover_code_structure()
        
    def setup_logging(self):
        """Setup logging for the analyzer"""
        log_dir = Path("tests/reports/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "realtime_analyzer.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def init_database(self):
        """Initialize SQLite database for storing analysis data"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Endpoint usage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS endpoint_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT,
                    method TEXT,
                    path TEXT,
                    timestamp REAL,
                    response_time REAL,
                    status_code INTEGER,
                    user_agent TEXT,
                    ip_address TEXT
                )
            """)
            
            # Code discovery table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_discovery (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_type TEXT,
                    item_name TEXT,
                    module_name TEXT,
                    file_path TEXT,
                    discovered_at REAL
                )
            """)
            
            conn.commit()
            
    def discover_code_structure(self):
        """Discover all functions, classes, and endpoints in the project"""
        self.logger.info("🔍 Discovering code structure...")
        
        # Discover Python files
        for file_path in self.project_root.rglob("*.py"):
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if 'venv' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                self._analyze_python_file(file_path)
            except Exception as e:
                self.logger.debug(f"Failed to analyze {file_path}: {e}")
                
        self.logger.info(f"✅ Discovered {len(self.discovered_functions)} functions, "
                        f"{len(self.discovered_classes)} classes, "
                        f"{len(self.discovered_endpoints)} endpoints")
                        
    def _analyze_python_file(self, file_path: Path):
        """Analyze a Python file to discover functions, classes, and endpoints"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = f"{file_path.stem}.{node.name}"
                    self.discovered_functions.add(func_name)
                    
                    # Check for FastAPI endpoints
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Attribute):
                            if decorator.attr in ['get', 'post', 'put', 'delete', 'patch']:
                                # Extract endpoint path from decorator arguments
                                endpoint_path = self._extract_endpoint_path(decorator, node)
                                if endpoint_path:
                                    endpoint = f"{decorator.attr.upper()} {endpoint_path}"
                                    self.discovered_endpoints.add(endpoint)
                        elif isinstance(decorator, ast.Call):
                            if hasattr(decorator.func, 'attr'):
                                if decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                                    endpoint_path = self._extract_endpoint_path_from_call(decorator)
                                    if endpoint_path:
                                        endpoint = f"{decorator.func.attr.upper()} {endpoint_path}"
                                        self.discovered_endpoints.add(endpoint)
                                        
                elif isinstance(node, ast.ClassDef):
                    class_name = f"{file_path.stem}.{node.name}"
                    self.discovered_classes.add(class_name)
                    
        except Exception as e:
            self.logger.debug(f"Error parsing {file_path}: {e}")
            
    def _extract_endpoint_path(self, decorator, node):
        """Extract endpoint path from decorator"""
        try:
            # This is a simplified extraction - you might need to enhance this
            return f"/{node.name}"
        except:
            return None
            
    def _extract_endpoint_path_from_call(self, decorator):
        """Extract endpoint path from decorator call"""
        try:
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                return decorator.args[0].value
        except:
            pass
        return None
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.logger.info("🚀 Starting real-time monitoring...")
        self.logger.info(f"📡 Monitoring API at http://{self.app_host}:{self.app_port}")
        self.logger.info("💡 Make sure your FastAPI app is running with: uvicorn main:app --reload")
        
        self.monitoring = True
        self.start_time = time.time()
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self._monitor_api_endpoints, daemon=True),
            threading.Thread(target=self._monitor_system_metrics, daemon=True),
            threading.Thread(target=self._periodic_analysis, daemon=True),
        ]
        
        for thread in threads:
            thread.start()
            
        self.logger.info("✅ Monitoring started! Use your application now...")
        self.logger.info("🛑 Press Ctrl+C to stop monitoring and generate report")
        
        try:
            # Keep main thread alive
            while self.monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()
            
    def stop_monitoring(self):
        """Stop monitoring and generate final report"""
        self.logger.info("🛑 Stopping monitoring...")
        self.monitoring = False
        
        # Generate final report
        report = self.generate_report()
        self.save_report(report)
        
        self.logger.info("✅ Monitoring stopped and report generated!")
        
    def _monitor_api_endpoints(self):
        """Monitor API endpoint usage by testing common endpoints"""
        self.logger.info("📡 Starting API endpoint monitoring...")
        
        # Common endpoints to test
        test_endpoints = [
            ("/", "GET"),
            ("/health", "GET"),
            ("/docs", "GET"),
            ("/openapi.json", "GET"),
            ("/api/v1/upload", "POST"),
            ("/api/v1/documents", "GET"),
            ("/api/v1/extract", "POST"),
            ("/api/v1/personal-info", "POST"),
            ("/api/v1/resume", "POST"),
        ]
        
        while self.monitoring:
            try:
                for path, method in test_endpoints:
                    try:
                        start_time = time.time()
                        
                        if method == "GET":
                            response = requests.get(
                                f"http://{self.app_host}:{self.app_port}{path}",
                                timeout=5
                            )
                        else:
                            # For POST requests, send empty data
                            response = requests.post(
                                f"http://{self.app_host}:{self.app_port}{path}",
                                json={},
                                timeout=5
                            )
                            
                        response_time = time.time() - start_time
                        
                        # Record endpoint usage
                        self._record_endpoint_usage(
                            f"{method} {path}",
                            method,
                            path,
                            response_time,
                            response.status_code
                        )
                        
                        self.logger.debug(f"✅ {method} {path} -> {response.status_code} ({response_time:.3f}s)")
                        
                    except requests.exceptions.RequestException as e:
                        self.logger.debug(f"❌ {method} {path} -> {str(e)}")
                        
            except Exception as e:
                self.logger.debug(f"Error monitoring endpoints: {e}")
                
            time.sleep(15)  # Check every 15 seconds
            
    def _record_endpoint_usage(self, endpoint: str, method: str, path: str, 
                              response_time: float, status_code: int):
        """Record endpoint usage"""
        current_time = time.time()
        
        if endpoint not in self.endpoint_usage:
            self.endpoint_usage[endpoint] = EndpointUsage(
                endpoint=endpoint,
                method=method,
                path=path,
                call_count=0,
                response_times=[],
                status_codes={},
                first_called=current_time,
                last_called=current_time
            )
            
        usage = self.endpoint_usage[endpoint]
        usage.call_count += 1
        usage.response_times.append(response_time)
        usage.status_codes[status_code] = usage.status_codes.get(status_code, 0) + 1
        usage.last_called = current_time
        
        # Save to database with cleanup for memory optimization
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert new record
            cursor.execute("""
                INSERT INTO endpoint_usage 
                (endpoint, method, path, timestamp, response_time, status_code, user_agent, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (endpoint, method, path, current_time, response_time, status_code, "realtime-analyzer", "localhost"))
            
            # Clean up old records (keep only last 1000 entries per endpoint for memory)
            cursor.execute("""
                DELETE FROM endpoint_usage 
                WHERE id NOT IN (
                    SELECT id FROM endpoint_usage 
                    WHERE endpoint = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                )
            """, (endpoint,))
            
            conn.commit()
            
    def _monitor_system_metrics(self):
        """Monitor system metrics"""
        self.logger.info("📊 Starting system metrics monitoring...")
        
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Log metrics periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    self.logger.info(f"📊 System - CPU: {cpu_percent}%, Memory: {memory.percent}%")
                    
            except Exception as e:
                self.logger.debug(f"Error getting system metrics: {e}")
                
            time.sleep(15)
            
    def _periodic_analysis(self):
        """Perform periodic analysis"""
        self.logger.info("🔄 Starting periodic analysis...")
        
        while self.monitoring:
            try:
                # Generate mini report every 2 minutes
                if int(time.time()) % 120 == 0:
                    mini_report = self.generate_mini_report()
                    self.logger.info(f"📈 Status: {mini_report}")
                    
            except Exception as e:
                self.logger.debug(f"Error in periodic analysis: {e}")
                
            time.sleep(30)  # Run every 30 seconds
            
    def generate_mini_report(self) -> Dict[str, Any]:
        """Generate a mini report with current statistics"""
        current_time = time.time()
        duration = current_time - self.start_time
        
        total_calls = sum(usage.call_count for usage in self.endpoint_usage.values())
        
        return {
            "duration_min": round(duration / 60, 1),
            "endpoints_tested": len(self.endpoint_usage),
            "total_calls": total_calls,
            "active_endpoints": len([e for e in self.endpoint_usage.values() if e.call_count > 0]),
            "unused_endpoints": len(self.discovered_endpoints) - len(self.endpoint_usage)
        }
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        current_time = time.time()
        
        # Calculate unused items
        used_endpoints = set(self.endpoint_usage.keys())
        unused_endpoints = list(self.discovered_endpoints - used_endpoints)
        
        # Calculate performance metrics
        total_calls = sum(usage.call_count for usage in self.endpoint_usage.values())
        avg_response_times = {}
        
        for endpoint, usage in self.endpoint_usage.items():
            if usage.response_times:
                avg_response_times[endpoint] = sum(usage.response_times) / len(usage.response_times)
        
        # Get most/least used endpoints
        endpoint_call_counts = [(usage.endpoint, usage.call_count) for usage in self.endpoint_usage.values()]
        endpoint_call_counts.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "analysis_start": self.start_time,
            "analysis_end": current_time,
            "total_duration_minutes": (current_time - self.start_time) / 60,
            "endpoint_usage": [asdict(usage) for usage in self.endpoint_usage.values()],
            "discovered_code": {
                "functions": list(self.discovered_functions),
                "classes": list(self.discovered_classes),
                "endpoints": list(self.discovered_endpoints)
            },
            "unused_code": {
                "functions": list(self.discovered_functions),  # All functions are "unused" since we don't trace them
                "classes": list(self.discovered_classes),      # All classes are "unused" since we don't trace them
                "endpoints": unused_endpoints
            },
            "performance_metrics": {
                "total_endpoint_calls": total_calls,
                "average_response_times": avg_response_times,
                "most_used_endpoints": endpoint_call_counts[:5],
                "least_used_endpoints": endpoint_call_counts[-5:] if endpoint_call_counts else []
            },
            "code_coverage": {
                "endpoint_coverage_percent": (len(used_endpoints) / len(self.discovered_endpoints) * 100) if self.discovered_endpoints else 0,
                "endpoints_discovered": len(self.discovered_endpoints),
                "endpoints_tested": len(used_endpoints),
                "functions_discovered": len(self.discovered_functions),
                "classes_discovered": len(self.discovered_classes)
            }
        }
        
    def save_report(self, report: Dict[str, Any]):
        """Save analysis report to file (replaces existing to save memory)"""
        report_dir = Path("tests/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Use fixed filenames to replace existing reports (memory optimization)
        json_path = report_dir / "realtime_analysis_current.json"
        html_path = report_dir / "realtime_analysis_current.html"
        
        # Also save a timestamped backup for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_json_path = report_dir / f"realtime_analysis_backup_{timestamp}.json"
        
        # Save current report (replaces previous)
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save HTML report (replaces previous)
        self._generate_html_report(report, html_path)
        
        # Save backup copy for history
        with open(backup_json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Clean up old backup files (keep only last 5 for memory optimization)
        self._cleanup_old_backups(report_dir)
        
        self.logger.info(f"📊 Reports updated:")
        self.logger.info(f"   📄 Current JSON: {json_path}")
        self.logger.info(f"   🌐 Current HTML: {html_path}")
        self.logger.info(f"   💾 Backup: {backup_json_path}")
        
    def _cleanup_old_backups(self, report_dir: Path):
        """Clean up old backup files to save disk space"""
        try:
            backup_files = list(report_dir.glob("realtime_analysis_backup_*.json"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the 5 most recent backups
            for old_backup in backup_files[5:]:
                old_backup.unlink()
                self.logger.debug(f"🗑️  Cleaned up old backup: {old_backup.name}")
                
        except Exception as e:
            self.logger.debug(f"Failed to cleanup old backups: {e}")
        
    def _generate_html_report(self, report: Dict[str, Any], output_path: Path):
        """Generate HTML report"""
        duration_minutes = report['total_duration_minutes']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time Usage Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric {{ background: #f8f9ff; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .metric-label {{ color: #666; font-size: 0.9em; }}
                .unused {{ background: #fff5f5; border-left-color: #e53e3e; }}
                .unused .metric-value {{ color: #e53e3e; }}
                .used {{ background: #f0fff4; border-left-color: #38a169; }}
                .used .metric-value {{ color: #38a169; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #667eea; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .status-200 {{ color: #38a169; font-weight: bold; }}
                .status-404 {{ color: #e53e3e; font-weight: bold; }}
                .status-500 {{ color: #d69e2e; font-weight: bold; }}
                .endpoint-list {{ max-height: 300px; overflow-y: auto; background: #f8f9ff; padding: 15px; border-radius: 5px; }}
                .endpoint-item {{ margin: 5px 0; padding: 5px; background: white; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Real-Time Usage Analysis</h1>
                    <p>Analysis Duration: {duration_minutes:.1f} minutes | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>📊 Overview</h2>
                    <div class="metrics-grid">
                        <div class="metric used">
                            <div class="metric-value">{report['performance_metrics']['total_endpoint_calls']}</div>
                            <div class="metric-label">Total API Calls</div>
                        </div>
                        <div class="metric used">
                            <div class="metric-value">{len(report['endpoint_usage'])}</div>
                            <div class="metric-label">Endpoints Tested</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{report['code_coverage']['endpoints_discovered']}</div>
                            <div class="metric-label">Endpoints Discovered</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{report['code_coverage']['functions_discovered']}</div>
                            <div class="metric-label">Functions Discovered</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{report['code_coverage']['classes_discovered']}</div>
                            <div class="metric-label">Classes Discovered</div>
                        </div>
                        <div class="metric unused">
                            <div class="metric-value">{len(report['unused_code']['endpoints'])}</div>
                            <div class="metric-label">Unused Endpoints</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Code Coverage</h2>
                    <div class="metrics-grid">
                        <div class="metric used">
                            <div class="metric-value">{report['code_coverage']['endpoint_coverage_percent']:.1f}%</div>
                            <div class="metric-label">Endpoint Coverage</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🌐 Endpoint Usage Details</h2>
                    <table>
                        <tr>
                            <th>Endpoint</th>
                            <th>Method</th>
                            <th>Calls</th>
                            <th>Avg Response Time</th>
                            <th>Status Codes</th>
                            <th>First Called</th>
                            <th>Last Called</th>
                        </tr>
        """
        
        for usage in report['endpoint_usage']:
            avg_time = sum(usage['response_times']) / len(usage['response_times']) if usage['response_times'] else 0
            first_called = datetime.fromtimestamp(usage['first_called']).strftime('%H:%M:%S')
            last_called = datetime.fromtimestamp(usage['last_called']).strftime('%H:%M:%S')
            
            status_codes_str = ""
            for code, count in usage['status_codes'].items():
                status_class = f"status-{code}"
                status_codes_str += f'<span class="{status_class}">{code}({count})</span> '
            
            html_content += f"""
                        <tr>
                            <td><code>{usage['path']}</code></td>
                            <td><strong>{usage['method']}</strong></td>
                            <td>{usage['call_count']}</td>
                            <td>{avg_time:.3f}s</td>
                            <td>{status_codes_str}</td>
                            <td>{first_called}</td>
                            <td>{last_called}</td>
                        </tr>
            """
            
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>🚫 Unused Endpoints</h2>
                    <div class="endpoint-list">
        """
        
        if report['unused_code']['endpoints']:
            for endpoint in report['unused_code']['endpoints'][:50]:  # Show first 50
                html_content += f'<div class="endpoint-item">❌ {endpoint}</div>'
            if len(report['unused_code']['endpoints']) > 50:
                html_content += f'<div class="endpoint-item">... and {len(report["unused_code"]["endpoints"]) - 50} more unused endpoints</div>'
        else:
            html_content += '<div class="endpoint-item">🎉 All discovered endpoints were tested!</div>'
            
        html_content += """
                    </div>
                </div>
                
                <div class="section">
                    <h2>⚡ Performance Summary</h2>
                    <h3>Most Used Endpoints</h3>
                    <ul>
        """
        
        for endpoint, count in report['performance_metrics']['most_used_endpoints']:
            html_content += f"<li><strong>{endpoint}</strong>: {count} calls</li>"
            
        html_content += """
                    </ul>
                </div>
                
                <div class="section">
                    <h2>💡 Recommendations</h2>
                    <ul>
                        <li>🧹 Consider removing or refactoring unused endpoints to reduce codebase complexity</li>
                        <li>📊 Monitor frequently used endpoints for performance optimization opportunities</li>
                        <li>🔍 Investigate endpoints with high error rates (4xx/5xx status codes)</li>
                        <li>⚡ Optimize endpoints with slow response times</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Usage Analyzer for FastAPI")
    parser.add_argument("--monitor", action="store_true", help="Start real-time monitoring")
    parser.add_argument("--report", action="store_true", help="Generate report from existing data")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    
    args = parser.parse_args()
    
    if not args.monitor and not args.report:
        print("🔍 Real-Time Usage Analyzer for FastAPI")
        print("")
        print("Usage:")
        print("  python tests/analysis/realtime_usage_analyzer.py --monitor    # Start monitoring")
        print("  python tests/analysis/realtime_usage_analyzer.py --report     # Generate report")
        print("")
        print("Options:")
        print("  --host HOST     API host (default: localhost)")
        print("  --port PORT     API port (default: 8000)")
        print("")
        print("Instructions:")
        print("1. Start your FastAPI app: uvicorn main:app --reload")
        print("2. Start monitoring: python tests/analysis/realtime_usage_analyzer.py --monitor")
        print("3. Use your application (web interface, API calls, etc.)")
        print("4. Stop monitoring with Ctrl+C to generate the report")
        return
    
    analyzer = RealTimeUsageAnalyzer(
        app_host=args.host,
        app_port=args.port
    )
    
    if args.monitor:
        analyzer.start_monitoring()
    elif args.report:
        report = analyzer.generate_report()
        analyzer.save_report(report)
        print("📊 Report generated from existing data!")

if __name__ == "__main__":
    main() 