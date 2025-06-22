#!/usr/bin/env python3
"""
Integrated Usage Analyzer for FastAPI Application

This service integrates directly with your FastAPI app to provide real-time analysis
without needing to run a separate process. It automatically starts when the app starts
and generates reports when the app shuts down.
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import ast
import psutil
import sqlite3
from fastapi import Request
from loguru import logger

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
    user_agents: Set[str]
    ip_addresses: Set[str]

class IntegratedUsageAnalyzer:
    """Integrated usage analyzer that runs within FastAPI app"""
    
    def __init__(self, enabled: bool = None):
        """
        Initialize the analyzer
        
        Args:
            enabled: Whether to enable analysis. If None, checks ENABLE_USAGE_ANALYSIS env var
        """
        # Check if analysis should be enabled
        if enabled is None:
            enabled = os.getenv("ENABLE_USAGE_ANALYSIS", "true").lower() in ("true", "1", "yes", "on")
            
        self.enabled = enabled
        
        if not self.enabled:
            logger.info("📊 Usage analysis disabled (set ENABLE_USAGE_ANALYSIS=true to enable)")
            return
            
        # Initialize paths
        self.project_root = Path(__file__).parent.parent.parent
        self.db_path = "tests/reports/integrated_analysis.db"
        
        # Analysis data
        self.endpoint_usage: Dict[str, EndpointUsage] = {}
        self.start_time = time.time()
        
        # Code discovery
        self.discovered_functions: Set[str] = set()
        self.discovered_classes: Set[str] = set()
        self.discovered_endpoints: Set[str] = set()
        
        # Monitoring state
        self.monitoring = False
        
        # Initialize components
        self._setup_logging()
        self._init_database()
        self._discover_code_structure()
        
        logger.info("📊 Integrated Usage Analyzer initialized")
        
    def _setup_logging(self):
        """Setup logging specific to analyzer"""
        log_dir = Path("tests/reports/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
    def _init_database(self):
        """Initialize SQLite database"""
        if not self.enabled:
            return
            
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
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
            
            conn.commit()
            
    def _discover_code_structure(self):
        """Discover all functions, classes, and endpoints"""
        if not self.enabled:
            return
            
        logger.info("🔍 Discovering code structure...")
        
        for file_path in self.project_root.rglob("*.py"):
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if 'venv' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                self._analyze_python_file(file_path)
            except Exception as e:
                logger.debug(f"Failed to analyze {file_path}: {e}")
                
        logger.info(f"✅ Discovered {len(self.discovered_functions)} functions, "
                   f"{len(self.discovered_classes)} classes, "
                   f"{len(self.discovered_endpoints)} endpoints")
                   
    def _analyze_python_file(self, file_path: Path):
        """Analyze a Python file for code structure"""
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
                                endpoint = f"{decorator.attr.upper()} /{node.name}"
                                self.discovered_endpoints.add(endpoint)
                        elif isinstance(decorator, ast.Call):
                            if hasattr(decorator.func, 'attr'):
                                if decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                                    if decorator.args and isinstance(decorator.args[0], ast.Constant):
                                        endpoint = f"{decorator.func.attr.upper()} {decorator.args[0].value}"
                                        self.discovered_endpoints.add(endpoint)
                                        
                elif isinstance(node, ast.ClassDef):
                    class_name = f"{file_path.stem}.{node.name}"
                    self.discovered_classes.add(class_name)
                    
        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")
            
    def start_monitoring(self):
        """Start monitoring (called when FastAPI app starts)"""
        if not self.enabled:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        
        # Start background monitoring thread
        monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        monitor_thread.start()
        
        logger.info("📊 Integrated usage monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring and generate report (called when FastAPI app shuts down)"""
        if not self.enabled:
            return
            
        self.monitoring = False
        
        # Generate final report
        report = self._generate_report()
        self._save_report(report)
        
        logger.info("📊 Usage monitoring stopped and report generated")
        
    def record_request(self, request: Request, response_time: float, status_code: int):
        """Record an API request (called from middleware)"""
        if not self.enabled or not self.monitoring:
            return
            
        try:
            endpoint = f"{request.method} {request.url.path}"
            current_time = time.time()
            
            # Get client info
            user_agent = request.headers.get("user-agent", "unknown")
            client_host = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
            
            # Update endpoint usage
            if endpoint not in self.endpoint_usage:
                self.endpoint_usage[endpoint] = EndpointUsage(
                    endpoint=endpoint,
                    method=request.method,
                    path=request.url.path,
                    call_count=0,
                    response_times=[],
                    status_codes={},
                    first_called=current_time,
                    last_called=current_time,
                    user_agents=set(),
                    ip_addresses=set()
                )
                
            usage = self.endpoint_usage[endpoint]
            usage.call_count += 1
            usage.response_times.append(response_time)
            usage.status_codes[status_code] = usage.status_codes.get(status_code, 0) + 1
            usage.last_called = current_time
            usage.user_agents.add(user_agent)
            usage.ip_addresses.add(client_host)
            
            # Save to database periodically
            if usage.call_count % 10 == 0:  # Every 10 requests
                self._save_to_database(endpoint, request.method, request.url.path, 
                                     current_time, response_time, status_code, 
                                     user_agent, client_host)
                                     
        except Exception as e:
            logger.debug(f"Error recording request: {e}")
            
    def _save_to_database(self, endpoint: str, method: str, path: str, 
                         timestamp: float, response_time: float, status_code: int,
                         user_agent: str, ip_address: str):
        """Save request data to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO endpoint_usage 
                    (endpoint, method, path, timestamp, response_time, status_code, user_agent, ip_address)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (endpoint, method, path, timestamp, response_time, status_code, user_agent, ip_address))
                
                # Cleanup old records (keep last 1000 per endpoint)
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
        except Exception as e:
            logger.debug(f"Database save error: {e}")
            
    def _background_monitor(self):
        """Background monitoring thread"""
        while self.monitoring:
            try:
                # Log periodic status
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    total_calls = sum(usage.call_count for usage in self.endpoint_usage.values())
                    logger.info(f"📊 Analysis Status: {len(self.endpoint_usage)} endpoints, {total_calls} total calls")
                    
            except Exception as e:
                logger.debug(f"Background monitor error: {e}")
                
            time.sleep(30)  # Check every 30 seconds
            
    def _generate_report(self) -> Dict[str, Any]:
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
            "endpoint_usage": [self._serialize_endpoint_usage(usage) for usage in self.endpoint_usage.values()],
            "discovered_code": {
                "functions": list(self.discovered_functions),
                "classes": list(self.discovered_classes),
                "endpoints": list(self.discovered_endpoints)
            },
            "unused_code": {
                "functions": list(self.discovered_functions),
                "classes": list(self.discovered_classes),
                "endpoints": unused_endpoints
            },
            "performance_metrics": {
                "total_endpoint_calls": total_calls,
                "average_response_times": avg_response_times,
                "most_used_endpoints": endpoint_call_counts[:10],
                "least_used_endpoints": endpoint_call_counts[-10:] if endpoint_call_counts else []
            },
            "code_coverage": {
                "endpoint_coverage_percent": (len(used_endpoints) / len(self.discovered_endpoints) * 100) if self.discovered_endpoints else 0,
                "endpoints_discovered": len(self.discovered_endpoints),
                "endpoints_tested": len(used_endpoints),
                "functions_discovered": len(self.discovered_functions),
                "classes_discovered": len(self.discovered_classes)
            }
        }
        
    def _serialize_endpoint_usage(self, usage: EndpointUsage) -> Dict[str, Any]:
        """Serialize EndpointUsage for JSON"""
        return {
            "endpoint": usage.endpoint,
            "method": usage.method,
            "path": usage.path,
            "call_count": usage.call_count,
            "response_times": usage.response_times,
            "status_codes": usage.status_codes,
            "first_called": usage.first_called,
            "last_called": usage.last_called,
            "user_agents": list(usage.user_agents),
            "ip_addresses": list(usage.ip_addresses)
        }
        
    def _save_report(self, report: Dict[str, Any]):
        """Save analysis report to files"""
        report_dir = Path("tests/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Use fixed filenames (memory optimization)
        json_path = report_dir / "integrated_analysis_current.json"
        html_path = report_dir / "integrated_analysis_current.html"
        
        # Also save timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_json_path = report_dir / f"integrated_analysis_backup_{timestamp}.json"
        
        # Save current report
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save HTML report
        self._generate_html_report(report, html_path)
        
        # Save backup
        with open(backup_json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Cleanup old backups
        self._cleanup_old_backups(report_dir)
        
        logger.info(f"📊 Analysis reports saved:")
        logger.info(f"   📄 Current JSON: {json_path}")
        logger.info(f"   🌐 Current HTML: {html_path}")
        logger.info(f"   💾 Backup: {backup_json_path}")
        
    def _cleanup_old_backups(self, report_dir: Path):
        """Clean up old backup files - keep only 1 most recent"""
        try:
            backup_files = list(report_dir.glob("integrated_analysis_backup_*.json"))
            
            if len(backup_files) <= 1:
                return  # Keep at least 1 backup
                
            # Sort by filename (which includes timestamp) - newer files have later timestamps
            backup_files.sort(key=lambda x: x.name, reverse=True)
            
            # Keep only the 1 most recent backup, delete the rest
            files_to_delete = backup_files[1:]
            for old_backup in files_to_delete:
                old_backup.unlink()
                logger.info(f"🗑️  Cleaned up old backup: {old_backup.name}")
                
            if files_to_delete:
                logger.info(f"📊 Backup cleanup: Kept {len(backup_files) - len(files_to_delete)} backup, removed {len(files_to_delete)} old files")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")
            
    def _generate_html_report(self, report: Dict[str, Any], output_path: Path):
        """Generate HTML report"""
        duration_minutes = report['total_duration_minutes']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Integrated Usage Analysis Report</title>
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
                .toggle-info {{ background: #e6f7ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Integrated Usage Analysis</h1>
                    <p>Analysis Duration: {duration_minutes:.1f} minutes | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="toggle-info">
                    <h3>💡 Toggle Analysis</h3>
                    <p><strong>Enable:</strong> Set environment variable <code>ENABLE_USAGE_ANALYSIS=true</code></p>
                    <p><strong>Disable:</strong> Set environment variable <code>ENABLE_USAGE_ANALYSIS=false</code></p>
                    <p><strong>Current Status:</strong> {"✅ Enabled" if self.enabled else "❌ Disabled"}</p>
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
                            <div class="metric-label">Endpoints Used</div>
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
                    <h2>🌐 Most Used Endpoints</h2>
                    <table>
                        <tr><th>Endpoint</th><th>Calls</th><th>Avg Response Time</th></tr>
        """
        
        for endpoint, count in report['performance_metrics']['most_used_endpoints'][:10]:
            avg_time = report['performance_metrics']['average_response_times'].get(endpoint, 0)
            html_content += f"""
                        <tr>
                            <td><code>{endpoint}</code></td>
                            <td>{count}</td>
                            <td>{avg_time:.3f}s</td>
                        </tr>
            """
            
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>🚫 Unused Endpoints</h2>
                    <ul>
        """
        
        for endpoint in report['unused_code']['endpoints'][:20]:
            html_content += f"<li>❌ {endpoint}</li>"
            
        html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
            
    def get_status(self) -> Dict[str, Any]:
        """Get current analysis status"""
        if not self.enabled:
            return {"enabled": False, "message": "Analysis disabled"}
            
        total_calls = sum(usage.call_count for usage in self.endpoint_usage.values())
        duration = (time.time() - self.start_time) / 60
        
        return {
            "enabled": True,
            "monitoring": self.monitoring,
            "duration_minutes": duration,
            "endpoints_tracked": len(self.endpoint_usage),
            "total_calls": total_calls,
            "endpoints_discovered": len(self.discovered_endpoints),
            "functions_discovered": len(self.discovered_functions),
            "classes_discovered": len(self.discovered_classes)
        }

# Global analyzer instance
_analyzer = None

def get_analyzer() -> IntegratedUsageAnalyzer:
    """Get the global analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = IntegratedUsageAnalyzer()
    return _analyzer

def start_analysis():
    """Start analysis (call from app startup)"""
    analyzer = get_analyzer()
    analyzer.start_monitoring()

def stop_analysis():
    """Stop analysis and generate report (call from app shutdown)"""
    analyzer = get_analyzer()
    analyzer.stop_monitoring()

def record_request(request, response_time: float, status_code: int):
    """Record a request (call from middleware)"""
    analyzer = get_analyzer()
    analyzer.record_request(request, response_time, status_code) 