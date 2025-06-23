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
import traceback
import inspect
import uuid

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

@dataclass
class FunctionUsage:
    """Represents function usage tracking"""
    function_name: str
    file_name: str
    file_path: str
    call_count: int
    execution_times: List[float]
    input_types: Set[str]
    output_types: Set[str]
    input_examples: List[str]
    output_examples: List[str]
    first_called: float
    last_called: float
    error_count: int
    success_count: int

@dataclass
class ClassUsage:
    """Represents class usage tracking"""
    class_name: str
    file_name: str
    file_path: str
    instantiation_count: int
    methods_called: Dict[str, int]
    first_instantiated: float
    last_instantiated: float
    constructor_args: List[str]
    instance_attributes: Set[str]

@dataclass
class DetailedFunctionCall:
    """Detailed function call tracking"""
    function_name: str
    file_name: str
    file_path: str
    line_number: int
    timestamp: float
    execution_time: float
    input_args: str
    output_result: str
    success: bool
    error_message: str
    call_stack: List[str]
    memory_usage: float
    cpu_usage: float
    thread_id: str
    request_id: str
    endpoint_triggered: str
    parent_function: str
    child_functions: List[str]
    database_queries: int
    api_calls: int
    cache_hits: int
    cache_misses: int

@dataclass
class EndpointExecutionTrace:
    """Complete execution trace for an endpoint"""
    endpoint: str
    request_id: str
    timestamp: float
    total_execution_time: float
    function_calls: List[DetailedFunctionCall]
    execution_flow: List[str]
    call_tree: Dict[str, List[str]]
    performance_bottlenecks: List[Dict[str, Any]]
    resource_usage: Dict[str, float]
    status_code: int
    error_details: Optional[str]

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
        self.function_usage: Dict[str, FunctionUsage] = {}
        self.class_usage: Dict[str, ClassUsage] = {}
        self.start_time = time.time()
        
        # Deep tracking data
        self.detailed_function_calls: List[DetailedFunctionCall] = []
        self.execution_traces: List[EndpointExecutionTrace] = []
        self.active_requests: Dict[str, Dict[str, Any]] = {}  # Track active requests
        self.call_stacks: Dict[str, List[str]] = {}  # Thread-specific call stacks
        
        # Code discovery
        self.discovered_functions: Dict[str, Dict[str, str]] = {}  # {func_name: {file_name, file_path, signature}}
        self.discovered_classes: Dict[str, Dict[str, str]] = {}   # {class_name: {file_name, file_path, methods}}
        self.discovered_endpoints: Set[str] = set()
        
        # Monitoring state
        self.monitoring = False
        
        # Initialize components
        self._setup_logging()
        self._init_database()
        self._discover_code_structure()
        
        logger.info("📊 Integrated Usage Analyzer initialized with deep tracking")
        
    def _setup_logging(self):
        """Setup logging specific to analyzer"""
        log_dir = Path("tests/reports/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
    def _init_database(self):
        """Initialize SQLite database with enhanced schema for detailed tracking"""
        if not self.enabled:
            return
            
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic endpoint usage table
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
            
            # Detailed function calls table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detailed_function_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    function_name TEXT,
                    file_name TEXT,
                    file_path TEXT,
                    line_number INTEGER,
                    timestamp REAL,
                    execution_time REAL,
                    input_args TEXT,
                    output_result TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    call_stack TEXT,
                    memory_usage REAL,
                    cpu_usage REAL,
                    thread_id TEXT,
                    request_id TEXT,
                    endpoint_triggered TEXT,
                    parent_function TEXT,
                    child_functions TEXT,
                    database_queries INTEGER,
                    api_calls INTEGER,
                    cache_hits INTEGER,
                    cache_misses INTEGER
                )
            """)
            
            # Execution traces table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT,
                    request_id TEXT,
                    timestamp REAL,
                    total_execution_time REAL,
                    execution_flow TEXT,
                    call_tree TEXT,
                    performance_bottlenecks TEXT,
                    resource_usage TEXT,
                    status_code INTEGER,
                    error_details TEXT
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp REAL,
                    context TEXT
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
                    
                    # Extract function signature
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            try:
                                arg_str += f": {ast.unparse(arg.annotation)}"
                            except:
                                pass
                        args.append(arg_str)
                    
                    # Extract return type
                    return_type = "Any"
                    if node.returns:
                        try:
                            return_type = ast.unparse(node.returns)
                        except:
                            pass
                    
                    signature = f"({', '.join(args)}) -> {return_type}"
                    
                    self.discovered_functions[func_name] = {
                        'file_name': file_path.stem,
                        'file_path': str(file_path),
                        'signature': signature,
                        'line_number': node.lineno
                    }
                    
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
                    
                    # Extract class methods
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                    
                    # Extract base classes
                    base_classes = []
                    for base in node.bases:
                        try:
                            base_classes.append(ast.unparse(base))
                        except:
                            pass
                    
                    self.discovered_classes[class_name] = {
                        'file_name': file_path.stem,
                        'file_path': str(file_path),
                        'methods': methods,
                        'base_classes': base_classes,
                        'line_number': node.lineno
                    }
                    
        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")
            
    def start_monitoring(self):
        """Start monitoring (called when FastAPI app starts)"""
        if not self.enabled:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        
        # Enable automatic function/class tracking
        try:
            from app.services.auto_tracker import enable_auto_tracking
            enable_auto_tracking()
            logger.info("🔍 Automatic function/class tracking enabled")
        except ImportError:
            logger.debug("Auto tracking not available")
        
        # Start background monitoring thread
        monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        monitor_thread.start()
        
        logger.info("📊 Integrated usage monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring and generate report (called when FastAPI app shuts down)"""
        if not self.enabled:
            return
            
        self.monitoring = False
        
        # Disable automatic tracking
        try:
            from app.services.auto_tracker import disable_auto_tracking
            disable_auto_tracking()
            logger.info("🔍 Automatic function/class tracking disabled")
        except ImportError:
            pass
        
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
            
    def record_function_call(self, func_name: str, file_name: str, file_path: str, 
                           execution_time: float, input_args: str, output_result: str, 
                           success: bool = True):
        """Record a function call for analysis"""
        if not self.enabled or not self.monitoring:
            return
            
        try:
            current_time = time.time()
            full_func_name = f"{file_name}.{func_name}"
            
            if full_func_name not in self.function_usage:
                self.function_usage[full_func_name] = FunctionUsage(
                    function_name=func_name,
                    file_name=file_name,
                    file_path=file_path,
                    call_count=0,
                    execution_times=[],
                    input_types=set(),
                    output_types=set(),
                    input_examples=[],
                    output_examples=[],
                    first_called=current_time,
                    last_called=current_time,
                    error_count=0,
                    success_count=0
                )
            
            usage = self.function_usage[full_func_name]
            usage.call_count += 1
            usage.execution_times.append(execution_time)
            usage.last_called = current_time
            
            # Track input/output examples (keep only last 5)
            if len(usage.input_examples) >= 5:
                usage.input_examples.pop(0)
            if len(usage.output_examples) >= 5:
                usage.output_examples.pop(0)
                
            usage.input_examples.append(input_args[:200])  # Limit length
            usage.output_examples.append(str(output_result)[:200])  # Limit length
            
            # Track types
            usage.input_types.add(type(input_args).__name__)
            usage.output_types.add(type(output_result).__name__)
            
            if success:
                usage.success_count += 1
            else:
                usage.error_count += 1
                
        except Exception as e:
            logger.debug(f"Error recording function call: {e}")
            
    def record_class_instantiation(self, class_name: str, file_name: str, file_path: str,
                                 constructor_args: str):
        """Record a class instantiation for analysis"""
        if not self.enabled or not self.monitoring:
            return
            
        try:
            current_time = time.time()
            full_class_name = f"{file_name}.{class_name}"
            
            if full_class_name not in self.class_usage:
                self.class_usage[full_class_name] = ClassUsage(
                    class_name=class_name,
                    file_name=file_name,
                    file_path=file_path,
                    instantiation_count=0,
                    methods_called={},
                    first_instantiated=current_time,
                    last_instantiated=current_time,
                    constructor_args=[],
                    instance_attributes=set()
                )
            
            usage = self.class_usage[full_class_name]
            usage.instantiation_count += 1
            usage.last_instantiated = current_time
            
            # Track constructor args (keep only last 5)
            if len(usage.constructor_args) >= 5:
                usage.constructor_args.pop(0)
            usage.constructor_args.append(constructor_args[:200])  # Limit length
            
        except Exception as e:
            logger.debug(f"Error recording class instantiation: {e}")
            
    def record_method_call(self, class_name: str, method_name: str, file_name: str):
        """Record a method call on a class instance"""
        if not self.enabled or not self.monitoring:
            return
            
        try:
            full_class_name = f"{file_name}.{class_name}"
            if full_class_name in self.class_usage:
                usage = self.class_usage[full_class_name]
                if method_name not in usage.methods_called:
                    usage.methods_called[method_name] = 0
                usage.methods_called[method_name] += 1
                
        except Exception as e:
            logger.debug(f"Error recording method call: {e}")
            
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
        
        used_functions = set(self.function_usage.keys())
        unused_functions = list(set(self.discovered_functions.keys()) - used_functions)
        
        used_classes = set(self.class_usage.keys())
        unused_classes = list(set(self.discovered_classes.keys()) - used_classes)
        
        # Calculate performance metrics
        total_calls = sum(usage.call_count for usage in self.endpoint_usage.values())
        total_function_calls = sum(usage.call_count for usage in self.function_usage.values())
        total_class_instantiations = sum(usage.instantiation_count for usage in self.class_usage.values())
        
        avg_response_times = {}
        avg_function_times = {}
        
        for endpoint, usage in self.endpoint_usage.items():
            if usage.response_times:
                avg_response_times[endpoint] = sum(usage.response_times) / len(usage.response_times)
                
        for func_name, usage in self.function_usage.items():
            if usage.execution_times:
                avg_function_times[func_name] = sum(usage.execution_times) / len(usage.execution_times)
        
        # Get most/least used items
        endpoint_call_counts = [(usage.endpoint, usage.call_count) for usage in self.endpoint_usage.values()]
        endpoint_call_counts.sort(key=lambda x: x[1], reverse=True)
        
        function_call_counts = [(usage.function_name, usage.call_count) for usage in self.function_usage.values()]
        function_call_counts.sort(key=lambda x: x[1], reverse=True)
        
        class_instantiation_counts = [(usage.class_name, usage.instantiation_count) for usage in self.class_usage.values()]
        class_instantiation_counts.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "analysis_start": self.start_time,
            "analysis_end": current_time,
            "total_duration_minutes": (current_time - self.start_time) / 60,
            "endpoint_usage": [self._serialize_endpoint_usage(usage) for usage in self.endpoint_usage.values()],
            "function_usage": [self._serialize_function_usage(usage) for usage in self.function_usage.values()],
            "class_usage": [self._serialize_class_usage(usage) for usage in self.class_usage.values()],
            "discovered_code": {
                "functions": self.discovered_functions,
                "classes": self.discovered_classes,
                "endpoints": list(self.discovered_endpoints)
            },
            "unused_code": {
                "functions": unused_functions,
                "classes": unused_classes,
                "endpoints": unused_endpoints
            },
            "performance_metrics": {
                "total_endpoint_calls": total_calls,
                "total_function_calls": total_function_calls,
                "total_class_instantiations": total_class_instantiations,
                "average_response_times": avg_response_times,
                "average_function_times": avg_function_times,
                "most_used_endpoints": endpoint_call_counts[:10],
                "most_used_functions": function_call_counts[:10],
                "most_used_classes": class_instantiation_counts[:10],
                "least_used_endpoints": endpoint_call_counts[-10:] if endpoint_call_counts else []
            },
            "code_coverage": {
                "endpoint_coverage_percent": (len(used_endpoints) / len(self.discovered_endpoints) * 100) if self.discovered_endpoints else 0,
                "function_coverage_percent": (len(used_functions) / len(self.discovered_functions) * 100) if self.discovered_functions else 0,
                "class_coverage_percent": (len(used_classes) / len(self.discovered_classes) * 100) if self.discovered_classes else 0,
                "endpoints_discovered": len(self.discovered_endpoints),
                "endpoints_tested": len(used_endpoints),
                "functions_discovered": len(self.discovered_functions),
                "functions_tested": len(used_functions),
                "classes_discovered": len(self.discovered_classes),
                "classes_tested": len(used_classes)
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
        
    def _serialize_function_usage(self, usage: FunctionUsage) -> Dict[str, Any]:
        """Serialize FunctionUsage for JSON"""
        return {
            "function_name": usage.function_name,
            "file_name": usage.file_name,
            "file_path": usage.file_path,
            "call_count": usage.call_count,
            "execution_times": usage.execution_times,
            "avg_execution_time": sum(usage.execution_times) / len(usage.execution_times) if usage.execution_times else 0,
            "input_types": list(usage.input_types),
            "output_types": list(usage.output_types),
            "input_examples": usage.input_examples,
            "output_examples": usage.output_examples,
            "first_called": usage.first_called,
            "last_called": usage.last_called,
            "error_count": usage.error_count,
            "success_count": usage.success_count,
            "success_rate": (usage.success_count / (usage.success_count + usage.error_count) * 100) if (usage.success_count + usage.error_count) > 0 else 0
        }
        
    def _serialize_class_usage(self, usage: ClassUsage) -> Dict[str, Any]:
        """Serialize ClassUsage for JSON"""
        return {
            "class_name": usage.class_name,
            "file_name": usage.file_name,
            "file_path": usage.file_path,
            "instantiation_count": usage.instantiation_count,
            "methods_called": usage.methods_called,
            "total_method_calls": sum(usage.methods_called.values()),
            "most_used_method": max(usage.methods_called.items(), key=lambda x: x[1])[0] if usage.methods_called else None,
            "first_instantiated": usage.first_instantiated,
            "last_instantiated": usage.last_instantiated,
            "constructor_args": usage.constructor_args,
            "instance_attributes": list(usage.instance_attributes)
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
        """Generate enhanced HTML report with deep detailed real-time tracking data"""
        
        # Get actual function usage data
        actual_function_usage = report.get('function_usage', [])
        actual_endpoint_usage = report.get('endpoint_usage', [])
        
        # Get detailed tracking data
        detailed_calls = self.detailed_function_calls[-100:]  # Last 100 detailed calls
        execution_traces = self.execution_traces[-20:]  # Last 20 execution traces
        
        # Calculate real statistics
        total_api_calls = sum(endpoint.get('call_count', 0) for endpoint in actual_endpoint_usage)
        total_function_calls = sum(func.get('call_count', 0) for func in actual_function_usage)
        total_detailed_calls = len(detailed_calls)
        total_traces = len(execution_traces)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Real-Time API Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h3 {{ color: #2c3e50; margin-top: 25px; }}
        
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin-bottom: 25px; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        
        .deep-tracking {{ background: #e3f2fd; border: 2px solid #2196f3; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .deep-title {{ color: #1565c0; font-weight: bold; font-size: 18px; margin-bottom: 15px; }}
        
        .execution-trace {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin: 15px 0; padding: 15px; }}
        .trace-header {{ display: flex; justify-content: between; align-items: center; margin-bottom: 10px; }}
        .trace-endpoint {{ font-weight: bold; color: #1976d2; font-size: 16px; }}
        .trace-time {{ color: #666; font-size: 12px; }}
        .trace-stats {{ display: flex; gap: 20px; margin: 10px 0; }}
        .trace-stat {{ text-align: center; }}
        .trace-stat-value {{ font-weight: bold; color: #1976d2; }}
        .trace-stat-label {{ font-size: 11px; color: #666; }}
        
        .call-stack {{ background: #f8f9fa; border-left: 4px solid #007bff; padding: 10px; margin: 10px 0; font-family: monospace; font-size: 12px; }}
        .call-tree {{ background: #f1f8e9; border: 1px solid #8bc34a; border-radius: 4px; padding: 10px; margin: 10px 0; }}
        .tree-node {{ margin: 5px 0; padding-left: 20px; }}
        .tree-function {{ font-weight: bold; color: #2e7d32; }}
        
        .bottleneck {{ background: #ffebee; border: 1px solid #f44336; border-radius: 4px; padding: 10px; margin: 5px 0; }}
        .bottleneck-slow {{ border-color: #ff9800; background: #fff3e0; }}
        .bottleneck-memory {{ border-color: #9c27b0; background: #f3e5f5; }}
        
        .detailed-call {{ background: #f5f5f5; border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin: 8px 0; }}
        .call-header {{ display: flex; justify-content: space-between; align-items: center; }}
        .call-function {{ font-weight: bold; color: #1976d2; }}
        .call-time {{ background: #4caf50; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px; }}
        .call-details {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px; }}
        .call-detail {{ font-size: 12px; }}
        .call-detail-label {{ font-weight: bold; color: #666; }}
        
        .resource-usage {{ background: #e8f5e8; border: 1px solid #4caf50; border-radius: 4px; padding: 10px; margin: 10px 0; }}
        .resource-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
        .resource-item {{ text-align: center; }}
        .resource-value {{ font-weight: bold; color: #2e7d32; }}
        .resource-label {{ font-size: 11px; color: #666; }}
        
        .no-usage {{ background: #ffebee; border: 1px solid #f44336; border-radius: 8px; padding: 20px; text-align: center; color: #c62828; }}
        .highlight {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        
        .expandable {{ cursor: pointer; }}
        .expandable:hover {{ background-color: #f0f0f0; }}
        .collapsed {{ display: none; }}
        
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; font-size: 12px; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
    </style>
    <script>
        function toggleExpand(id) {{
            const element = document.getElementById(id);
            element.classList.toggle('collapsed');
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>🔍 Deep Real-Time API Analysis Report</h1>
        
        <div class="summary">
            <h2>📊 Deep Tracking Summary</h2>
            <div class="metric">
                <div class="metric-value">{total_api_calls}</div>
                <div class="metric-label">Total API Calls</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_detailed_calls}</div>
                <div class="metric-label">Detailed Function Calls</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_traces}</div>
                <div class="metric-label">Execution Traces</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(self.active_requests)}</div>
                <div class="metric-label">Active Requests</div>
            </div>
        </div>
"""

        if not detailed_calls and not execution_traces:
            html_content += """
        <div class="no-usage">
            <h3>⚠️ No Deep Tracking Data Available</h3>
            <p><strong>To see detailed function tracking:</strong></p>
            <ol style="text-align: left; display: inline-block;">
                <li>Start your FastAPI server</li>
                <li>Make API calls to trigger function execution</li>
                <li>The system will capture call stacks, execution flow, and performance metrics</li>
                <li>Regenerate this report to see deep analysis</li>
            </ol>
        </div>
"""
        else:
            html_content += f"""
        <div class="deep-tracking">
            <div class="deep-title">🔥 Deep Function Tracking Active</div>
            <p>Capturing call stacks, execution flow, memory usage, and performance bottlenecks in real-time.</p>
        </div>
        
        <h2>🎯 Recent Execution Traces</h2>
"""
            
            # Show execution traces
            for i, trace in enumerate(execution_traces[-10:]):  # Show last 10 traces
                html_content += f"""
        <div class="execution-trace">
            <div class="trace-header">
                <div class="trace-endpoint">🎯 {trace.endpoint}</div>
                <div class="trace-time">Request ID: {trace.request_id}</div>
            </div>
            <div class="trace-stats">
                <div class="trace-stat">
                    <div class="trace-stat-value">{trace.total_execution_time:.3f}s</div>
                    <div class="trace-stat-label">Total Time</div>
                </div>
                <div class="trace-stat">
                    <div class="trace-stat-value">{len(trace.function_calls)}</div>
                    <div class="trace-stat-label">Functions Called</div>
                </div>
                <div class="trace-stat">
                    <div class="trace-stat-value">{trace.status_code}</div>
                    <div class="trace-stat-label">Status Code</div>
                </div>
                <div class="trace-stat">
                    <div class="trace-stat-value">{len(trace.performance_bottlenecks)}</div>
                    <div class="trace-stat-label">Bottlenecks</div>
                </div>
            </div>
            
            <div class="expandable" onclick="toggleExpand('trace-{i}')">
                <strong>📋 Click to view execution flow ({len(trace.execution_flow)} functions)</strong>
            </div>
            <div id="trace-{i}" class="collapsed">
                <div class="call-tree">
                    <strong>🌳 Execution Flow:</strong><br>
"""
                for j, func in enumerate(trace.execution_flow):
                    html_content += f"<div class='tree-node'>{j+1}. <span class='tree-function'>{func}</span></div>"
                    
                html_content += """
                </div>
"""
                
                # Show performance bottlenecks
                if trace.performance_bottlenecks:
                    html_content += """
                <strong>⚠️ Performance Bottlenecks:</strong><br>
"""
                    for bottleneck in trace.performance_bottlenecks:
                        bottleneck_type = bottleneck.get('type', 'unknown')
                        css_class = f"bottleneck-{bottleneck_type}" if bottleneck_type in ['slow', 'memory'] else "bottleneck"
                        
                        if bottleneck_type == 'slow_function':
                            html_content += f"""
                <div class="bottleneck {css_class}">
                    🐌 <strong>Slow Function:</strong> {bottleneck['function']} 
                    ({bottleneck['execution_time']:.3f}s in {bottleneck['file']}:{bottleneck['line']})
                </div>
"""
                        elif bottleneck_type == 'high_memory':
                            memory_mb = bottleneck['memory_usage'] / (1024 * 1024)
                            html_content += f"""
                <div class="bottleneck {css_class}">
                    🧠 <strong>High Memory:</strong> {bottleneck['function']} 
                    ({memory_mb:.1f}MB in {bottleneck['file']})
                </div>
"""
                
                # Show resource usage
                html_content += f"""
                <div class="resource-usage">
                    <strong>📊 Resource Usage:</strong>
                    <div class="resource-grid">
                        <div class="resource-item">
                            <div class="resource-value">{trace.resource_usage.get('memory_used', 0) / (1024*1024):.1f}MB</div>
                            <div class="resource-label">Memory Used</div>
                        </div>
                        <div class="resource-item">
                            <div class="resource-value">{trace.resource_usage.get('total_time', 0):.3f}s</div>
                            <div class="resource-label">Total Time</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
"""
            
            # Show detailed function calls
            html_content += f"""
        
        <h2>⚡ Recent Detailed Function Calls</h2>
        <p>Showing last {min(20, len(detailed_calls))} function calls with full context</p>
"""
            
            for call in detailed_calls[-20:]:  # Show last 20 detailed calls
                success_icon = "✅" if call.success else "❌"
                html_content += f"""
        <div class="detailed-call">
            <div class="call-header">
                <div class="call-function">{success_icon} {call.function_name}</div>
                <div class="call-time">{call.execution_time:.3f}s</div>
            </div>
            <div class="call-details">
                <div class="call-detail">
                    <span class="call-detail-label">File:</span> {call.file_name}:{call.line_number}
                </div>
                <div class="call-detail">
                    <span class="call-detail-label">Thread:</span> {call.thread_id}
                </div>
                <div class="call-detail">
                    <span class="call-detail-label">Memory:</span> {call.memory_usage / (1024*1024):.1f}MB
                </div>
                <div class="call-detail">
                    <span class="call-detail-label">CPU:</span> {call.cpu_usage:.1f}%
                </div>
                <div class="call-detail">
                    <span class="call-detail-label">Parent:</span> {call.parent_function.split('/')[-1] if call.parent_function else 'N/A'}
                </div>
                <div class="call-detail">
                    <span class="call-detail-label">Endpoint:</span> {call.endpoint_triggered or 'N/A'}
                </div>
            </div>
"""
                
                if call.input_args:
                    html_content += f"""
            <div class="call-detail">
                <span class="call-detail-label">Input:</span> <code>{call.input_args[:200]}...</code>
            </div>
"""
                
                if call.output_result:
                    html_content += f"""
            <div class="call-detail">
                <span class="call-detail-label">Output:</span> <code>{call.output_result[:200]}...</code>
            </div>
"""
                
                if call.error_message:
                    html_content += f"""
            <div class="call-detail" style="color: #d32f2f;">
                <span class="call-detail-label">Error:</span> {call.error_message}
            </div>
"""
                
                # Show call stack
                if call.call_stack:
                    html_content += f"""
            <div class="expandable" onclick="toggleExpand('stack-{id(call)}')">
                <strong>📚 Call Stack ({len(call.call_stack)} frames)</strong>
            </div>
            <div id="stack-{id(call)}" class="call-stack collapsed">
"""
                    for frame in call.call_stack[-5:]:  # Show last 5 frames
                        html_content += f"<div>{frame}</div>"
                    html_content += """
            </div>
"""
                
                html_content += """
        </div>
"""

        # Add instructions
        html_content += f"""
        
        <div class="highlight">
            <h3>💡 Deep Tracking Features</h3>
            <p><strong>Currently tracking:</strong> {len(detailed_calls)} detailed function calls</p>
            <ul>
                <li>📚 <strong>Call Stacks:</strong> Complete execution context for each function</li>
                <li>🔄 <strong>Execution Flow:</strong> Step-by-step function call sequences</li>
                <li>⚡ <strong>Performance Bottlenecks:</strong> Automatic detection of slow functions</li>
                <li>📊 <strong>Resource Usage:</strong> Memory and CPU tracking per function</li>
                <li>🎯 <strong>Request Tracing:</strong> End-to-end request execution analysis</li>
                <li>🧬 <strong>Call Trees:</strong> Parent-child function relationships</li>
            </ul>
        </div>
        
        <div class="summary" style="margin-top: 30px;">
            <p style="text-align: center; color: #6c757d; font-size: 12px;">
                Deep analysis report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                Tracking {len(self.active_requests)} active requests |
                {len(detailed_calls)} function calls analyzed
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _get_generate_field_answer_functions(self) -> Dict[str, Dict[str, Any]]:
        """Define all functions involved in /api/generate-field-answer endpoint"""
        return {
            # Main endpoint entry
            "main.generate_field_answer": {
                "category": "endpoint",
                "file": "main.py",
                "line": 258,
                "description": "Main FastAPI endpoint handler",
                "dependencies": ["get_session_user", "get_smart_llm_service"]
            },
            
            # Authentication layer
            "main.get_session_user": {
                "category": "authentication",
                "file": "main.py", 
                "line": 67,
                "description": "Session-based user authentication",
                "dependencies": ["get_db", "User.model_validation"]
            },
            "db.session.get_db": {
                "category": "authentication",
                "file": "db/session.py",
                "description": "Database session management",
                "dependencies": []
            },
            
            # Service initialization
            "main.get_smart_llm_service": {
                "category": "service_init",
                "file": "main.py",
                "line": 147,
                "description": "Smart LLM service factory",
                "dependencies": ["SmartLLMService.__init__"]
            },
            "SmartLLMService.__init__": {
                "category": "service_init",
                "file": "app/services/llm_service.py",
                "line": 62,
                "description": "Initialize Smart LLM service",
                "dependencies": ["EmbeddingService.__init__", "RedisVectorStore.__init__", "DocumentService.__init__"]
            },
            "EmbeddingService.__init__": {
                "category": "service_init",
                "file": "app/services/embedding_service.py",
                "description": "Initialize embedding service",
                "dependencies": ["RedisVectorStore.__init__"]
            },
            "RedisVectorStore.__init__": {
                "category": "service_init",
                "file": "app/services/vector_store.py",
                "description": "Initialize Redis vector store",
                "dependencies": []
            },
            "DocumentService.__init__": {
                "category": "service_init",
                "file": "app/services/document_service.py",
                "description": "Initialize document service",
                "dependencies": []
            },
            
            # Tool setup
            "SmartLLMService._setup_tools": {
                "category": "tool_setup",
                "file": "app/services/llm_service.py",
                "line": 99,
                "description": "Setup LangChain tools",
                "dependencies": ["search_resume_data", "search_personal_info"]
            },
            "SmartLLMService._setup_graph": {
                "category": "tool_setup", 
                "file": "app/services/llm_service.py",
                "line": 164,
                "description": "Setup LangGraph workflow",
                "dependencies": ["agent_node", "should_continue", "extract_final_answer"]
            },
            
            # Core processing
            "SmartLLMService.generate_field_answer": {
                "category": "core_processing",
                "file": "app/services/llm_service.py",
                "line": 238,
                "description": "Main LLM processing function",
                "dependencies": ["_search_resume_vectors", "_search_personal_vectors", "_clean_answer"]
            },
            "SmartLLMService._search_resume_vectors": {
                "category": "core_processing",
                "file": "app/services/llm_service.py", 
                "line": 401,
                "description": "Search resume vector database",
                "dependencies": ["EmbeddingService.search_similar_by_document_type"]
            },
            "SmartLLMService._search_personal_vectors": {
                "category": "core_processing",
                "file": "app/services/llm_service.py",
                "line": 419, 
                "description": "Search personal info vector database",
                "dependencies": ["EmbeddingService.search_similar_by_document_type"]
            },
            "EmbeddingService.search_similar_by_document_type": {
                "category": "vector_search",
                "file": "app/services/embedding_service.py",
                "description": "Vector similarity search",
                "dependencies": ["RedisVectorStore.search_similar", "generate_embedding"]
            },
            "RedisVectorStore.search_similar": {
                "category": "vector_search",
                "file": "app/services/vector_store.py",
                "description": "Redis vector search",
                "dependencies": ["_execute_search", "_parse_results"]
            },
            "RedisVectorStore._execute_search": {
                "category": "vector_search",
                "file": "app/services/vector_store.py",
                "description": "Execute Redis FT.SEARCH",
                "dependencies": []
            },
            "RedisVectorStore._parse_results": {
                "category": "vector_search", 
                "file": "app/services/vector_store.py",
                "description": "Parse Redis search results",
                "dependencies": []
            },
            
            # LLM processing
            "ChatOpenAI.invoke": {
                "category": "llm_processing",
                "file": "langchain_openai",
                "description": "OpenAI LLM invocation",
                "dependencies": []
            },
            "SmartLLMService._clean_answer": {
                "category": "post_processing",
                "file": "app/services/llm_service.py",
                "line": 453,
                "description": "Clean and process LLM response",
                "dependencies": ["meta_response_detection", "field_extraction", "text_cleaning"]
            },
            
            # Response generation
            "FieldAnswerResponse": {
                "category": "response",
                "file": "main.py",
                "line": 290,
                "description": "Response model creation",
                "dependencies": []
            },
            
            # Tracking functions
            "track_class_creation": {
                "category": "tracking",
                "file": "app/services/integrated_usage_analyzer.py",
                "description": "Track class instantiation",
                "dependencies": []
            },
            "track_service_call": {
                "category": "tracking", 
                "file": "app/services/integrated_usage_analyzer.py",
                "description": "Track service method calls",
                "dependencies": []
            },
            "track_method_call": {
                "category": "tracking",
                "file": "app/services/integrated_usage_analyzer.py", 
                "description": "Track method invocations",
                "dependencies": []
            },
            
            # Utility functions
            "logger.info": {
                "category": "utility",
                "file": "app/utils/logger.py",
                "description": "Structured logging",
                "dependencies": []
            },
            "logger.error": {
                "category": "utility",
                "file": "app/utils/logger.py", 
                "description": "Error logging",
                "dependencies": []
            }
        }
    
    def _analyze_generate_field_answer_functions(self, functions_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze all functions used by generate-field-answer endpoint"""
        functions_called = []
        execution_flow = []
        performance_breakdown = {}
        dependencies = {}
        
        # Build function call hierarchy
        for func_name, func_info in functions_data.items():
            functions_called.append({
                "name": func_name,
                "category": func_info["category"],
                "file": func_info["file"],
                "description": func_info["description"],
                "line": func_info.get("line", "N/A"),
                "dependencies": func_info.get("dependencies", [])
            })
            
            execution_flow.append(func_name)
            performance_breakdown[func_info["category"]] = performance_breakdown.get(func_info["category"], 0) + 1
            dependencies[func_name] = func_info.get("dependencies", [])
        
        # Categorize functions by their role
        function_categories = {}
        for func in functions_called:
            category = func["category"]
            if category not in function_categories:
                function_categories[category] = []
            function_categories[category].append(func)
        
        # Create execution flow diagram data
        execution_flow_steps = []
        for i, func_name in enumerate(execution_flow):
            func_info = next((f for f in functions_called if f["name"] == func_name), None)
            if func_info:
                execution_flow_steps.append({
                    "step": i + 1,
                    "function": func_name,
                    "category": func_info["category"],
                    "file": func_info["file"],
                    "description": func_info["description"],
                    "dependencies": func_info["dependencies"]
                })
        
        # Performance breakdown by category
        category_stats = {}
        total_functions = len(functions_called)
        for category, count in performance_breakdown.items():
            category_stats[category] = {
                "function_count": count,
                "percentage": (count / total_functions) * 100 if total_functions > 0 else 0
            }
        
        return {
            "endpoint": "POST /api/generate-field-answer",
            "total_functions_involved": total_functions,
            "function_categories": function_categories,
            "execution_flow": execution_flow_steps,
            "performance_breakdown": category_stats,
            "dependency_graph": dependencies,
            "complexity_analysis": {
                "total_layers": len(function_categories),
                "max_dependency_depth": self._calculate_max_dependency_depth(dependencies),
                "most_complex_category": max(category_stats.keys(), key=lambda k: category_stats[k]["function_count"]) if category_stats else None
            }
        }
    
    def _calculate_max_dependency_depth(self, dependencies: Dict[str, List[str]]) -> int:
        """Calculate the maximum dependency depth in the function call graph"""
        def get_depth(func_name, visited=None):
            if visited is None:
                visited = set()
            if func_name in visited:
                return 0  # Circular dependency
            visited.add(func_name)
            
            deps = dependencies.get(func_name, [])
            if not deps:
                return 1
            
            max_child_depth = max((get_depth(dep, visited.copy()) for dep in deps), default=0)
            return max_child_depth + 1
        
        if not dependencies:
            return 0
        
        return max(get_depth(func) for func in dependencies.keys())

    def start_request_trace(self, request_id: str, endpoint: str) -> None:
        """Start tracking a new request"""
        if not self.enabled or not self.monitoring:
            return
            
        self.active_requests[request_id] = {
            "endpoint": endpoint,
            "start_time": time.time(),
            "function_calls": [],
            "call_stack": [],
            "resource_usage": {"memory_start": psutil.Process().memory_info().rss}
        }
        
    def end_request_trace(self, request_id: str, status_code: int, error_details: str = None) -> None:
        """End tracking for a request and save the trace"""
        if not self.enabled or not self.monitoring or request_id not in self.active_requests:
            return
            
        request_data = self.active_requests[request_id]
        end_time = time.time()
        total_time = end_time - request_data["start_time"]
        
        # Calculate resource usage
        memory_end = psutil.Process().memory_info().rss
        memory_used = memory_end - request_data["resource_usage"]["memory_start"]
        
        # Create execution trace
        trace = EndpointExecutionTrace(
            endpoint=request_data["endpoint"],
            request_id=request_id,
            timestamp=request_data["start_time"],
            total_execution_time=total_time,
            function_calls=request_data["function_calls"],
            execution_flow=[call.function_name for call in request_data["function_calls"]],
            call_tree=self._build_call_tree(request_data["function_calls"]),
            performance_bottlenecks=self._identify_bottlenecks(request_data["function_calls"]),
            resource_usage={"memory_used": memory_used, "total_time": total_time},
            status_code=status_code,
            error_details=error_details
        )
        
        self.execution_traces.append(trace)
        self._save_execution_trace_to_db(trace)
        
        # Clean up
        del self.active_requests[request_id]
        
    def record_detailed_function_call(self, func_name: str, file_name: str, file_path: str, 
                                    line_number: int, execution_time: float, input_args: str, 
                                    output_result: str, success: bool = True, error_message: str = "",
                                    request_id: str = "", endpoint: str = "") -> None:
        """Record a detailed function call with full context"""
        if not self.enabled or not self.monitoring:
            return
            
        try:
            import threading
            import traceback
            
            current_time = time.time()
            thread_id = str(threading.current_thread().ident)
            
            # Get call stack
            call_stack = [frame.filename + ":" + str(frame.lineno) + " " + frame.name 
                         for frame in traceback.extract_stack()[:-1]]
            
            # Get resource usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss
            cpu_usage = process.cpu_percent()
            
            # Determine parent function from call stack
            parent_function = call_stack[-2] if len(call_stack) > 1 else ""
            
            # Create detailed function call record
            detailed_call = DetailedFunctionCall(
                function_name=func_name,
                file_name=file_name,
                file_path=file_path,
                line_number=line_number,
                timestamp=current_time,
                execution_time=execution_time,
                input_args=input_args[:1000],  # Limit size
                output_result=str(output_result)[:1000],  # Limit size
                success=success,
                error_message=error_message,
                call_stack=call_stack,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                thread_id=thread_id,
                request_id=request_id,
                endpoint_triggered=endpoint,
                parent_function=parent_function,
                child_functions=[],  # Will be populated later
                database_queries=0,  # TODO: Track DB queries
                api_calls=0,  # TODO: Track API calls
                cache_hits=0,  # TODO: Track cache hits
                cache_misses=0   # TODO: Track cache misses
            )
            
            self.detailed_function_calls.append(detailed_call)
            
            # Add to active request if exists
            if request_id in self.active_requests:
                self.active_requests[request_id]["function_calls"].append(detailed_call)
                
            # Save to database periodically
            if len(self.detailed_function_calls) % 10 == 0:
                self._save_detailed_calls_to_db()
                
        except Exception as e:
            logger.debug(f"Error recording detailed function call: {e}")
            
    def _build_call_tree(self, function_calls: List[DetailedFunctionCall]) -> Dict[str, List[str]]:
        """Build a call tree from function calls"""
        call_tree = {}
        for call in function_calls:
            parent = call.parent_function
            if parent:
                if parent not in call_tree:
                    call_tree[parent] = []
                call_tree[parent].append(call.function_name)
        return call_tree
        
    def _identify_bottlenecks(self, function_calls: List[DetailedFunctionCall]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in function calls"""
        bottlenecks = []
        
        # Sort by execution time
        sorted_calls = sorted(function_calls, key=lambda x: x.execution_time, reverse=True)
        
        # Top 5 slowest functions
        for call in sorted_calls[:5]:
            if call.execution_time > 0.1:  # Functions taking more than 100ms
                bottlenecks.append({
                    "type": "slow_function",
                    "function": call.function_name,
                    "execution_time": call.execution_time,
                    "file": call.file_name,
                    "line": call.line_number
                })
                
        # High memory usage
        for call in function_calls:
            if call.memory_usage > 100 * 1024 * 1024:  # More than 100MB
                bottlenecks.append({
                    "type": "high_memory",
                    "function": call.function_name,
                    "memory_usage": call.memory_usage,
                    "file": call.file_name
                })
                
        return bottlenecks
        
    def _save_detailed_calls_to_db(self):
        """Save detailed function calls to database"""
        if not self.detailed_function_calls:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for call in self.detailed_function_calls[-10:]:  # Save last 10 calls
                    cursor.execute("""
                        INSERT INTO detailed_function_calls 
                        (function_name, file_name, file_path, line_number, timestamp, execution_time,
                         input_args, output_result, success, error_message, call_stack, memory_usage,
                         cpu_usage, thread_id, request_id, endpoint_triggered, parent_function,
                         child_functions, database_queries, api_calls, cache_hits, cache_misses)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        call.function_name, call.file_name, call.file_path, call.line_number,
                        call.timestamp, call.execution_time, call.input_args, call.output_result,
                        call.success, call.error_message, json.dumps(call.call_stack),
                        call.memory_usage, call.cpu_usage, call.thread_id, call.request_id,
                        call.endpoint_triggered, call.parent_function, json.dumps(call.child_functions),
                        call.database_queries, call.api_calls, call.cache_hits, call.cache_misses
                    ))
                    
                conn.commit()
                
        except Exception as e:
            logger.debug(f"Error saving detailed calls to DB: {e}")
            
    def _save_execution_trace_to_db(self, trace: EndpointExecutionTrace):
        """Save execution trace to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO execution_traces 
                    (endpoint, request_id, timestamp, total_execution_time, execution_flow,
                     call_tree, performance_bottlenecks, resource_usage, status_code, error_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.endpoint, trace.request_id, trace.timestamp, trace.total_execution_time,
                    json.dumps(trace.execution_flow), json.dumps(trace.call_tree),
                    json.dumps(trace.performance_bottlenecks, default=str), 
                    json.dumps(trace.resource_usage), trace.status_code, trace.error_details
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.debug(f"Error saving execution trace to DB: {e}")

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

def record_function_call(func_name: str, file_name: str, file_path: str, 
                        execution_time: float, input_args: str, output_result: str, 
                        success: bool = True):
    """Record a function call (call from decorators)"""
    analyzer = get_analyzer()
    analyzer.record_function_call(func_name, file_name, file_path, execution_time, 
                                 input_args, output_result, success)

def record_class_instantiation(class_name: str, file_name: str, file_path: str,
                              constructor_args: str):
    """Record a class instantiation (call from class decorators)"""
    analyzer = get_analyzer()
    analyzer.record_class_instantiation(class_name, file_name, file_path, constructor_args)

def record_method_call(class_name: str, method_name: str, file_name: str):
    """Record a method call (call from method decorators)"""
    analyzer = get_analyzer()
    analyzer.record_method_call(class_name, method_name, file_name)

# Decorator for automatic deep function tracking
def deep_track_function(func):
    """Decorator to automatically track function calls with deep analysis"""
    import asyncio
    import functools
    
    if asyncio.iscoroutinefunction(func):
        # Handle async functions
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            analyzer = get_analyzer()
            if not analyzer or not analyzer.enabled or not analyzer.monitoring:
                return await func(*args, **kwargs)
                
            # Get function metadata
            func_name = func.__name__
            file_path = inspect.getfile(func)
            file_name = os.path.basename(file_path)
            line_number = inspect.getsourcelines(func)[1]
            
            # Prepare input args (limit size to avoid memory issues)
            input_args = str(args)[:500] + ", " + str(kwargs)[:500]
            
            start_time = time.time()
            success = True
            error_message = ""
            result = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                output_result = str(result)[:500] if result is not None else ""
                
                # Record the detailed function call
                analyzer.record_detailed_function_call(
                    func_name=func_name,
                    file_name=file_name,
                    file_path=file_path,
                    line_number=line_number,
                    execution_time=execution_time,
                    input_args=input_args,
                    output_result=output_result,
                    success=success,
                    error_message=error_message,
                    request_id=getattr(threading.current_thread(), 'request_id', ''),
                    endpoint=getattr(threading.current_thread(), 'endpoint', '')
                )
                
        return async_wrapper
    else:
        # Handle sync functions
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            analyzer = get_analyzer()
            if not analyzer or not analyzer.enabled or not analyzer.monitoring:
                return func(*args, **kwargs)
                
            # Get function metadata
            func_name = func.__name__
            file_path = inspect.getfile(func)
            file_name = os.path.basename(file_path)
            line_number = inspect.getsourcelines(func)[1]
            
            # Prepare input args (limit size to avoid memory issues)
            input_args = str(args)[:500] + ", " + str(kwargs)[:500]
            
            start_time = time.time()
            success = True
            error_message = ""
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                output_result = str(result)[:500] if result is not None else ""
                
                # Record the detailed function call
                analyzer.record_detailed_function_call(
                    func_name=func_name,
                    file_name=file_name,
                    file_path=file_path,
                    line_number=line_number,
                    execution_time=execution_time,
                    input_args=input_args,
                    output_result=output_result,
                    success=success,
                    error_message=error_message,
                    request_id=getattr(threading.current_thread(), 'request_id', ''),
                    endpoint=getattr(threading.current_thread(), 'endpoint', '')
                )
                
        return sync_wrapper

# Middleware for request tracing (FastAPI compatible)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class DeepTrackingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware to automatically track requests and their function calls"""
    
    async def dispatch(self, request: Request, call_next):
        print(f"🔍 DeepTrackingMiddleware called for: {request.url.path}")
        
        analyzer = get_analyzer()
        if not analyzer or not analyzer.enabled:
            print(f"⚠️  Analyzer not enabled: analyzer={analyzer is not None}, enabled={analyzer.enabled if analyzer else False}")
            return await call_next(request)
        
        # Generate request ID
        import uuid
        request_id = str(uuid.uuid4())
        endpoint = str(request.url.path)
        
        print(f"🎯 Starting request trace: {request_id} -> {endpoint}")
        
        # Set thread-local data
        threading.current_thread().request_id = request_id
        threading.current_thread().endpoint = endpoint
        
        # Start request trace
        analyzer.start_request_trace(request_id, endpoint)
        
        status_code = 200
        error_details = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            error_details = str(e)
            status_code = 500
            raise
        finally:
            # End request trace
            analyzer.end_request_trace(request_id, status_code, error_details)
            
            # Clean up thread-local data
            if hasattr(threading.current_thread(), 'request_id'):
                delattr(threading.current_thread(), 'request_id')
            if hasattr(threading.current_thread(), 'endpoint'):
                delattr(threading.current_thread(), 'endpoint') 