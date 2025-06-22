#!/usr/bin/env python3
"""
🧪 PYTEST-BASED COMPREHENSIVE TEST RUNNER
CI-ready test execution with coverage reporting and organized structure
"""

import subprocess
import sys
import time
import json
import os
from datetime import datetime
from pathlib import Path
import argparse

class PytestTestRunner:
    """Pytest-based test runner with CI integration and coverage reporting"""
    
    def __init__(self):
        self.start_time = time.time()
        self.base_dir = Path(__file__).parent.parent  # Go up to tests/ directory
        self.root_dir = self.base_dir.parent  # Project root
        self.reports_dir = self.base_dir / "reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "📝", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
        icon = icons.get(level, "📝")
        print(f"[{timestamp}] {icon} {message}")
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        required_packages = [
            "pytest",
            "pytest-cov", 
            "pytest-html",
            "pytest-asyncio",
            "requests"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.log("Installing missing test dependencies...", "WARNING")
            for package in missing_packages:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                 check=True, capture_output=True)
                    self.log(f"Installed {package}", "SUCCESS")
                except subprocess.CalledProcessError as e:
                    self.log(f"Failed to install {package}: {e}", "ERROR")
                    return False
        
        return True
    
    def check_server_health(self) -> bool:
        """Check if the server is running"""
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_pytest_command(self, args: list, description: str) -> dict:
        """Run a pytest command and capture results"""
        self.log(f"Running {description}...")
        
        try:
            start_time = time.time()
            
            # Run pytest with the given arguments
            result = subprocess.run(
                [sys.executable, "-m", "pytest"] + args,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=str(self.root_dir)  # Run from project root
            )
            
            execution_time = time.time() - start_time
            
            return {
                "status": "completed",
                "description": description,
                "return_code": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "command": " ".join(["pytest"] + args)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "description": description,
                "error": "Test timed out after 10 minutes",
                "command": " ".join(["pytest"] + args)
            }
        except Exception as e:
            return {
                "status": "error",
                "description": description,
                "error": str(e),
                "command": " ".join(["pytest"] + args)
            }
    
    def run_coverage_report(self):
        """Generate comprehensive coverage report"""
        self.log("🔍 Generating coverage report...")
        
        coverage_args = [
            "--cov=app",
            "--cov=main",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=json",
            f"--cov-report=html:{self.reports_dir}/coverage_html",
            f"--cov-report=json:{self.reports_dir}/coverage.json",
            "tests/",
            "--tb=short"
        ]
        
        result = self.run_pytest_command(coverage_args, "Coverage Analysis")
        
        if result["success"]:
            self.log("✅ Coverage report generated successfully", "SUCCESS")
            self.log(f"📊 HTML report: {self.reports_dir}/coverage_html/index.html")
            self.log(f"📄 JSON report: {self.reports_dir}/coverage.json")
        else:
            self.log("❌ Coverage report generation failed", "ERROR")
        
        return result
    
    def run_tests_by_category(self, category: str = None, coverage: bool = False):
        """Run tests by category with optional coverage"""
        if category:
            self.log(f"🎯 Running {category.upper()} tests...")
            
            base_args = ["-m", category, "-v"]
            if coverage:
                base_args.extend([
                    "--cov=app",
                    "--cov=main", 
                    f"--cov-report=term",
                    f"--cov-report=html:{self.reports_dir}/coverage_{category}_html"
                ])
            
            base_args.append("tests/")
            
            return self.run_pytest_command(base_args, f"{category.title()} Tests")
        else:
            self.log("🚀 Running ALL tests...")
            
            base_args = ["-v", "--tb=short"]
            if coverage:
                base_args.extend([
                    "--cov=app",
                    "--cov=main",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                    f"--cov-report=html:{self.reports_dir}/coverage_html"
                ])
            
            base_args.append("tests/")
            
            return self.run_pytest_command(base_args, "All Tests")
    
    def run_quick_tests(self):
        """Run quick tests (exclude slow tests)"""
        self.log("⚡ Running quick tests (excluding slow tests)...")
        
        args = [
            "-m", "not slow",
            "-v",
            "--tb=short",
            "tests/"
        ]
        
        return self.run_pytest_command(args, "Quick Tests")
    
    def run_ci_tests(self):
        """Run tests in CI mode with full reporting"""
        self.log("🏗️ Running CI tests with full reporting...")
        
        args = [
            "-v",
            "--tb=short",
            "--maxfail=5",
            f"--junit-xml={self.reports_dir}/junit.xml",
            f"--html={self.reports_dir}/pytest_report.html",
            "--self-contained-html",
            "--cov=app",
            "--cov=main",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=json",
            f"--cov-report=html:{self.reports_dir}/coverage_html",
            f"--cov-report=json:{self.reports_dir}/coverage.json",
            "tests/"
        ]
        
        return self.run_pytest_command(args, "CI Tests")
    
    def generate_consolidated_report(self, test_results: list):
        """Generate consolidated test report"""
        total_time = time.time() - self.start_time
        
        print("=" * 80)
        print("📊 CONSOLIDATED PYTEST TEST RESULTS")
        print("=" * 80)
        
        # Calculate statistics
        total_runs = len(test_results)
        successful_runs = len([r for r in test_results if r.get("success", False)])
        failed_runs = total_runs - successful_runs
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Test Runs: {total_runs}")
        print(f"   Successful: {successful_runs} ✅")
        print(f"   Failed: {failed_runs} ❌")
        print(f"   Success Rate: {(successful_runs/total_runs*100):.1f}%" if total_runs > 0 else "   Success Rate: N/A")
        print(f"   Total Execution Time: {total_time:.1f}s")
        
        # Show detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in test_results:
            status_icon = "✅" if result.get("success") else "❌"
            time_info = f"({result.get('execution_time', 0):.1f}s)"
            
            print(f"   {status_icon} {result['description']} {time_info}")
            
            if not result.get("success") and result.get("stderr"):
                # Show first few lines of error
                error_lines = result["stderr"].split('\n')[:3]
                for line in error_lines:
                    if line.strip():
                        print(f"      └─ {line.strip()}")
        
        # Show available reports
        print(f"\n📊 GENERATED REPORTS:")
        report_files = [
            ("HTML Report", "pytest_report.html"),
            ("Coverage HTML", "coverage_html/index.html"),
            ("Coverage JSON", "coverage.json"),
            ("JUnit XML", "junit.xml"),
            ("Pytest Log", "pytest.log")
        ]
        
        for name, filename in report_files:
            report_path = self.reports_dir / filename
            if report_path.exists() or (report_path.is_dir() and any(report_path.iterdir())):
                print(f"   📄 {name}: {report_path}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if successful_runs == total_runs:
            print("   🏆 EXCELLENT - All test runs successful!")
        elif successful_runs / total_runs >= 0.8:
            print("   ✅ GOOD - Most tests passing")
        else:
            print("   ⚠️ NEEDS ATTENTION - Multiple test failures")
        
        # Save consolidated report
        self.save_consolidated_report(test_results, total_time)
        
        print("=" * 80)
    
    def save_consolidated_report(self, test_results: list, total_time: float):
        """Save detailed test report to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"consolidated_pytest_report_{timestamp}.json"
        
        # Calculate statistics
        total_runs = len(test_results)
        successful_runs = len([r for r in test_results if r.get("success", False)])
        
        detailed_report = {
            "test_run_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time": total_time,
                "total_test_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": total_runs - successful_runs,
                "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0
            },
            "test_results": test_results,
            "system_info": {
                "python_version": sys.version,
                "pytest_version": "pytest-based",
                "test_runner_version": "3.0.0-pytest",
                "test_structure": "pytest_organized"
            },
            "available_reports": {
                "html_report": str(self.reports_dir / "pytest_report.html"),
                "coverage_html": str(self.reports_dir / "coverage_html" / "index.html"),
                "coverage_json": str(self.reports_dir / "coverage.json"),
                "junit_xml": str(self.reports_dir / "junit.xml")
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        self.log(f"💾 Consolidated report saved to: {report_file}")

def main():
    """Main runner with argument parsing"""
    parser = argparse.ArgumentParser(description="Pytest-based comprehensive test runner")
    parser.add_argument("--category", "-m", choices=["unit", "integration", "e2e", "performance", "analysis"], 
                       help="Run tests for specific category")
    parser.add_argument("--coverage", "-c", action="store_true", 
                       help="Generate coverage report")
    parser.add_argument("--quick", "-q", action="store_true", 
                       help="Run quick tests only (exclude slow tests)")
    parser.add_argument("--ci", action="store_true", 
                       help="Run in CI mode with full reporting")
    parser.add_argument("--no-server-check", action="store_true",
                       help="Skip server health check")
    
    args = parser.parse_args()
    
    runner = PytestTestRunner()
    
    runner.log("🚀 Starting Pytest-Based Test Runner")
    print("=" * 80)
    
    # Check dependencies
    if not runner.check_dependencies():
        runner.log("❌ Failed to install required dependencies", "ERROR")
        return 1
    
    # Check server status (unless skipped)
    if not args.no_server_check:
        if runner.check_server_health():
            runner.log("✅ Server is running - proceeding with tests")
        else:
            runner.log("❌ Server is not running", "ERROR")
            runner.log("💡 Please start the server first: python main.py", "WARNING")
            runner.log("💡 Or use --no-server-check to skip this check", "WARNING")
            return 1
    
    test_results = []
    
    # Run tests based on arguments
    if args.ci:
        result = runner.run_ci_tests()
        test_results.append(result)
    elif args.quick:
        result = runner.run_quick_tests()
        test_results.append(result)
    elif args.category:
        result = runner.run_tests_by_category(args.category, args.coverage)
        test_results.append(result)
    elif args.coverage:
        result = runner.run_coverage_report()
        test_results.append(result)
    else:
        # Run all tests with coverage
        result = runner.run_tests_by_category(None, True)
        test_results.append(result)
    
    # Generate consolidated report
    runner.generate_consolidated_report(test_results)
    
    # Return appropriate exit code
    failed_runs = len([r for r in test_results if not r.get("success", False)])
    return 1 if failed_runs > 0 else 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 