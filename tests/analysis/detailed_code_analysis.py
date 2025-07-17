#!/usr/bin/env python3
"""
Detailed Code Analysis for main.py
Identifies all functions, endpoints, classes, and their usage patterns
"""

import ast
import re
from typing import Dict, List, Set, Any
from collections import defaultdict

class DetailedCodeAnalyzer:
    """Comprehensive code analyzer for main.py"""
    
    def __init__(self, file_path: str = "main.py"):
        self.file_path = file_path
        self.content = ""
        
        # Analysis results
        self.functions = {}  # function_name -> {line_number, decorators, calls}
        self.classes = {}    # class_name -> {line_number, methods}
        self.endpoints = {}  # endpoint -> {method, function_name, line_number}
        self.imports = {}    # module -> [imported_items]
        self.globals = set() # global variables
        
        # Usage tracking
        self.function_calls = defaultdict(list)  # function -> [call_locations]
        self.class_usage = defaultdict(list)     # class -> [usage_locations]
        
    def analyze(self):
        """Main analysis method"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            
            # Parse AST
            tree = ast.parse(self.content)
            
            # Analyze different aspects
            self._analyze_imports(tree)
            self._analyze_functions(tree)
            self._analyze_classes(tree)
            self._analyze_endpoints()
            self._analyze_usage_patterns(tree)
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing {self.file_path}: {e}")
            return False
    
    def _analyze_imports(self, tree):
        """Analyze import statements"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports[alias.name] = []
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_items = [alias.name for alias in node.names]
                    self.imports[node.module] = imported_items
    
    def _analyze_functions(self, tree):
        """Analyze function definitions"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                decorators = []
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            decorators.append(f"{decorator.func.attr}")
                        elif isinstance(decorator, ast.Name):
                            decorators.append(decorator.func.id)
                
                self.functions[node.name] = {
                    'line_number': node.lineno,
                    'decorators': decorators,
                    'args': [arg.arg for arg in node.args.args],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'docstring': ast.get_docstring(node)
                }
    
    def _analyze_classes(self, tree):
        """Analyze class definitions"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                
                self.classes[node.name] = {
                    'line_number': node.lineno,
                    'methods': methods,
                    'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    'docstring': ast.get_docstring(node)
                }
    
    def _analyze_endpoints(self):
        """Analyze FastAPI endpoints using regex"""
        lines = self.content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Look for FastAPI decorators
            endpoint_match = re.search(r'@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']', line)
            if endpoint_match:
                method = endpoint_match.group(1).upper()
                path = endpoint_match.group(2)
                
                # Find the function name (usually on next line)
                func_name = None
                for j in range(i, min(i + 5, len(lines))):  # Look ahead up to 5 lines
                    func_match = re.search(r'(?:async\s+)?def\s+(\w+)', lines[j])
                    if func_match:
                        func_name = func_match.group(1)
                        break
                
                self.endpoints[f"{method} {path}"] = {
                    'method': method,
                    'path': path,
                    'function_name': func_name,
                    'line_number': i
                }
    
    def _analyze_usage_patterns(self, tree):
        """Analyze function and class usage patterns"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Function calls
                if isinstance(node.func, ast.Name):
                    self.function_calls[node.func.id].append(getattr(node, 'lineno', 'unknown'))
                elif isinstance(node.func, ast.Attribute):
                    self.function_calls[node.func.attr].append(getattr(node, 'lineno', 'unknown'))
            
            elif isinstance(node, ast.Name):
                # Class usage (instantiation, type hints, etc.)
                if node.id in self.classes:
                    self.class_usage[node.id].append(getattr(node, 'lineno', 'unknown'))
    
    def get_unused_functions(self) -> List[str]:
        """Get functions that are defined but never called"""
        unused = []
        
        # Exclude special functions and endpoints
        special_patterns = [
            '__', 'main', 'lifespan', 'root', 'health_check',
            'get_', 'create_', 'delete_', 'update_', 'upload_',
            'download_', 'validate_', 'test_', 'simple_', 'demo_'
        ]
        
        for func_name in self.functions:
            # Skip if it's called somewhere
            if func_name in self.function_calls and self.function_calls[func_name]:
                continue
            
            # Skip if it's an endpoint function
            is_endpoint = any(endpoint_info['function_name'] == func_name 
                            for endpoint_info in self.endpoints.values())
            if is_endpoint:
                continue
            
            # Skip special functions
            if any(pattern in func_name for pattern in special_patterns):
                continue
            
            unused.append(func_name)
        
        return unused
    
    def get_unused_classes(self) -> List[str]:
        """Get classes that are defined but never used"""
        unused = []
        
        for class_name in self.classes:
            # Skip if it's used somewhere
            if class_name in self.class_usage and self.class_usage[class_name]:
                continue
            
            # Skip BaseModel subclasses (they're used implicitly)
            if 'BaseModel' in self.classes[class_name]['bases']:
                continue
            
            unused.append(class_name)
        
        return unused
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        unused_functions = self.get_unused_functions()
        unused_classes = self.get_unused_classes()
        
        # Categorize functions
        endpoint_functions = set()
        service_functions = set()
        utility_functions = set()
        
        for func_name, func_info in self.functions.items():
            is_endpoint = any(endpoint_info['function_name'] == func_name 
                            for endpoint_info in self.endpoints.values())
            
            if is_endpoint:
                endpoint_functions.add(func_name)
            elif func_name.startswith('get_') and 'lru_cache' in func_info.get('decorators', []):
                service_functions.add(func_name)
            else:
                utility_functions.add(func_name)
        
        return {
            "summary": {
                "total_functions": len(self.functions),
                "total_classes": len(self.classes),
                "total_endpoints": len(self.endpoints),
                "total_imports": len(self.imports),
                "unused_functions": len(unused_functions),
                "unused_classes": len(unused_classes)
            },
            "functions": {
                "all": list(self.functions.keys()),
                "endpoint_functions": list(endpoint_functions),
                "service_functions": list(service_functions),
                "utility_functions": list(utility_functions),
                "unused": unused_functions,
                "details": self.functions
            },
            "classes": {
                "all": list(self.classes.keys()),
                "unused": unused_classes,
                "details": self.classes
            },
            "endpoints": {
                "all": list(self.endpoints.keys()),
                "details": self.endpoints
            },
            "imports": self.imports,
            "usage_analysis": {
                "function_calls": dict(self.function_calls),
                "class_usage": dict(self.class_usage)
            }
        }
    
    def print_detailed_report(self):
        """Print a human-readable detailed report"""
        print("=" * 80)
        print("📊 DETAILED CODE ANALYSIS REPORT")
        print("=" * 80)
        
        report = self.generate_report()
        
        # Summary
        print(f"\n📈 SUMMARY:")
        print(f"   • Functions: {report['summary']['total_functions']}")
        print(f"   • Classes: {report['summary']['total_classes']}")
        print(f"   • Endpoints: {report['summary']['total_endpoints']}")
        print(f"   • Imports: {report['summary']['total_imports']}")
        print(f"   • Unused Functions: {report['summary']['unused_functions']}")
        print(f"   • Unused Classes: {report['summary']['unused_classes']}")
        
        # Endpoints
        print(f"\n🚀 ENDPOINTS ({len(self.endpoints)}):")
        for endpoint, info in self.endpoints.items():
            print(f"   • {endpoint} → {info['function_name']}() [line {info['line_number']}]")
        
        # Functions by category
        print(f"\n⚙️ FUNCTIONS BY CATEGORY:")
        print(f"   📡 Endpoint Functions ({len(report['functions']['endpoint_functions'])}):")
        for func in report['functions']['endpoint_functions']:
            line_num = self.functions[func]['line_number']
            print(f"      • {func}() [line {line_num}]")
        
        print(f"   🔧 Service Functions ({len(report['functions']['service_functions'])}):")
        for func in report['functions']['service_functions']:
            line_num = self.functions[func]['line_number']
            decorators = ', '.join(self.functions[func]['decorators'])
            print(f"      • {func}() [line {line_num}] ({decorators})")
        
        print(f"   🛠️  Utility Functions ({len(report['functions']['utility_functions'])}):")
        for func in report['functions']['utility_functions']:
            line_num = self.functions[func]['line_number']
            print(f"      • {func}() [line {line_num}]")
        
        # Classes
        print(f"\n📦 CLASSES ({len(self.classes)}):")
        for class_name, info in self.classes.items():
            bases = ', '.join(info['bases']) if info['bases'] else 'object'
            usage_count = len(self.class_usage.get(class_name, []))
            print(f"   • {class_name}({bases}) [line {info['line_number']}] - Used {usage_count} times")
        
        # Unused items
        if report['functions']['unused']:
            print(f"\n⚠️  UNUSED FUNCTIONS ({len(report['functions']['unused'])}):")
            for func in report['functions']['unused']:
                line_num = self.functions[func]['line_number']
                print(f"   • {func}() [line {line_num}]")
        
        if report['classes']['unused']:
            print(f"\n⚠️  UNUSED CLASSES ({len(report['classes']['unused'])}):")
            for cls in report['classes']['unused']:
                line_num = self.classes[cls]['line_number']
                print(f"   • {cls} [line {line_num}]")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if not report['functions']['unused'] and not report['classes']['unused']:
            print("   ✅ Great! No unused functions or classes detected.")
        else:
            if report['functions']['unused']:
                print(f"   • Consider removing {len(report['functions']['unused'])} unused functions")
            if report['classes']['unused']:
                print(f"   • Consider removing {len(report['classes']['unused'])} unused classes")
        
        endpoint_coverage = len(report['functions']['endpoint_functions']) / report['summary']['total_endpoints'] * 100 if report['summary']['total_endpoints'] > 0 else 0
        print(f"   • Endpoint coverage: {endpoint_coverage:.1f}% ({len(report['functions']['endpoint_functions'])}/{report['summary']['total_endpoints']})")
        
        print("\n" + "=" * 80)

def main():
    """Run detailed analysis"""
    analyzer = DetailedCodeAnalyzer()
    
    if analyzer.analyze():
        analyzer.print_detailed_report()
        
        # Save report
        import json
        from datetime import datetime
        
        report = analyzer.generate_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_code_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Detailed report saved to: {filename}")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main() 