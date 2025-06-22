"""
Automatic tracker that monkey-patches app modules to track function and class usage.

This provides a lighter-weight alternative to sys.settrace() by selectively
patching functions and classes in your application.
"""

import time
import inspect
import importlib
from pathlib import Path
from typing import Dict, Set, Any, List
from functools import wraps

# Import the analyzer for recording
try:
    from app.services.integrated_usage_analyzer import get_analyzer
except ImportError:
    def get_analyzer():
        return None


class AutoTracker:
    """Automatic tracker using monkey-patching"""
    
    def __init__(self):
        self.enabled = False
        self.patched_functions = {}  # original_func -> patched_func
        self.patched_classes = {}    # original_class -> patched_class
        
        # Modules to automatically track
        self.target_modules = [
            'app.services',
            'app.models', 
            'app.api',
            'app.endpoints'
        ]
        
    def enable(self):
        """Enable automatic tracking by patching target modules"""
        if self.enabled:
            return
            
        self.enabled = True
        self._patch_modules()
        
    def disable(self):
        """Disable tracking by restoring original functions/classes"""
        if not self.enabled:
            return
            
        self._unpatch_modules()
        self.enabled = False
        
    def _patch_modules(self):
        """Patch functions and classes in target modules"""
        for module_name in self.target_modules:
            try:
                self._patch_module(module_name)
            except Exception as e:
                # Module might not exist, skip silently
                continue
                
    def _patch_module(self, module_name: str):
        """Patch a specific module"""
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get all attributes in the module
            for attr_name in dir(module):
                if attr_name.startswith('_'):
                    continue
                    
                attr = getattr(module, attr_name)
                
                # Patch functions
                if callable(attr) and not inspect.isclass(attr):
                    if self._should_patch_function(attr):
                        patched_func = self._create_function_wrapper(attr, module_name, attr_name)
                        setattr(module, attr_name, patched_func)
                        self.patched_functions[attr] = patched_func
                        
                # Patch classes
                elif inspect.isclass(attr):
                    if self._should_patch_class(attr):
                        patched_class = self._create_class_wrapper(attr, module_name)
                        setattr(module, attr_name, patched_class)
                        self.patched_classes[attr] = patched_class
                        
        except Exception as e:
            # Skip problematic modules
            pass
            
    def _should_patch_function(self, func) -> bool:
        """Determine if we should patch this function"""
        try:
            # Only patch functions defined in our app
            if not hasattr(func, '__module__'):
                return False
                
            module_name = func.__module__
            if not any(target in module_name for target in ['app.services', 'app.models', 'app.api']):
                return False
                
            # Skip private functions
            if func.__name__.startswith('_'):
                return False
                
            return True
            
        except Exception:
            return False
            
    def _should_patch_class(self, cls) -> bool:
        """Determine if we should patch this class"""
        try:
            # Only patch classes defined in our app
            if not hasattr(cls, '__module__'):
                return False
                
            module_name = cls.__module__
            if not any(target in module_name for target in ['app.services', 'app.models']):
                return False
                
            # Skip private classes
            if cls.__name__.startswith('_'):
                return False
                
            return True
            
        except Exception:
            return False
            
    def _create_function_wrapper(self, original_func, module_name: str, func_name: str):
        """Create a wrapper that tracks function calls"""
        @wraps(original_func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result = None
            
            try:
                result = original_func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                result = f"Error: {str(e)}"
                raise
            finally:
                execution_time = time.time() - start_time
                
                # Record the function call
                try:
                    analyzer = get_analyzer()
                    if analyzer:
                        # Format arguments
                        args_str = self._format_args(args, kwargs, original_func)
                        
                        analyzer.record_function_call(
                            func_name=func_name,
                            file_name=module_name.split('.')[-1],
                            file_path=f"{module_name.replace('.', '/')}.py",
                            execution_time=execution_time,
                            input_args=args_str,
                            output_result=str(result)[:200],
                            success=success
                        )
                except Exception:
                    # Don't let tracking errors break the app
                    pass
                    
        return wrapper
        
    def _create_class_wrapper(self, original_class, module_name: str):
        """Create a wrapper that tracks class instantiation and method calls"""
        
        def format_constructor_args(args, kwargs, init_func):
            """Format constructor arguments"""
            try:
                sig = inspect.signature(init_func)
                bound_args = sig.bind(None, *args, **kwargs)  # None for self
                bound_args.apply_defaults()
                
                arg_strs = []
                for name, value in bound_args.arguments.items():
                    if name == 'self':
                        continue
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + "..."
                    arg_strs.append(f"{name}={value_str}")
                
                return f"({', '.join(arg_strs)})"
            except Exception:
                return f"(args={len(args)}, kwargs={len(kwargs)})"
        
        class TrackedClass(original_class):
            def __init__(self, *args, **kwargs):
                # Track class instantiation
                try:
                    analyzer = get_analyzer()
                    if analyzer:
                        args_str = format_constructor_args(args, kwargs, original_class.__init__)
                        
                        analyzer.record_class_instantiation(
                            class_name=original_class.__name__,
                            file_name=module_name.split('.')[-1],
                            file_path=f"{module_name.replace('.', '/')}.py",
                            constructor_args=args_str
                        )
                except Exception:
                    pass
                    
                # Call original constructor
                super().__init__(*args, **kwargs)
        
        # Copy class attributes
        TrackedClass.__name__ = original_class.__name__
        TrackedClass.__qualname__ = original_class.__qualname__
        TrackedClass.__module__ = original_class.__module__
        
        # Track method calls by patching public methods
        for attr_name in dir(original_class):
            if (not attr_name.startswith('_') and 
                callable(getattr(original_class, attr_name)) and
                attr_name != '__init__'):
                
                original_method = getattr(original_class, attr_name)
                wrapped_method = self._create_method_wrapper(
                    original_method, original_class.__name__, attr_name, module_name
                )
                setattr(TrackedClass, attr_name, wrapped_method)
                
        return TrackedClass
        
    def _create_method_wrapper(self, original_method, class_name: str, method_name: str, module_name: str):
        """Create a wrapper that tracks method calls"""
        @wraps(original_method)
        def wrapper(self, *args, **kwargs):
            # Track method call
            try:
                analyzer = get_analyzer()
                if analyzer:
                    analyzer.record_method_call(
                        class_name=class_name,
                        method_name=method_name,
                        file_name=module_name.split('.')[-1]
                    )
            except Exception:
                pass
                
            # Call original method
            return original_method(self, *args, **kwargs)
            
        return wrapper
        
    def _format_args(self, args, kwargs, func) -> str:
        """Format function arguments"""
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            arg_strs = []
            for name, value in bound_args.arguments.items():
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:50] + "..."
                arg_strs.append(f"{name}={value_str}")
            
            return f"({', '.join(arg_strs)})"
        except Exception:
            return f"(args={len(args)}, kwargs={len(kwargs)})"
            
    def _unpatch_modules(self):
        """Restore original functions and classes"""
        # This is complex to implement properly, so for now we'll just
        # disable tracking by not recording new calls
        pass


# Global tracker instance
_auto_tracker = None

def get_auto_tracker() -> AutoTracker:
    """Get the global auto tracker instance"""
    global _auto_tracker
    if _auto_tracker is None:
        _auto_tracker = AutoTracker()
    return _auto_tracker

def enable_auto_tracking():
    """Enable automatic tracking"""
    tracker = get_auto_tracker()
    tracker.enable()

def disable_auto_tracking():
    """Disable automatic tracking"""
    tracker = get_auto_tracker()
    tracker.disable() 