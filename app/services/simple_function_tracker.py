"""
Simple function tracker that manually tracks key service functions.

This provides a reliable way to track function and class usage by adding
tracking calls to key service functions.
"""

import time
import functools
from typing import Any, Callable

# Import the analyzer for recording
try:
    from app.services.integrated_usage_analyzer import get_analyzer
except ImportError:
    def get_analyzer():
        return None


def track_service_call(func_name: str, file_name: str, execution_time: float, 
                      input_args: str, output_result: str, success: bool = True):
    """Manually track a service function call"""
    try:
        analyzer = get_analyzer()
        if analyzer:
            analyzer.record_function_call(
                func_name=func_name,
                file_name=file_name,
                file_path=f"app/services/{file_name}.py",
                execution_time=execution_time,
                input_args=input_args,
                output_result=str(output_result)[:200],
                success=success
            )
    except Exception:
        # Don't let tracking errors break the app
        pass


def track_class_creation(class_name: str, file_name: str, constructor_args: str):
    """Manually track a class instantiation"""
    try:
        analyzer = get_analyzer()
        if analyzer:
            analyzer.record_class_instantiation(
                class_name=class_name,
                file_name=file_name,
                file_path=f"app/services/{file_name}.py",
                constructor_args=constructor_args
            )
    except Exception:
        # Don't let tracking errors break the app
        pass


def track_method_call(class_name: str, method_name: str, file_name: str):
    """Manually track a method call"""
    try:
        analyzer = get_analyzer()
        if analyzer:
            analyzer.record_method_call(
                class_name=class_name,
                method_name=method_name,
                file_name=file_name
            )
    except Exception:
        # Don't let tracking errors break the app
        pass


# Decorator for easy function tracking
def track_function(func: Callable) -> Callable:
    """Decorator to track function calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        result = None
        
        # Get function metadata
        func_name = func.__name__
        module_name = func.__module__.split('.')[-1] if hasattr(func, '__module__') else 'unknown'
        
        # Format arguments
        try:
            args_str = f"(args={len(args)}, kwargs={len(kwargs)})"
            if args:
                first_arg = str(args[0])[:50]
                args_str = f"({first_arg}{'...' if len(str(args[0])) > 50 else ''})"
        except:
            args_str = "(args unavailable)"
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            result = f"Error: {str(e)}"
            raise
        finally:
            execution_time = time.time() - start_time
            track_service_call(func_name, module_name, execution_time, args_str, str(result), success)
    
    return wrapper


# Decorator for easy class tracking
def track_class(cls):
    """Decorator to track class instantiation and method calls"""
    original_init = cls.__init__
    
    @functools.wraps(original_init)
    def tracked_init(self, *args, **kwargs):
        # Track class instantiation
        module_name = cls.__module__.split('.')[-1] if hasattr(cls, '__module__') else 'unknown'
        try:
            args_str = f"(args={len(args)}, kwargs={len(kwargs)})"
            if args:
                first_arg = str(args[0])[:50]
                args_str = f"({first_arg}{'...' if len(str(args[0])) > 50 else ''})"
        except:
            args_str = "(args unavailable)"
            
        track_class_creation(cls.__name__, module_name, args_str)
        
        # Call original constructor
        original_init(self, *args, **kwargs)
    
    # Replace __init__
    cls.__init__ = tracked_init
    
    # Track public methods
    for attr_name in dir(cls):
        if (not attr_name.startswith('_') and 
            callable(getattr(cls, attr_name)) and
            attr_name != '__init__'):
            
            original_method = getattr(cls, attr_name)
            if callable(original_method):
                wrapped_method = create_method_wrapper(original_method, cls.__name__, attr_name, cls.__module__.split('.')[-1])
                setattr(cls, attr_name, wrapped_method)
    
    return cls


def create_method_wrapper(original_method, class_name: str, method_name: str, module_name: str):
    """Create a wrapper for method tracking"""
    @functools.wraps(original_method)
    def wrapper(self, *args, **kwargs):
        # Track method call
        track_method_call(class_name, method_name, module_name)
        
        # Call original method
        return original_method(self, *args, **kwargs)
    
    return wrapper 