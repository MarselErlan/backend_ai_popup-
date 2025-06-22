"""
Decorators for automatic function and class usage tracking.

Usage:
    @track_function
    def my_function(arg1, arg2):
        return result
        
    @track_class
    class MyClass:
        def __init__(self, arg):
            self.arg = arg
"""

import functools
import time
import inspect
from pathlib import Path
from typing import Any, Callable

# Import the global tracking functions
try:
    from app.services.integrated_usage_analyzer import (
        record_function_call, 
        record_class_instantiation, 
        record_method_call
    )
except ImportError:
    # Fallback if analyzer not available
    def record_function_call(*args, **kwargs):
        pass
    def record_class_instantiation(*args, **kwargs):
        pass
    def record_method_call(*args, **kwargs):
        pass


def track_function(func: Callable) -> Callable:
    """
    Decorator to automatically track function calls.
    
    Usage:
        @track_function
        def my_function(arg1: str, arg2: int) -> str:
            return f"{arg1}_{arg2}"
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        result = None
        
        # Get function metadata
        func_name = func.__name__
        file_path = inspect.getfile(func)
        file_name = Path(file_path).stem
        
        # Format input arguments
        input_args = _format_args(args, kwargs, func)
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            result = f"Error: {str(e)}"
            raise
        finally:
            execution_time = time.time() - start_time
            
            # Record the function call
            record_function_call(
                func_name=func_name,
                file_name=file_name,
                file_path=file_path,
                execution_time=execution_time,
                input_args=input_args,
                output_result=str(result),
                success=success
            )
    
    return wrapper


def track_class(cls):
    """
    Class decorator to automatically track class instantiation and method calls.
    
    Usage:
        @track_class
        class MyClass:
            def __init__(self, arg):
                self.arg = arg
                
            def my_method(self):
                return self.arg
    """
    original_init = cls.__init__
    
    @functools.wraps(original_init)
    def tracked_init(self, *args, **kwargs):
        # Get class metadata
        class_name = cls.__name__
        file_path = inspect.getfile(cls)
        file_name = Path(file_path).stem
        
        # Format constructor arguments
        constructor_args = _format_args(args, kwargs, original_init)
        
        # Record class instantiation
        record_class_instantiation(
            class_name=class_name,
            file_name=file_name,
            file_path=file_path,
            constructor_args=constructor_args
        )
        
        # Call original __init__
        original_init(self, *args, **kwargs)
    
    # Replace __init__ with tracked version
    cls.__init__ = tracked_init
    
    # Track all methods
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if (callable(attr) and 
            not attr_name.startswith('_') and 
            not attr_name.startswith('__')):
            
            # Wrap method to track calls
            wrapped_method = _track_method(attr, cls.__name__, Path(inspect.getfile(cls)).stem)
            setattr(cls, attr_name, wrapped_method)
    
    return cls


def _track_method(method, class_name: str, file_name: str):
    """Helper to wrap a method for tracking"""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Record method call
        record_method_call(
            class_name=class_name,
            method_name=method.__name__,
            file_name=file_name
        )
        
        # Call original method
        return method(self, *args, **kwargs)
    
    return wrapper


def _format_args(args, kwargs, func) -> str:
    """Format function arguments for logging"""
    try:
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Format arguments
        arg_strs = []
        for name, value in bound_args.arguments.items():
            if name == 'self':
                continue
            
            # Limit argument string length
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            
            arg_strs.append(f"{name}={value_str}")
        
        return f"({', '.join(arg_strs)})"
    
    except Exception:
        # Fallback to simple string representation
        args_str = ', '.join([str(arg)[:50] for arg in args])
        kwargs_str = ', '.join([f"{k}={str(v)[:50]}" for k, v in kwargs.items()])
        all_args = [args_str, kwargs_str] if kwargs_str else [args_str]
        return f"({', '.join(filter(None, all_args))})"


# Convenience decorators for specific use cases
def track_api_function(func: Callable) -> Callable:
    """Decorator specifically for API endpoint functions"""
    return track_function(func)


def track_service_function(func: Callable) -> Callable:
    """Decorator specifically for service layer functions"""
    return track_function(func)


def track_model_class(cls):
    """Decorator specifically for model classes"""
    return track_class(cls)


def track_service_class(cls):
    """Decorator specifically for service classes"""
    return track_class(cls) 