"""
Usage Analysis Middleware

This middleware captures all FastAPI requests and sends them to the usage analyzer.
It also enables runtime tracking of function calls and class instantiations.
"""

import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from loguru import logger

class UsageAnalysisMiddleware(BaseHTTPMiddleware):
    """Middleware to capture API requests and runtime function/class usage"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next):
        """Capture request, response, and runtime function/class usage"""
        start_time = time.time()
        
        # Process the request (auto-tracking is global, no per-request setup needed)
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Record the request (import here to avoid circular imports)
        try:
            from app.services.integrated_usage_analyzer import record_request
            record_request(request, response_time, response.status_code)
        except Exception as e:
            logger.debug(f"Failed to record request: {e}")
        
        return response 