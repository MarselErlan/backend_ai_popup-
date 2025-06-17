"""
Database models for the Smart Form Fill API
These represent the data structures as they are stored in PostgreSQL.
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, date


class FormDb(BaseModel):
    """Database model for form entries"""
    id: str
    url: str
    status: str  # "applied" or "not_applied"
    applied_counter: int = 0
    applied_date: Optional[str] = None  # ISO date string
    created_at: str
    
    class Config:
        from_attributes = True


class FormField(BaseModel):
    """Model for a form field extracted during analysis"""
    field_type: str
    purpose: str
    selector: str
    validation: Optional[str] = None


class PostgresResult(BaseModel):
    """Result model for PostgreSQL operations"""
    status: str
    message: str
    id: Optional[str] = None
