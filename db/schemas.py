"""
API schemas for the Smart Form Fill API
These define the shape of requests and responses for API endpoints.
"""

from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Dict, List, Any, Optional
from datetime import datetime


class FormAnalysisRequest(BaseModel):
    """Request model for form analysis"""
    url: HttpUrl = Field(..., description="URL of the form to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/job-application"
            }
        }


class FormResponse(BaseModel):
    """Response model for form entries"""
    id: str
    url: str
    created_at: str
    analyzed: bool
    status: str  # "applied" or "not_applied"
    applied_counter: int = 0
    applied_date: Optional[str] = None  # ISO date string
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com/job-application",
                "created_at": "2023-06-01T12:00:00Z",
                "analyzed": True,
                "status": "not_applied",
                "applied_counter": 0,
                "applied_date": None
            }
        }


# URL Tracking Schemas
class SaveUrlRequest(BaseModel):
    """Request model for saving URL from browser extension"""
    url: str = Field(..., description="URL to save")
    title: Optional[str] = Field(None, description="Page title from browser")
    notes: Optional[str] = Field(None, description="Optional user notes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://jobs.company.com/apply/123",
                "title": "Software Engineer - Company Inc",
                "notes": "Interesting role, good benefits"
            }
        }


class UpdateUrlStatusRequest(BaseModel):
    """Request model for updating URL application status"""
    status: str = Field(..., description="New status: not_applied, applied, in_progress")
    notes: Optional[str] = Field(None, description="Optional status update notes")
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['not_applied', 'applied', 'in_progress']
        if v not in allowed_statuses:
            raise ValueError(f'Status must be one of: {", ".join(allowed_statuses)}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "applied",
                "notes": "Applied via company website"
            }
        }


class TrackedUrlResponse(BaseModel):
    """Response model for tracked URLs"""
    id: str
    url: str
    title: Optional[str]
    domain: Optional[str]
    status: str
    applied_at: Optional[str]  # ISO datetime string
    created_at: str  # ISO datetime string
    updated_at: str  # ISO datetime string
    notes: Optional[str]
    is_active: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://jobs.company.com/apply/123",
                "title": "Software Engineer - Company Inc",
                "domain": "jobs.company.com",
                "status": "not_applied",
                "applied_at": None,
                "created_at": "2023-06-01T12:00:00Z",
                "updated_at": "2023-06-01T12:00:00Z",
                "notes": "Interesting role, good benefits",
                "is_active": True
            }
        }


class TrackedUrlsListResponse(BaseModel):
    """Response model for list of tracked URLs"""
    urls: List[TrackedUrlResponse]
    total: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "url": "https://jobs.company.com/apply/123",
                        "title": "Software Engineer - Company Inc",
                        "domain": "jobs.company.com",
                        "status": "applied",
                        "applied_at": "2023-06-01T14:30:00Z",
                        "created_at": "2023-06-01T12:00:00Z",
                        "updated_at": "2023-06-01T14:30:00Z",
                        "notes": "Applied successfully",
                        "is_active": True
                    }
                ],
                "total": 1
            }
        }


class FormFieldSchema(BaseModel):
    """Schema for a form field extracted during analysis"""
    field_type: str
    purpose: str
    selector: str
    validation: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "field_type": "text",
                "purpose": "full_name",
                "selector": "#fullName",
                "validation": "required"
            }
        }


class FormAnalysisResponse(BaseModel):
    """Response model for form analysis results"""
    status: str
    field_map: str
    timestamp: str
    database_id: Optional[str] = None
    url: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "field_map": "Field 1: Name, Field 2: Email...",
                "timestamp": "2023-06-01T12:00:00Z",
                "database_id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com/job-application"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy"
            }
        }


class DetailedFormAnalysisResponse(FormAnalysisResponse):
    """Extended response model with structured field data"""
    fields: List[FormFieldSchema] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "field_map": "Field 1: Name, Field 2: Email...",
                "timestamp": "2023-06-01T12:00:00Z",
                "database_id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com/job-application",
                "fields": [
                    {
                        "field_type": "text",
                        "purpose": "full_name",
                        "selector": "#fullName",
                        "validation": "required"
                    }
                ]
            }
        } 
