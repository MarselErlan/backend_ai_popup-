"""
URL Tracking API Endpoints - Handles browser extension URL saving and status management
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, Dict, Any

from db.schemas import (
    SaveUrlRequest, 
    UpdateUrlStatusRequest, 
    TrackedUrlResponse, 
    TrackedUrlsListResponse
)
from app.services.url_tracking_service import UrlTrackingService
from database import get_db
from models import User, UserSession
from app.utils.logger import logger
from sqlalchemy.orm import Session
from fastapi import Header
from datetime import datetime

router = APIRouter(prefix="/api/urls", tags=["URL Tracking"])

# Initialize service
url_service = UrlTrackingService()

# Session-based authentication dependency
def get_session_user(db: Session = Depends(get_db), session_id: str = Header(None, alias="Authorization")):
    """
    Dependency to get the current user based on session ID from Authorization header
    Raises HTTPException if session is invalid or expired
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="No session ID provided")
    
    # Query the active session
    session = db.query(UserSession).filter(
        UserSession.session_id == session_id,
        UserSession.is_active == True
    ).first()
    
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    # Get the associated user
    user = db.query(User).filter(User.id == session.user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Update last_used_at timestamp
    session.last_used_at = datetime.utcnow()
    db.commit()
    
    return user


@router.post("/save", response_model=Dict[str, Any])
async def save_url(
    request: SaveUrlRequest,
    user: User = Depends(get_session_user)
):
    """
    Save a URL from browser extension
    
    This endpoint is called by the browser extension when a user
    wants to save a job application URL for tracking.
    """
    try:
        result = url_service.save_url(user.id, request)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to save URL: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/list", response_model=TrackedUrlsListResponse)
async def get_user_urls(
    status: Optional[str] = Query(None, description="Filter by status: not_applied, applied, in_progress"),
    user: User = Depends(get_session_user)
):
    """
    Get all tracked URLs for the current user
    
    Optionally filter by application status.
    Returns URLs ordered by creation date (newest first).
    """
    try:
        # Validate status filter if provided
        if status and status not in ["not_applied", "applied", "in_progress"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid status. Must be: not_applied, applied, or in_progress"
            )
        
        result = url_service.get_user_urls(user.id, status)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get user URLs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{url_id}", response_model=TrackedUrlResponse)
async def get_url_by_id(
    url_id: str,
    user: User = Depends(get_session_user)
):
    """
    Get a specific tracked URL by ID
    """
    try:
        result = url_service.get_url_by_id(user.id, url_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="URL not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get URL by ID: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{url_id}/status", response_model=Dict[str, Any])
async def update_url_status(
    url_id: str,
    request: UpdateUrlStatusRequest,
    user: User = Depends(get_session_user)
):
    """
    Update the application status of a tracked URL
    
    This endpoint is called when a user clicks the "Apply" button
    or updates the status of a job application.
    """
    try:
        result = url_service.update_url_status(user.id, url_id, request)
        
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail=result["message"])
        elif result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update URL status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{url_id}", response_model=Dict[str, Any])
async def delete_url(
    url_id: str,
    user: User = Depends(get_session_user)
):
    """
    Delete a tracked URL (soft delete)
    
    This marks the URL as inactive rather than permanently deleting it.
    """
    try:
        result = url_service.delete_url(user.id, url_id)
        
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail=result["message"])
        elif result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete URL: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats/summary", response_model=Dict[str, Any])
async def get_url_stats(
    user: User = Depends(get_session_user)
):
    """
    Get URL tracking statistics for the user
    
    Returns counts by status and other useful metrics.
    """
    try:
        # Get all URLs
        all_urls = url_service.get_user_urls(user.id)
        
        # Calculate stats
        total = all_urls.total
        not_applied = len([url for url in all_urls.urls if url.status == "not_applied"])
        applied = len([url for url in all_urls.urls if url.status == "applied"])
        in_progress = len([url for url in all_urls.urls if url.status == "in_progress"])
        
        # Recent activity (last 7 days)
        from datetime import datetime, timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_urls = [
            url for url in all_urls.urls 
            if datetime.fromisoformat(url.created_at.replace('Z', '+00:00')) > week_ago
        ]
        
        return {
            "status": "success",
            "stats": {
                "total_urls": total,
                "not_applied": not_applied,
                "applied": applied,
                "in_progress": in_progress,
                "recent_activity": len(recent_urls),
                "application_rate": round((applied / total * 100) if total > 0 else 0, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get URL stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") 