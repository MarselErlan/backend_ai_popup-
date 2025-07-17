"""
URL Tracking Service - Handles saving and managing URLs from browser extension
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from db.models import TrackedUrl
from db.schemas import SaveUrlRequest, UpdateUrlStatusRequest, TrackedUrlResponse, TrackedUrlsListResponse
from db.database import SessionLocal
from app.utils.logger import logger


class UrlTrackingService:
    """Service for managing URL tracking functionality"""
    
    def __init__(self):
        """Initialize the service"""
        pass
    
    def get_db_session(self) -> Session:
        """Get database session"""
        return SessionLocal()
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception as e:
            logger.warning(f"Failed to extract domain from {url}: {e}")
            return ""
    
    def save_url(self, user_id: str, request: SaveUrlRequest) -> Dict[str, Any]:
        """
        Save a URL from browser extension
        
        Args:
            user_id: ID of the user saving the URL
            request: SaveUrlRequest with URL and metadata
            
        Returns:
            Dict with operation result and URL data
        """
        session = self.get_db_session()
        try:
            # Check if URL already exists for this user
            existing = session.query(TrackedUrl).filter(
                TrackedUrl.user_id == user_id,
                TrackedUrl.url == request.url,
                TrackedUrl.is_active == True
            ).first()
            
            if existing:
                return {
                    "status": "exists",
                    "message": "URL already tracked",
                    "url": self._convert_to_response(existing)
                }
            
            # Extract domain
            domain = self.extract_domain(request.url)
            
            # Create new tracked URL
            tracked_url = TrackedUrl(
                user_id=user_id,
                url=request.url,
                title=request.title,
                domain=domain,
                notes=request.notes,
                status="not_applied"
            )
            
            session.add(tracked_url)
            session.commit()
            session.refresh(tracked_url)
            
            logger.info(f"✅ URL saved for user {user_id}: {request.url}")
            
            return {
                "status": "success",
                "message": "URL saved successfully",
                "url": self._convert_to_response(tracked_url)
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to save URL: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            session.close()
    
    def get_user_urls(self, user_id: str, status_filter: Optional[str] = None) -> TrackedUrlsListResponse:
        """
        Get all URLs for a user with optional status filtering
        
        Args:
            user_id: ID of the user
            status_filter: Optional status to filter by
            
        Returns:
            TrackedUrlsListResponse with user's URLs
        """
        session = self.get_db_session()
        try:
            query = session.query(TrackedUrl).filter(
                TrackedUrl.user_id == user_id,
                TrackedUrl.is_active == True
            )
            
            if status_filter:
                query = query.filter(TrackedUrl.status == status_filter)
            
            urls = query.order_by(desc(TrackedUrl.created_at)).all()
            
            url_responses = [self._convert_to_response(url) for url in urls]
            
            return TrackedUrlsListResponse(
                urls=url_responses,
                total=len(url_responses)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get user URLs: {e}")
            return TrackedUrlsListResponse(urls=[], total=0)
        finally:
            session.close()
    
    def update_url_status(self, user_id: str, url_id: str, request: UpdateUrlStatusRequest) -> Dict[str, Any]:
        """
        Update the application status of a tracked URL
        
        Args:
            user_id: ID of the user
            url_id: ID of the URL to update
            request: UpdateUrlStatusRequest with new status
            
        Returns:
            Dict with operation result and updated URL data
        """
        session = self.get_db_session()
        try:
            # Find the URL
            tracked_url = session.query(TrackedUrl).filter(
                TrackedUrl.id == url_id,
                TrackedUrl.user_id == user_id,
                TrackedUrl.is_active == True
            ).first()
            
            if not tracked_url:
                return {
                    "status": "not_found",
                    "message": "URL not found"
                }
            
            # Update status
            old_status = tracked_url.status
            tracked_url.status = request.status
            tracked_url.updated_at = datetime.utcnow()
            
            # Set applied_at timestamp if status changed to applied
            if request.status == "applied" and old_status != "applied":
                tracked_url.applied_at = datetime.utcnow()
            elif request.status != "applied":
                tracked_url.applied_at = None
            
            # Update notes if provided
            if request.notes is not None:
                tracked_url.notes = request.notes
            
            session.commit()
            session.refresh(tracked_url)
            
            logger.info(f"✅ URL status updated for user {user_id}: {url_id} -> {request.status}")
            
            return {
                "status": "success",
                "message": "URL status updated successfully",
                "url": self._convert_to_response(tracked_url)
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to update URL status: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            session.close()
    
    def delete_url(self, user_id: str, url_id: str) -> Dict[str, Any]:
        """
        Soft delete a tracked URL (mark as inactive)
        
        Args:
            user_id: ID of the user
            url_id: ID of the URL to delete
            
        Returns:
            Dict with operation result
        """
        session = self.get_db_session()
        try:
            # Find the URL
            tracked_url = session.query(TrackedUrl).filter(
                TrackedUrl.id == url_id,
                TrackedUrl.user_id == user_id,
                TrackedUrl.is_active == True
            ).first()
            
            if not tracked_url:
                return {
                    "status": "not_found",
                    "message": "URL not found"
                }
            
            # Soft delete
            tracked_url.is_active = False
            tracked_url.updated_at = datetime.utcnow()
            
            session.commit()
            
            logger.info(f"✅ URL deleted for user {user_id}: {url_id}")
            
            return {
                "status": "success",
                "message": "URL deleted successfully"
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to delete URL: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            session.close()
    
    def get_url_by_id(self, user_id: str, url_id: str) -> Optional[TrackedUrlResponse]:
        """
        Get a specific URL by ID
        
        Args:
            user_id: ID of the user
            url_id: ID of the URL
            
        Returns:
            TrackedUrlResponse or None if not found
        """
        session = self.get_db_session()
        try:
            tracked_url = session.query(TrackedUrl).filter(
                TrackedUrl.id == url_id,
                TrackedUrl.user_id == user_id,
                TrackedUrl.is_active == True
            ).first()
            
            if tracked_url:
                return self._convert_to_response(tracked_url)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get URL by ID: {e}")
            return None
        finally:
            session.close()
    
    def _convert_to_response(self, tracked_url: TrackedUrl) -> TrackedUrlResponse:
        """Convert database model to response model"""
        return TrackedUrlResponse(
            id=tracked_url.id,
            url=tracked_url.url,
            title=tracked_url.title,
            domain=tracked_url.domain,
            status=tracked_url.status,
            applied_at=tracked_url.applied_at.isoformat() if tracked_url.applied_at else None,
            created_at=tracked_url.created_at.isoformat(),
            updated_at=tracked_url.updated_at.isoformat(),
            notes=tracked_url.notes,
            is_active=tracked_url.is_active
        ) 