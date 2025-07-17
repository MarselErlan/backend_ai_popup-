"""
Database models for the Smart Form Fill API
These represent the data structures as they are stored in PostgreSQL.
"""

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, HttpUrl, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import bcrypt
import uuid

Base = declarative_base()

class User(Base):
    """User model for authentication"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def set_password(self, password: str):
        """Hash and set the password"""
        salt = bcrypt.gensalt()
        self.hashed_password = bcrypt.hashpw(password.encode(), salt).decode()

    def verify_password(self, password: str) -> bool:
        """Verify the password"""
        return bcrypt.checkpw(password.encode(), self.hashed_password.encode())

class UserSession(Base):
    """User session model for simple session management"""
    __tablename__ = "user_sessions"

    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), index=True)
    device_info = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class TrackedUrl(Base):
    """Model for tracking URLs saved from browser extension"""
    __tablename__ = "tracked_urls"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), index=True)
    url = Column(String, nullable=False)
    title = Column(String, nullable=True)  # Page title from browser
    domain = Column(String, nullable=True)  # Extracted domain
    status = Column(String, default="not_applied")  # "not_applied", "applied", "in_progress"
    applied_at = Column(DateTime, nullable=True)  # When status changed to applied
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(String, nullable=True)  # User notes
    is_active = Column(Boolean, default=True)

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
