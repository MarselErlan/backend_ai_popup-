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

# TrackedUrl model removed - URL tracking feature disabled

# FormDb, FormField, and PostgresResult models removed - URL tracking feature disabled
