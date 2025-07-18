from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from passlib.context import CryptContext
import uuid

Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    """
    🔐 PRODUCTION USER MODEL - MATCHES RAILWAY DATABASE SCHEMA
    
    Schema matches existing production database with 63 users:
    - id: integer (auto-increment, primary key)
    - username: string (required)
    - email: string (optional in production, but we'll require it)
    - phone_number: string (optional)
    - hashed_password: string (required - note different name)
    - is_active: boolean (optional, defaults to true)
    """
    __tablename__ = "ai_popup_users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False)  # Required in production
    email = Column(String, nullable=True)  # Optional in production schema
    phone_number = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)  # Production uses 'hashed_password' not 'password_hash'
    is_active = Column(Boolean, default=True)
    
    def set_password(self, password: str):
        """Hash and set password"""
        self.hashed_password = pwd_context.hash(password)
    
    def verify_password(self, password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(password, self.hashed_password)
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, username={self.username})>"

# For JWT tokens
class UserToken(Base):
    """
    🎫 SIMPLE TOKEN MODEL
    
    Track active sessions
    """
    __tablename__ = "ai_popup_user_tokens"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    token = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True) 

# Simple Session Management (No JWT needed)
class UserSession(Base):
    """
    🔑 ULTRA SIMPLE SESSION MODEL
    
    Store user sessions with simple session_id
    Perfect for browser extensions!
    """
    __tablename__ = "ai_popup_user_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    device_info = Column(String, nullable=True)  # Browser, extension info
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<UserSession(session_id={self.session_id}, user_id={self.user_id})>" 