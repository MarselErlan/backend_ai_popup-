"""
Database models for document storage
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, LargeBinary, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ResumeDocument(Base):
    """Resume document model"""
    __tablename__ = "resume_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # Changed from String to Integer to match new auth system
    filename = Column(String(255))
    file_content = Column(LargeBinary)
    content_type = Column(String(100))
    file_size = Column(Integer)
    processing_status = Column(String(50), default="pending")

class PersonalInfoDocument(Base):
    """Personal info document model"""
    __tablename__ = "personal_info_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # Changed from String to Integer to match new auth system
    filename = Column(String(255))
    file_content = Column(LargeBinary)
    content_type = Column(String(100))
    file_size = Column(Integer)
    processing_status = Column(String(50), default="pending")

class DocumentProcessingLog(Base):
    """Log for document processing operations"""
    __tablename__ = "document_processing_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    document_type = Column(String(50))  # "resume" or "personal_info"
    document_id = Column(Integer)
    user_id = Column(Integer, nullable=True)  # Changed from String to Integer to match new auth system
    status = Column(String(50))  # "started", "completed", "failed"
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    processing_time = Column(Integer, nullable=True)  # in seconds
    total_chunks = Column(Integer, nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    model_used = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True) 