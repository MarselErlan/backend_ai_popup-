"""
Database models for document storage
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, LargeBinary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class ResumeDocument(Base):
    """Table for storing resume documents"""
    __tablename__ = 'resume_documents'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=True, index=True)  # For multi-user support
    filename = Column(String(255), nullable=False)
    processing_status = Column(String(50), default='pending')  # pending, processing, completed, failed

    def __repr__(self):
        return f"<ResumeDocument(id={self.id}, filename='{self.filename}', user_id='{self.user_id}')>"


class PersonalInfoDocument(Base):
    """Table for storing personal information documents"""
    __tablename__ = 'personal_info_documents'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=True, index=True)  # For multi-user support
    filename = Column(String(255), nullable=False)
    file_content = Column(LargeBinary, nullable=False)  # Store file binary data
    content_type = Column(String(100), nullable=False)  # MIME type (e.g., 'text/plain', 'application/pdf')
    file_size = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)  # For soft delete
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Processing status
    last_processed_at = Column(DateTime(timezone=True), nullable=True)
    processing_status = Column(String(50), default='pending')  # pending, processing, completed, failed
    
    def __repr__(self):
        return f"<PersonalInfoDocument(id={self.id}, filename='{self.filename}', user_id='{self.user_id}')>"


class DocumentProcessingLog(Base):
    """Table for tracking document processing history"""
    __tablename__ = 'document_processing_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    document_type = Column(String(50), nullable=False)  # 'resume' or 'personal_info'
    document_id = Column(Integer, nullable=False)
    user_id = Column(String(100), nullable=True, index=True)
    
    # Processing details
    processing_status = Column(String(50), nullable=False)  # started, completed, failed
    processing_time_seconds = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Embedding details
    total_chunks = Column(Integer, nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    model_used = Column(String(100), nullable=True)
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<DocumentProcessingLog(id={self.id}, document_type='{self.document_type}', status='{self.processing_status}')>" 