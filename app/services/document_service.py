"""
Document service for database operations
"""
import io
import tempfile
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import os

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, desc
from loguru import logger

from app.models.document_models import Base, ResumeDocument, PersonalInfoDocument, DocumentProcessingLog


class DocumentService:
    """Service for managing documents in database"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url or os.getenv("DATABASE_URL", "postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqTpIKa@interchange.proxy.rlwy.net:30153/railway")
        self.engine = create_engine(self.database_url)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
        logger.info("📊 Database tables created/verified")
    
    def get_session(self) -> Session:
        """Get database session"""
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        return SessionLocal()
    
    # ============================================================================
    # RESUME DOCUMENT OPERATIONS
    # ============================================================================
    
    def save_resume_document(
        self, 
        filename: str, 
        file_content: bytes,
        content_type: str,
        user_id: int = None
    ) -> int:
        """Save resume document to database"""
        try:
            with self.get_session() as session:
                # Delete previous resume for this user
                if user_id:
                    session.query(ResumeDocument).filter(
                        ResumeDocument.user_id == user_id
                    ).delete()
                
                resume_doc = ResumeDocument(
                    user_id=user_id,
                    filename=filename,
                    file_content=file_content,
                    content_type=content_type,
                    file_size=len(file_content),
                    processing_status='pending'
                )
                
                session.add(resume_doc)
                session.commit()
                session.refresh(resume_doc)
                
                logger.info(f"💾 Saved resume document: {filename} (ID: {resume_doc.id}, Size: {resume_doc.file_size} bytes)")
                return resume_doc.id
                
        except Exception as e:
            logger.error(f"❌ Error saving resume document: {e}")
            raise
    
    def get_user_resume(self, user_id: int = None) -> Optional[ResumeDocument]:
        """Get the resume document for a user"""
        try:
            with self.get_session() as session:
                query = session.query(ResumeDocument)
                
                if user_id:
                    query = query.filter(ResumeDocument.user_id == user_id)
                
                resume_doc = query.order_by(desc(ResumeDocument.id)).first()
                
                if resume_doc:
                    logger.info(f"📄 Retrieved resume: {resume_doc.filename} (ID: {resume_doc.id}, Size: {resume_doc.file_size} bytes)")
                else:
                    logger.warning("⚠️ No resume document found")
                
                return resume_doc
                
        except Exception as e:
            logger.error(f"❌ Error retrieving resume document: {e}")
            raise
    
    def update_resume_status(self, document_id: int, status: str):
        """Update resume processing status"""
        try:
            with self.get_session() as session:
                session.query(ResumeDocument).filter(
                    ResumeDocument.id == document_id
                ).update({
                    "processing_status": status
                })
                session.commit()
                
                logger.info(f"✅ Updated resume status: {status} (ID: {document_id})")
                
        except Exception as e:
            logger.error(f"❌ Error updating resume status: {e}")
            raise
    
    # ============================================================================
    # PERSONAL INFO DOCUMENT OPERATIONS
    # ============================================================================
    
    def save_personal_info_document(
        self, 
        filename: str, 
        file_content: bytes,
        content_type: str,
        user_id: int = None
    ) -> int:
        """Save personal info document to database"""
        try:
            with self.get_session() as session:
                # Delete previous personal info for this user
                if user_id:
                    session.query(PersonalInfoDocument).filter(
                        PersonalInfoDocument.user_id == user_id
                    ).delete()
                
                personal_info_doc = PersonalInfoDocument(
                    user_id=user_id,
                    filename=filename,
                    file_content=file_content,
                    content_type=content_type,
                    file_size=len(file_content),
                    processing_status='pending'
                )
                
                session.add(personal_info_doc)
                session.commit()
                session.refresh(personal_info_doc)
                
                logger.info(f"💾 Saved personal info document: {filename} (ID: {personal_info_doc.id}, Size: {personal_info_doc.file_size} bytes)")
                return personal_info_doc.id
                
        except Exception as e:
            logger.error(f"❌ Error saving personal info document: {e}")
            raise
    
    def get_personal_info_document(self, user_id: int = None) -> Optional[PersonalInfoDocument]:
        """Get the personal info document for a user"""
        try:
            with self.get_session() as session:
                query = session.query(PersonalInfoDocument)
                
                if user_id:
                    query = query.filter(PersonalInfoDocument.user_id == user_id)
                
                personal_info_doc = query.order_by(desc(PersonalInfoDocument.id)).first()
                
                if personal_info_doc:
                    logger.info(f"📄 Retrieved personal info: {personal_info_doc.filename} (ID: {personal_info_doc.id}, Size: {personal_info_doc.file_size} bytes)")
                else:
                    logger.warning("⚠️ No personal info document found")
                
                return personal_info_doc
                
        except Exception as e:
            logger.error(f"❌ Error retrieving personal info document: {e}")
            raise
    
    def update_personal_info_status(self, document_id: int, status: str):
        """Update personal info processing status"""
        try:
            with self.get_session() as session:
                session.query(PersonalInfoDocument).filter(
                    PersonalInfoDocument.id == document_id
                ).update({
                    "processing_status": status
                })
                session.commit()
                
                logger.info(f"✅ Updated personal info status: {status} (ID: {document_id})")
                
        except Exception as e:
            logger.error(f"❌ Error updating personal info status: {e}")
            raise
    
    # ============================================================================
    # CONVENIENCE METHODS
    # ============================================================================
    
    def store_resume(self, user_id: int, filename: str, content: str) -> int:
        """Store resume content as text"""
        return self.save_resume_document(
            filename=filename,
            file_content=content.encode('utf-8'),
            content_type='text/plain',
            user_id=user_id
        )
    
    def store_personal_info(self, user_id: int, filename: str, content: str) -> int:
        """Store personal info content as text"""
        return self.save_personal_info_document(
            filename=filename,
            file_content=content.encode('utf-8'),
            content_type='text/plain',
            user_id=user_id
        )
    
    def get_resume_content(self, user_id: int) -> Optional[str]:
        """Get resume content as text"""
        resume_doc = self.get_user_resume(user_id)
        if resume_doc:
            return resume_doc.file_content.decode('utf-8')
        return None
    
    def get_personal_info_content(self, user_id: int) -> Optional[str]:
        """Get personal info content as text"""
        personal_info_doc = self.get_personal_info_document(user_id)
        if personal_info_doc:
            return personal_info_doc.file_content.decode('utf-8')
        return None
    
    def get_documents_status(self, user_id: int = None) -> Dict[str, Any]:
        """Get status of user's documents"""
        try:
            resume_doc = self.get_user_resume(user_id)
            personal_info_doc = self.get_personal_info_document(user_id)
            
            return {
                "resume": {
                    "filename": resume_doc.filename if resume_doc else None,
                    "file_size": resume_doc.file_size if resume_doc else None,
                    "processing_status": resume_doc.processing_status if resume_doc else None
                } if resume_doc else None,
                "personal_info": {
                    "filename": personal_info_doc.filename if personal_info_doc else None,
                    "file_size": personal_info_doc.file_size if personal_info_doc else None,
                    "processing_status": personal_info_doc.processing_status if personal_info_doc else None
                } if personal_info_doc else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting documents status: {e}")
            raise
    
    def get_user_resume_documents(self, user_id: int = None) -> List[ResumeDocument]:
        """⚡ OPTIMIZED: Get user resume documents"""
        try:
            with self.get_session() as session:
                query = session.query(ResumeDocument)
                
                if user_id:
                    query = query.filter(ResumeDocument.user_id == user_id)
                
                documents = query.order_by(desc(ResumeDocument.id)).all()
                return documents
                
        except Exception as e:
            logger.error(f"❌ Error getting user resume documents: {e}")
            raise
    
    def get_user_personal_info_documents(self, user_id: int = None) -> List[PersonalInfoDocument]:
        """⚡ OPTIMIZED: Get user personal info documents"""
        try:
            with self.get_session() as session:
                query = session.query(PersonalInfoDocument)
                
                if user_id:
                    query = query.filter(PersonalInfoDocument.user_id == user_id)
                
                documents = query.order_by(desc(PersonalInfoDocument.id)).all()
                return documents
                
        except Exception as e:
            logger.error(f"❌ Error getting user personal info documents: {e}")
            raise
    
    def cleanup_temp_file(self, temp_path: str):
        """Clean up temporary file"""
        try:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()
                logger.info(f"🧹 Cleaned up temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up temporary file {temp_path}: {e}") 
    
    def _get_file_extension(self, content_type: str) -> str:
        """Get file extension from content type"""
        content_type_map = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'text/plain': '.txt'
        }
        return content_type_map.get(content_type, '')