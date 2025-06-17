"""
Document service for database operations
"""
import io
import tempfile
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, desc
from loguru import logger

from app.models.document_models import Base, ResumeDocument, PersonalInfoDocument, DocumentProcessingLog


class DocumentService:
    """Service for managing documents in database"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
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
        user_id: str = None
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
    
    def get_user_resume(self, user_id: str = None) -> Optional[ResumeDocument]:
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
    
    def save_personal_info_document(self, filename: str, file_content: bytes, 
                                  content_type: str, user_id: str = None) -> int:
        """Save personal info document to database"""
        try:
            with self.get_session() as session:
                # Deactivate previous personal info for this user
                if user_id:
                    session.query(PersonalInfoDocument).filter(
                        PersonalInfoDocument.user_id == user_id,
                        PersonalInfoDocument.is_active == True
                    ).update({"is_active": False})
                
                personal_info_doc = PersonalInfoDocument(
                    user_id=user_id,
                    filename=filename,
                    file_content=file_content,
                    content_type=content_type,
                    file_size=len(file_content),
                    is_active=True,
                    processing_status='pending'
                )
                
                session.add(personal_info_doc)
                session.commit()
                session.refresh(personal_info_doc)
                
                logger.info(f"💾 Saved personal info document: {filename} (ID: {personal_info_doc.id})")
                return personal_info_doc.id
                
        except Exception as e:
            logger.error(f"❌ Error saving personal info document: {e}")
            raise
    
    def get_active_personal_info_document(self, user_id: str = None) -> Optional[PersonalInfoDocument]:
        """Get the active personal info document for a user"""
        try:
            with self.get_session() as session:
                query = session.query(PersonalInfoDocument).filter(
                    PersonalInfoDocument.is_active == True
                )
                
                if user_id:
                    query = query.filter(PersonalInfoDocument.user_id == user_id)
                
                personal_info_doc = query.order_by(desc(PersonalInfoDocument.created_at)).first()
                
                if personal_info_doc:
                    logger.info(f"📄 Retrieved active personal info: {personal_info_doc.filename} (ID: {personal_info_doc.id})")
                else:
                    logger.warning("⚠️ No active personal info document found")
                
                return personal_info_doc
                
        except Exception as e:
            logger.error(f"❌ Error retrieving personal info document: {e}")
            raise
    
    def get_personal_info_as_temp_file(self, user_id: str = None) -> Optional[Tuple[str, str]]:
        """Get personal info as temporary file for processing"""
        try:
            personal_info_doc = self.get_active_personal_info_document(user_id)
            if not personal_info_doc:
                return None
            
            # Create temporary file
            file_extension = self._get_file_extension(personal_info_doc.content_type)
            with tempfile.NamedTemporaryFile(mode='wb', suffix=file_extension, delete=False) as temp_file:
                temp_file.write(personal_info_doc.file_content)
                temp_path = temp_file.name
            
            logger.info(f"📂 Created temporary file for personal info: {temp_path}")
            return temp_path, personal_info_doc.filename
            
        except Exception as e:
            logger.error(f"❌ Error creating temporary personal info file: {e}")
            raise
    
    def update_personal_info_processing_status(self, document_id: int, status: str, 
                                             processed_at: datetime = None):
        """Update personal info processing status"""
        try:
            with self.get_session() as session:
                session.query(PersonalInfoDocument).filter(
                    PersonalInfoDocument.id == document_id
                ).update({
                    "processing_status": status,
                    "last_processed_at": processed_at or datetime.now()
                })
                session.commit()
                
                logger.info(f"✅ Updated personal info processing status: {status} (ID: {document_id})")
                
        except Exception as e:
            logger.error(f"❌ Error updating personal info processing status: {e}")
            raise
    
    # ============================================================================
    # PROCESSING LOG OPERATIONS
    # ============================================================================
    
    def log_processing_start(self, document_type: str, document_id: int, user_id: str = None) -> int:
        """Log the start of document processing"""
        try:
            with self.get_session() as session:
                log = DocumentProcessingLog(
                    document_type=document_type,
                    document_id=document_id,
                    user_id=user_id,
                    status="started",
                    started_at=datetime.now()
                )
                session.add(log)
                session.commit()
                session.refresh(log)
                return log.id
                
        except Exception as e:
            logger.error(f"❌ Error logging processing start: {e}")
            raise
    
    def log_processing_complete(self, log_id: int, processing_time: int = None,
                              total_chunks: int = None, embedding_dimension: int = None,
                              model_used: str = None):
        """Log successful completion of document processing"""
        try:
            with self.get_session() as session:
                session.query(DocumentProcessingLog).filter(
                    DocumentProcessingLog.id == log_id
                ).update({
                    "status": "completed",
                    "completed_at": datetime.now(),
                    "processing_time": processing_time,
                    "total_chunks": total_chunks,
                    "embedding_dimension": embedding_dimension,
                    "model_used": model_used
                })
                session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error logging processing completion: {e}")
            raise
    
    def log_processing_error(self, log_id: int, error_message: str):
        """Log processing error"""
        try:
            with self.get_session() as session:
                session.query(DocumentProcessingLog).filter(
                    DocumentProcessingLog.id == log_id
                ).update({
                    "status": "failed",
                    "completed_at": datetime.now(),
                    "error_message": error_message
                })
                session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error logging processing error: {e}")
            raise
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get document statistics"""
        try:
            with self.get_session() as session:
                resume_count = session.query(ResumeDocument).filter(
                    ResumeDocument.is_active == True
                ).count()
                
                personal_info_count = session.query(PersonalInfoDocument).filter(
                    PersonalInfoDocument.is_active == True
                ).count()
                
                processing_logs_count = session.query(DocumentProcessingLog).count()
                
                stats = {
                    "active_resume_documents": resume_count,
                    "active_personal_info_documents": personal_info_count,
                    "total_processing_logs": processing_logs_count,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"📊 Document stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"❌ Error getting document stats: {e}")
            raise
    
    def get_documents_status(self, user_id: str = None) -> Dict[str, Any]:
        """Get status of user's documents"""
        try:
            resume_doc = self.get_user_resume(user_id)
            personal_info_doc = self.get_active_personal_info_document(user_id)
            
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
    
    def get_user_resume_documents(self, user_id: str = None) -> List[ResumeDocument]:
        """⚡ OPTIMIZED: Get user resume documents"""
        try:
            with self.get_session() as session:
                query = session.query(ResumeDocument).filter(
                    ResumeDocument.is_active == True
                )
                
                if user_id:
                    query = query.filter(ResumeDocument.user_id == user_id)
                
                documents = query.order_by(desc(ResumeDocument.created_at)).all()
                return documents
                
        except Exception as e:
            logger.error(f"❌ Error getting user resume documents: {e}")
            raise
    
    def get_user_personal_info_documents(self, user_id: str = None) -> List[PersonalInfoDocument]:
        """⚡ OPTIMIZED: Get user personal info documents"""
        try:
            with self.get_session() as session:
                query = session.query(PersonalInfoDocument).filter(
                    PersonalInfoDocument.is_active == True
                )
                
                if user_id:
                    query = query.filter(PersonalInfoDocument.user_id == user_id)
                
                documents = query.order_by(desc(PersonalInfoDocument.created_at)).all()
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