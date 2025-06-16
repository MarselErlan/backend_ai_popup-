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
    
    def save_resume_document(self, filename: str, file_content: bytes, 
                           content_type: str, user_id: str = None) -> int:
        """Save resume document to database"""
        try:
            with self.get_session() as session:
                # Deactivate previous resume for this user
                if user_id:
                    session.query(ResumeDocument).filter(
                        ResumeDocument.user_id == user_id,
                        ResumeDocument.is_active == True
                    ).update({"is_active": False})
                
                resume_doc = ResumeDocument(
                    user_id=user_id,
                    filename=filename,
                    file_content=file_content,
                    content_type=content_type,
                    file_size=len(file_content),
                    is_active=True,
                    processing_status='pending'
                )
                
                session.add(resume_doc)
                session.commit()
                session.refresh(resume_doc)
                
                logger.info(f"💾 Saved resume document: {filename} (ID: {resume_doc.id})")
                return resume_doc.id
                
        except Exception as e:
            logger.error(f"❌ Error saving resume document: {e}")
            raise
    
    def get_active_resume_document(self, user_id: str = None) -> Optional[ResumeDocument]:
        """Get the active resume document for a user"""
        try:
            with self.get_session() as session:
                query = session.query(ResumeDocument).filter(
                    ResumeDocument.is_active == True
                )
                
                if user_id:
                    query = query.filter(ResumeDocument.user_id == user_id)
                
                resume_doc = query.order_by(desc(ResumeDocument.created_at)).first()
                
                if resume_doc:
                    logger.info(f"📄 Retrieved active resume: {resume_doc.filename} (ID: {resume_doc.id})")
                else:
                    logger.warning("⚠️ No active resume document found")
                
                return resume_doc
                
        except Exception as e:
            logger.error(f"❌ Error retrieving resume document: {e}")
            raise
    
    def get_resume_as_temp_file(self, user_id: str = None) -> Optional[Tuple[str, str]]:
        """Get resume as temporary file for processing"""
        try:
            resume_doc = self.get_active_resume_document(user_id)
            if not resume_doc:
                return None
            
            # Create temporary file
            suffix = Path(resume_doc.filename).suffix
            with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as temp_file:
                temp_file.write(resume_doc.file_content)
                temp_path = temp_file.name
            
            logger.info(f"📂 Created temporary file for resume: {temp_path}")
            return temp_path, resume_doc.filename
            
        except Exception as e:
            logger.error(f"❌ Error creating temporary resume file: {e}")
            raise
    
    def update_resume_processing_status(self, document_id: int, status: str, 
                                      processed_at: datetime = None):
        """Update resume processing status"""
        try:
            with self.get_session() as session:
                session.query(ResumeDocument).filter(
                    ResumeDocument.id == document_id
                ).update({
                    "processing_status": status,
                    "last_processed_at": processed_at or datetime.now()
                })
                session.commit()
                
                logger.info(f"✅ Updated resume processing status: {status} (ID: {document_id})")
                
        except Exception as e:
            logger.error(f"❌ Error updating resume processing status: {e}")
            raise
    
    # ============================================================================
    # PERSONAL INFO DOCUMENT OPERATIONS
    # ============================================================================
    
    def save_personal_info_document(self, filename: str, content: str, 
                                  user_id: str = None) -> int:
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
                    content=content,
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
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(personal_info_doc.content)
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
    
    def log_processing_start(self, document_type: str, document_id: int, 
                           user_id: str = None) -> int:
        """Log start of document processing"""
        try:
            with self.get_session() as session:
                processing_log = DocumentProcessingLog(
                    document_type=document_type,
                    document_id=document_id,
                    user_id=user_id,
                    processing_status='started'
                )
                
                session.add(processing_log)
                session.commit()
                session.refresh(processing_log)
                
                logger.info(f"📝 Started processing log for {document_type} (Log ID: {processing_log.id})")
                return processing_log.id
                
        except Exception as e:
            logger.error(f"❌ Error creating processing log: {e}")
            raise
    
    def log_processing_complete(self, log_id: int, processing_time: int = None,
                              total_chunks: int = None, embedding_dimension: int = None,
                              model_used: str = None):
        """Log completion of document processing"""
        try:
            with self.get_session() as session:
                session.query(DocumentProcessingLog).filter(
                    DocumentProcessingLog.id == log_id
                ).update({
                    "processing_status": "completed",
                    "completed_at": datetime.now(),
                    "processing_time_seconds": processing_time,
                    "total_chunks": total_chunks,
                    "embedding_dimension": embedding_dimension,
                    "model_used": model_used
                })
                session.commit()
                
                logger.info(f"✅ Completed processing log (Log ID: {log_id})")
                
        except Exception as e:
            logger.error(f"❌ Error updating processing log: {e}")
            raise
    
    def log_processing_error(self, log_id: int, error_message: str):
        """Log processing error"""
        try:
            with self.get_session() as session:
                session.query(DocumentProcessingLog).filter(
                    DocumentProcessingLog.id == log_id
                ).update({
                    "processing_status": "failed",
                    "completed_at": datetime.now(),
                    "error_message": error_message
                })
                session.commit()
                
                logger.error(f"❌ Failed processing log (Log ID: {log_id}): {error_message}")
                
        except Exception as e:
            logger.error(f"❌ Error updating processing log with error: {e}")
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
        """⚡ OPTIMIZED: Get comprehensive documents status for a user"""
        try:
            with self.get_session() as session:
                # Get resume documents
                resume_query = session.query(ResumeDocument)
                if user_id:
                    resume_query = resume_query.filter(ResumeDocument.user_id == user_id)
                resume_docs = resume_query.filter(ResumeDocument.is_active == True).all()
                
                # Get personal info documents
                personal_query = session.query(PersonalInfoDocument)
                if user_id:
                    personal_query = personal_query.filter(PersonalInfoDocument.user_id == user_id)
                personal_docs = personal_query.filter(PersonalInfoDocument.is_active == True).all()
                
                # Get recent processing logs
                log_query = session.query(DocumentProcessingLog)
                if user_id:
                    log_query = log_query.filter(DocumentProcessingLog.user_id == user_id)
                recent_logs = log_query.order_by(desc(DocumentProcessingLog.started_at)).limit(10).all()
                
                status_data = {
                    "resume_documents": {
                        "count": len(resume_docs),
                        "documents": [
                            {
                                "id": doc.id,
                                "filename": doc.filename,
                                "file_size": doc.file_size,
                                "processing_status": doc.processing_status,
                                "uploaded_at": doc.created_at.isoformat() if doc.created_at else None,
                                "last_processed": doc.last_processed_at.isoformat() if doc.last_processed_at else None
                            } for doc in resume_docs
                        ]
                    },
                    "personal_info_documents": {
                        "count": len(personal_docs),
                        "documents": [
                            {
                                "id": doc.id,
                                "title": doc.filename,
                                "content_length": len(doc.content) if doc.content else 0,
                                "processing_status": doc.processing_status,
                                "uploaded_at": doc.created_at.isoformat() if doc.created_at else None,
                                "last_processed": doc.last_processed_at.isoformat() if doc.last_processed_at else None
                            } for doc in personal_docs
                        ]
                    },
                    "recent_processing_logs": [
                        {
                            "id": log.id,
                            "document_type": log.document_type,
                            "status": log.processing_status,
                            "started_at": log.started_at.isoformat() if log.started_at else None,
                            "completed_at": log.completed_at.isoformat() if log.completed_at else None,
                            "processing_time": log.processing_time_seconds,
                            "total_chunks": log.total_chunks,
                            "model_used": log.model_used
                        } for log in recent_logs
                    ],
                    "summary": {
                        "total_active_documents": len(resume_docs) + len(personal_docs),
                        "resume_status": "available" if resume_docs else "missing",
                        "personal_info_status": "available" if personal_docs else "missing",
                        "last_activity": max([
                            doc.last_processed_at for doc in (resume_docs + personal_docs) 
                            if doc.last_processed_at
                        ], default=None)
                    }
                }
                
                if status_data["summary"]["last_activity"]:
                    status_data["summary"]["last_activity"] = status_data["summary"]["last_activity"].isoformat()
                
                logger.info(f"📋 Retrieved document status for user: {user_id or 'default'}")
                return status_data
                
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