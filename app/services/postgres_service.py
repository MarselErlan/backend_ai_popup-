"""
PostgreSQL Service - Handles interactions with PostgreSQL for form URL tracking
Replaces Supabase with local PostgreSQL database
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from loguru import logger

from db.models import FormDb, SupabaseResult
from db.schemas import FormResponse

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup")

# Create SQLAlchemy base
Base = declarative_base()

class FormTable(Base):
    """PostgreSQL table for form URL tracking"""
    __tablename__ = "forms"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    analyzed = Column(Boolean, default=False)
    status = Column(String, default="not_applied")
    applied_counter = Column(Integer, default=0)
    applied_date = Column(String, nullable=True)  # ISO date string

class PostgresService:
    """PostgreSQL service for form URL tracking"""
    
    def __init__(self):
        """Initialize the PostgreSQL connection"""
        try:
            self.engine = create_engine(DATABASE_URL)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.SessionLocal = SessionLocal
            
            # Create tables if they don't exist
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ PostgreSQL connection established and tables created")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
        
    def add_form_url(self, url: str) -> Dict[str, Any]:
        """
        Add a form URL to the PostgreSQL 'forms' table
        
        Args:
            url: The form URL to save
            
        Returns:
            Dict: Result of the operation
        """
        session = self.get_session()
        try:
            # Check if URL already exists
            existing = session.query(FormTable).filter(FormTable.url == url).first()
            
            if existing:
                # URL already exists
                result = SupabaseResult(
                    status="success",
                    message="URL already exists in database",
                    id=str(existing.id)
                )
                return result.model_dump()
            
            # Insert new URL with default status
            new_form = FormTable(
                url=url,
                created_at=datetime.now(),
                status="not_applied",
                applied_counter=0,
                applied_date=None
            )
            
            session.add(new_form)
            session.commit()
            session.refresh(new_form)
            
            return SupabaseResult(
                status="success",
                message="URL added successfully",
                id=str(new_form.id)
            ).model_dump()
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding form URL: {e}")
            return SupabaseResult(
                status="error", 
                message=str(e)
            ).model_dump()
        finally:
            session.close()
    
    def get_all_forms(self) -> List[FormResponse]:
        """
        Get all form URLs from the database
        
        Returns:
            List[FormResponse]: List of form entries
        """
        session = self.get_session()
        try:
            forms_data = session.query(FormTable).order_by(FormTable.created_at.desc()).all()
            forms = []
            
            for form_data in forms_data:
                form = FormResponse(
                    id=str(form_data.id),
                    url=form_data.url,
                    created_at=form_data.created_at.isoformat(),
                    analyzed=form_data.analyzed,
                    status=form_data.status,
                    applied_counter=form_data.applied_counter,
                    applied_date=form_data.applied_date
                )
                forms.append(form)
                
            return forms
        except Exception as e:
            logger.error(f"Error getting form URLs: {e}")
            return []
        finally:
            session.close()
    
    def update_form_analysis_status(self, url: str, analyzed: bool = True) -> Dict[str, Any]:
        """
        Update the analyzed status of a form
        
        Args:
            url: The form URL to update
            analyzed: The analysis status
            
        Returns:
            Dict: Result of the operation
        """
        session = self.get_session()
        try:
            form = session.query(FormTable).filter(FormTable.url == url).first()
            
            if form:
                form.analyzed = analyzed
                session.commit()
                
                return SupabaseResult(
                    status="success", 
                    message="Form status updated"
                ).model_dump()
            else:
                return SupabaseResult(
                    status="error", 
                    message="Form not found"
                ).model_dump()
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating form status: {e}")
            return SupabaseResult(
                status="error", 
                message=str(e)
            ).model_dump()
        finally:
            session.close()

    def mark_form_applied(self, url: str) -> Dict[str, Any]:
        """
        Mark a form as applied and update its application counter.
        If it's a new day, reset the counter to 1, otherwise increment it.
        
        Args:
            url: The form URL to update
            
        Returns:
            Dict: Result of the operation
        """
        session = self.get_session()
        try:
            today = date.today().isoformat()
            
            # Get current form data
            form = session.query(FormTable).filter(FormTable.url == url).first()
            
            if not form:
                return SupabaseResult(
                    status="error",
                    message="Form URL not found"
                ).model_dump()
                
            # Determine new counter value
            if form.applied_date == today:
                new_counter = form.applied_counter + 1
            else:
                new_counter = 1
                
            # Update form status and counter
            form.status = "applied"
            form.applied_counter = new_counter
            form.applied_date = today
            
            session.commit()
            
            return SupabaseResult(
                status="success",
                message=f"Form marked as applied (attempt #{new_counter} today)"
            ).model_dump()
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error marking form as applied: {e}")
            return SupabaseResult(
                status="error",
                message=str(e)
            ).model_dump()
        finally:
            session.close() 