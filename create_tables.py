#!/usr/bin/env python3
"""
Script to create database tables and test connection
"""

import os
from app.services.document_service import DocumentService
from loguru import logger

def main():
    try:
        # Get database URL
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            logger.error("❌ DATABASE_URL environment variable not set")
            return False
        
        logger.info(f"🔗 Connecting to database: {database_url[:50]}...")
        
        # Initialize document service (this should create tables)
        document_service = DocumentService(database_url)
        
        # Test the connection by getting stats
        stats = document_service.get_document_stats()
        logger.info(f"✅ Database connection successful!")
        logger.info(f"📊 Current stats: {stats}")
        
        # Test creating a session
        with document_service.get_session() as session:
            logger.info("✅ Database session created successfully!")
        
        logger.info("🎉 Tables should now be created in your Supabase database!")
        logger.info("💡 Check your Supabase dashboard for these tables:")
        logger.info("   - resume_documents")
        logger.info("   - personal_info_documents") 
        logger.info("   - document_processing_logs")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 