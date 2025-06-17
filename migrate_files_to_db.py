#!/usr/bin/env python3
"""
Migration script to move existing files from docs/ folder to database
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from app.services.document_service import DocumentService
from loguru import logger


def migrate_resume_files(document_service: DocumentService, user_id: str = "default"):
    """Migrate resume files from docs/resume/ to database"""
    try:
        resume_folder = Path("docs/resume")
        
        if not resume_folder.exists():
            logger.warning(f"📁 Resume folder not found: {resume_folder}")
            return False
        
        # Find resume files
        resume_files = list(resume_folder.glob("*.docx")) + list(resume_folder.glob("*.doc"))
        
        if not resume_files:
            logger.warning("📄 No resume files found in docs/resume/")
            return False
        
        # Take the most recent resume file
        latest_resume = max(resume_files, key=lambda f: f.stat().st_mtime)
        
        logger.info(f"📄 Migrating resume file: {latest_resume.name}")
        
        # Read file content
        with open(latest_resume, 'rb') as f:
            file_content = f.read()
        
        # Determine content type
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if latest_resume.suffix == ".docx" else "application/msword"
        
        # Save to database
        document_id = document_service.save_resume_document(
            filename=latest_resume.name,
            file_content=file_content,
            content_type=content_type,
            user_id=user_id
        )
        
        logger.info(f"✅ Resume migrated successfully! Document ID: {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Resume migration failed: {e}")
        return False


def migrate_personal_info_files(document_service: DocumentService, user_id: str = "default"):
    """Migrate personal info files from docs/info/ to database"""
    try:
        info_folder = Path("docs/info")
        
        if not info_folder.exists():
            logger.warning(f"📁 Info folder not found: {info_folder}")
            return False
        
        # Find text files
        info_files = list(info_folder.glob("*.txt"))
        
        if not info_files:
            logger.warning("📄 No personal info files found in docs/info/")
            return False
        
        # Take the most recent info file
        latest_info = max(info_files, key=lambda f: f.stat().st_mtime)
        
        logger.info(f"📄 Migrating personal info file: {latest_info.name}")
        
        # Read file content
        with open(latest_info, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            logger.warning("📄 Personal info file is empty, skipping migration")
            return False
        
        # Save to database
        document_id = document_service.save_personal_info_document(
            filename=latest_info.name,
            content=content,
            user_id=user_id
        )
        
        logger.info(f"✅ Personal info migrated successfully! Document ID: {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Personal info migration failed: {e}")
        return False


def main():
    """Run the migration"""
    try:
        logger.info("🚀 Starting file to database migration...")
        
        # Get database URL
        database_url = os.getenv("POSTGRES_DB_URL", "postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup")
        
        # Initialize document service
        document_service = DocumentService(database_url)
        
        # User ID for migration (you can customize this)
        user_id = os.getenv("MIGRATION_USER_ID", "default")
        
        logger.info(f"📊 Using database: {database_url}")
        logger.info(f"👤 Using user ID: {user_id}")
        
        # Migrate files
        resume_success = migrate_resume_files(document_service, user_id)
        personal_info_success = migrate_personal_info_files(document_service, user_id)
        
        # Summary
        logger.info("📝 Migration Summary:")
        logger.info(f"   Resume files: {'✅ Success' if resume_success else '❌ Failed/Skipped'}")
        logger.info(f"   Personal info files: {'✅ Success' if personal_info_success else '❌ Failed/Skipped'}")
        
        if resume_success or personal_info_success:
            logger.info("🎉 Migration completed! Files are now stored in database.")
            logger.info("💡 You can now use the new database-based endpoints:")
            logger.info("   - POST /api/reembed-resume-db")
            logger.info("   - POST /api/reembed-personal-info-db")
            logger.info("   - GET /api/documents/status")
        else:
            logger.warning("⚠️ No files were migrated. Check if files exist in docs/ folders.")
        
        # Get document stats
        stats = document_service.get_document_stats()
        logger.info(f"📊 Current database stats: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Migration script failed: {e}")
        return False


if __name__ == "__main__":
    main() 