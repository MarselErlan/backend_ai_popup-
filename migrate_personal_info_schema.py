#!/usr/bin/env python3
"""
Database Migration Script: Update PersonalInfoDocument Schema
From: content (Text) to file_content (LargeBinary) + content_type + file_size
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text, Column, Integer, String, LargeBinary, Boolean, DateTime
from sqlalchemy.orm import sessionmaker
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smart_form_filler")

def migrate_personal_info_schema():
    """Migrate PersonalInfoDocument schema"""
    try:
        logger.info("🚀 Starting PersonalInfoDocument schema migration...")
        
        # Create engine and session
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Step 1: Check if migration is needed
        logger.info("📋 Checking current schema...")
        
        # Check if new columns exist
        check_columns_sql = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'personal_info_documents' 
        AND column_name IN ('file_content', 'content_type', 'file_size');
        """
        result = session.execute(text(check_columns_sql))
        existing_columns = {row[0]: row[1] for row in result.fetchall()}
        
        if len(existing_columns) == 3:
            logger.info("✅ Schema already migrated. Skipping migration.")
            return True
        
        # Step 2: Create backup table
        logger.info("💾 Creating backup table...")
        backup_sql = """
        CREATE TABLE personal_info_documents_backup AS 
        SELECT * FROM personal_info_documents;
        """
        session.execute(text(backup_sql))
        session.commit()
        logger.info("✅ Backup table created: personal_info_documents_backup")
        
        # Step 3: Add new columns (keeping old content column temporarily)
        logger.info("🔧 Adding new columns...")
        
        if 'file_content' not in existing_columns:
            session.execute(text("ALTER TABLE personal_info_documents ADD COLUMN file_content BYTEA;"))
            logger.info("   ✅ Added file_content column")
        
        if 'content_type' not in existing_columns:
            session.execute(text("ALTER TABLE personal_info_documents ADD COLUMN content_type VARCHAR(100);"))
            logger.info("   ✅ Added content_type column")
        
        if 'file_size' not in existing_columns:
            session.execute(text("ALTER TABLE personal_info_documents ADD COLUMN file_size INTEGER;"))
            logger.info("   ✅ Added file_size column")
        
        session.commit()
        
        # Step 4: Migrate existing data
        logger.info("📦 Migrating existing data...")
        
        # Get all existing records with content
        migrate_data_sql = """
        UPDATE personal_info_documents 
        SET 
            file_content = content::bytea,
            content_type = 'text/plain',
            file_size = length(content::bytea)
        WHERE content IS NOT NULL 
        AND file_content IS NULL;
        """
        result = session.execute(text(migrate_data_sql))
        migrated_count = result.rowcount
        session.commit()
        
        logger.info(f"   ✅ Migrated {migrated_count} records")
        
        # Step 5: Make new columns NOT NULL (after data migration)
        logger.info("🔒 Setting constraints on new columns...")
        
        session.execute(text("ALTER TABLE personal_info_documents ALTER COLUMN file_content SET NOT NULL;"))
        session.execute(text("ALTER TABLE personal_info_documents ALTER COLUMN content_type SET NOT NULL;"))
        session.execute(text("ALTER TABLE personal_info_documents ALTER COLUMN file_size SET NOT NULL;"))
        session.commit()
        
        logger.info("   ✅ Set NOT NULL constraints")
        
        # Step 6: Drop old content column
        logger.info("🗑️ Dropping old content column...")
        session.execute(text("ALTER TABLE personal_info_documents DROP COLUMN content;"))
        session.commit()
        
        logger.info("   ✅ Dropped old content column")
        
        # Step 7: Verify migration
        logger.info("🔍 Verifying migration...")
        
        verify_sql = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(file_content) as records_with_content,
            AVG(file_size) as avg_file_size
        FROM personal_info_documents;
        """
        result = session.execute(text(verify_sql))
        stats = result.fetchone()
        
        logger.info(f"   📊 Total records: {stats[0]}")
        logger.info(f"   📊 Records with file content: {stats[1]}")
        logger.info(f"   📊 Average file size: {stats[2]:.2f} bytes" if stats[2] else "   📊 Average file size: 0 bytes")
        
        session.close()
        
        logger.info("🎉 PersonalInfoDocument schema migration completed successfully!")
        logger.info("💡 Backup table 'personal_info_documents_backup' created for rollback if needed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        
        # Rollback on error
        try:
            session.rollback()
            session.close()
        except:
            pass
        
        return False

def rollback_migration():
    """Rollback the migration using backup table"""
    try:
        logger.info("🔄 Rolling back PersonalInfoDocument schema migration...")
        
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check if backup exists
        check_backup_sql = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'personal_info_documents_backup'
        );
        """
        result = session.execute(text(check_backup_sql))
        backup_exists = result.fetchone()[0]
        
        if not backup_exists:
            logger.error("❌ Backup table not found. Cannot rollback.")
            return False
        
        # Drop current table and restore from backup
        session.execute(text("DROP TABLE personal_info_documents;"))
        session.execute(text("ALTER TABLE personal_info_documents_backup RENAME TO personal_info_documents;"))
        session.commit()
        session.close()
        
        logger.info("✅ Migration rolled back successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PersonalInfoDocument Schema Migration")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    args = parser.parse_args()
    
    if args.rollback:
        success = rollback_migration()
    else:
        success = migrate_personal_info_schema()
    
    sys.exit(0 if success else 1) 