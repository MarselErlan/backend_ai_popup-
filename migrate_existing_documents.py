#!/usr/bin/env python3
"""
Migration script to re-embed existing documents into Pinecone
Run this script to process existing documents in the database
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger

# Import models and services
from models import ResumeDocument, PersonalInfoDocument
from app.services.embedding_service import EmbeddingService
from app.utils.text_extractor import extract_text_from_file

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqTpIKa@interchange.proxy.rlwy.net:30153/railway")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class DocumentMigrator:
    """Migrates existing documents to Pinecone vector store"""
    
    def __init__(self):
        # Initialize database
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            redis_url=REDIS_URL,
            openai_api_key=OPENAI_API_KEY
        )
        
        logger.info("🚀 Document migrator initialized")
        logger.info(f"   Database: {DATABASE_URL}")
        logger.info(f"   Vector store: {'Pinecone' if hasattr(self.embedding_service.vector_store, 'index') else 'Redis'}")
    
    def get_all_resume_documents(self) -> List[ResumeDocument]:
        """Get all resume documents from database"""
        with self.SessionLocal() as session:
            return session.query(ResumeDocument).filter(ResumeDocument.is_active == True).all()
    
    def get_all_personal_info_documents(self) -> List[PersonalInfoDocument]:
        """Get all personal info documents from database"""
        with self.SessionLocal() as session:
            return session.query(PersonalInfoDocument).filter(PersonalInfoDocument.is_active == True).all()
    
    async def migrate_resume_document(self, document: ResumeDocument) -> bool:
        """Migrate a single resume document"""
        try:
            logger.info(f"📄 Processing resume document {document.id} for user {document.user_id}")
            
            # Extract text from document
            text = await extract_text_from_file(document.file_content, document.content_type)
            
            # Process document into vector store
            self.embedding_service.process_document(
                document_id=f"resume_{document.id}",
                user_id=document.user_id,
                content=text,
                reprocess=True
            )
            
            logger.info(f"✅ Successfully migrated resume document {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate resume document {document.id}: {e}")
            return False
    
    async def migrate_personal_info_document(self, document: PersonalInfoDocument) -> bool:
        """Migrate a single personal info document"""
        try:
            logger.info(f"📄 Processing personal info document {document.id} for user {document.user_id}")
            
            # Extract text from document
            text = await extract_text_from_file(document.file_content, document.content_type)
            
            # Process document into vector store
            self.embedding_service.process_document(
                document_id=f"personal_info_{document.id}",
                user_id=document.user_id,
                content=text,
                reprocess=True
            )
            
            logger.info(f"✅ Successfully migrated personal info document {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate personal info document {document.id}: {e}")
            return False
    
    async def migrate_all_documents(self) -> Dict[str, Any]:
        """Migrate all documents to vector store"""
        start_time = time.time()
        
        logger.info("🚀 Starting document migration to vector store")
        
        # Get all documents
        resume_documents = self.get_all_resume_documents()
        personal_info_documents = self.get_all_personal_info_documents()
        
        logger.info(f"📊 Found {len(resume_documents)} resume documents and {len(personal_info_documents)} personal info documents")
        
        # Migration statistics
        stats = {
            "resume_documents": {
                "total": len(resume_documents),
                "successful": 0,
                "failed": 0
            },
            "personal_info_documents": {
                "total": len(personal_info_documents),
                "successful": 0,
                "failed": 0
            }
        }
        
        # Migrate resume documents
        logger.info("📄 Migrating resume documents...")
        for document in resume_documents:
            success = await self.migrate_resume_document(document)
            if success:
                stats["resume_documents"]["successful"] += 1
            else:
                stats["resume_documents"]["failed"] += 1
            
            # Small delay between documents
            time.sleep(0.1)
        
        # Migrate personal info documents
        logger.info("📄 Migrating personal info documents...")
        for document in personal_info_documents:
            success = await self.migrate_personal_info_document(document)
            if success:
                stats["personal_info_documents"]["successful"] += 1
            else:
                stats["personal_info_documents"]["failed"] += 1
            
            # Small delay between documents
            time.sleep(0.1)
        
        # Calculate totals
        total_documents = stats["resume_documents"]["total"] + stats["personal_info_documents"]["total"]
        total_successful = stats["resume_documents"]["successful"] + stats["personal_info_documents"]["successful"]
        total_failed = stats["resume_documents"]["failed"] + stats["personal_info_documents"]["failed"]
        
        migration_time = time.time() - start_time
        
        # Log summary
        logger.info("📊 Migration Summary")
        logger.info("=" * 50)
        logger.info(f"Resume Documents: {stats['resume_documents']['successful']}/{stats['resume_documents']['total']} successful")
        logger.info(f"Personal Info Documents: {stats['personal_info_documents']['successful']}/{stats['personal_info_documents']['total']} successful")
        logger.info(f"Total: {total_successful}/{total_documents} documents migrated successfully")
        logger.info(f"Migration time: {migration_time:.2f} seconds")
        logger.info("=" * 50)
        
        if total_failed == 0:
            logger.info("🎉 All documents migrated successfully!")
        else:
            logger.warning(f"⚠️ {total_failed} documents failed to migrate")
        
        return {
            "stats": stats,
            "total_documents": total_documents,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "migration_time": migration_time
        }

async def main():
    """Main migration function"""
    try:
        # Check environment variables
        if not OPENAI_API_KEY:
            logger.error("❌ OPENAI_API_KEY environment variable is required")
            sys.exit(1)
        
        # Initialize migrator
        migrator = DocumentMigrator()
        
        # Run migration
        results = await migrator.migrate_all_documents()
        
        # Exit with appropriate code
        if results["total_failed"] == 0:
            logger.info("✅ Migration completed successfully")
            sys.exit(0)
        else:
            logger.error(f"❌ Migration completed with {results['total_failed']} failures")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 