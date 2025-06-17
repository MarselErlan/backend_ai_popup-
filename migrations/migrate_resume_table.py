"""
Migration script to restore file content columns in resume_documents table
"""
import os
from sqlalchemy import create_engine, text, inspect
from loguru import logger

# Get database URL from environment or use default
DATABASE_URL = os.getenv("POSTGRES_DB_URL", "postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup")

def table_exists(conn, table_name):
    """Check if a table exists"""
    return inspect(conn).has_table(table_name)

def migrate():
    """Migrate resume_documents table to include file content"""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Connect and execute migration
        with engine.connect() as conn:
            # Start transaction
            with conn.begin():
                logger.info("🔄 Starting resume_documents table migration...")
                
                # Check if table exists
                if table_exists(conn, "resume_documents"):
                    # Backup existing data
                    logger.info("📦 Backing up existing data...")
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS resume_documents_backup AS 
                        SELECT * FROM resume_documents;
                    """))
                    
                    # Drop existing table
                    logger.info("🗑️ Dropping existing table...")
                    conn.execute(text("DROP TABLE resume_documents;"))
                
                # Create new table with file content columns
                logger.info("📝 Creating new table structure...")
                conn.execute(text("""
                    CREATE TABLE resume_documents (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100),
                        filename VARCHAR(255),
                        file_content BYTEA,
                        content_type VARCHAR(100),
                        file_size INTEGER,
                        processing_status VARCHAR(50) DEFAULT 'pending'
                    );
                """))
                
                # Create index on user_id
                logger.info("📊 Creating index on user_id...")
                conn.execute(text("""
                    CREATE INDEX idx_resume_documents_user_id 
                    ON resume_documents(user_id);
                """))
                
                # Restore data if backup exists
                if table_exists(conn, "resume_documents_backup"):
                    logger.info("♻️ Restoring data from backup...")
                    conn.execute(text("""
                        INSERT INTO resume_documents (id, user_id, filename, processing_status)
                        SELECT id, user_id, filename, processing_status
                        FROM resume_documents_backup;
                    """))
                    
                    # Reset sequence
                    logger.info("🔄 Resetting ID sequence...")
                    conn.execute(text("""
                        SELECT setval(
                            'resume_documents_id_seq', 
                            COALESCE((SELECT MAX(id) FROM resume_documents), 1)
                        );
                    """))
                    
                    # Drop backup table
                    logger.info("🗑️ Cleaning up backup...")
                    conn.execute(text("DROP TABLE resume_documents_backup;"))
                
                logger.info("✅ Migration completed successfully!")
                
    except Exception as e:
        logger.error(f"❌ Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    migrate() 