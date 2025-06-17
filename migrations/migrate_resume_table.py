"""
Migration script to simplify resume_documents table
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
    """Migrate resume_documents table to simplified schema"""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Connect and execute migration
        with engine.connect() as conn:
            # Start transaction
            with conn.begin():
                logger.info("🔄 Starting resume_documents table migration...")
                
                # Check if the original table exists
                has_original_table = table_exists(conn, "resume_documents")
                
                if has_original_table:
                    # 1. Backup existing data
                    logger.info("📦 Backing up existing data...")
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS resume_documents_backup AS 
                        SELECT id, user_id, filename, processing_status 
                        FROM resume_documents;
                    """))
                    
                    # 2. Drop existing table
                    logger.info("🗑️ Dropping existing table...")
                    conn.execute(text("DROP TABLE IF EXISTS resume_documents CASCADE;"))
                else:
                    logger.info("📝 No existing table found, creating new one...")
                
                # 3. Create new table with simplified schema
                logger.info("📝 Creating new table...")
                conn.execute(text("""
                    CREATE TABLE resume_documents (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100),
                        filename VARCHAR(255) NOT NULL,
                        processing_status VARCHAR(50) DEFAULT 'pending'
                    );
                """))
                
                # 4. Create index on user_id
                logger.info("📇 Creating index...")
                conn.execute(text("""
                    CREATE INDEX ix_resume_documents_user_id ON resume_documents (user_id);
                """))
                
                if has_original_table:
                    # 5. Restore data from backup
                    logger.info("♻️ Restoring data...")
                    conn.execute(text("""
                        INSERT INTO resume_documents (id, user_id, filename, processing_status)
                        SELECT id, user_id, filename, processing_status 
                        FROM resume_documents_backup;
                    """))
                    
                    # 6. Reset sequence to max id
                    logger.info("🔄 Resetting sequence...")
                    conn.execute(text("""
                        SELECT setval(
                            'resume_documents_id_seq', 
                            COALESCE((SELECT MAX(id) FROM resume_documents), 1)
                        );
                    """))
                    
                    # 7. Drop backup table
                    logger.info("🗑️ Cleaning up backup...")
                    conn.execute(text("DROP TABLE resume_documents_backup;"))
                
            logger.info("✅ Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    migrate() 