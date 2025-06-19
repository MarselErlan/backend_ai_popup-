#!/usr/bin/env python3
"""
Create URL tracking table for Smart Form Fill API
"""
import os
from sqlalchemy import create_engine, text
from loguru import logger

# Database URL
POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL", "postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup")

def create_url_tracking_table():
    """Create the tracked_urls table"""
    try:
        # Create engine
        engine = create_engine(POSTGRES_DB_URL)
        
        # Create the table
        with engine.connect() as conn:
            with conn.begin():
                logger.info("🔄 Creating tracked_urls table...")
                
                # Create table SQL
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS tracked_urls (
                    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
                    user_id VARCHAR NOT NULL REFERENCES users(id),
                    url VARCHAR NOT NULL,
                    title VARCHAR,
                    domain VARCHAR,
                    status VARCHAR DEFAULT 'not_applied' CHECK (status IN ('not_applied', 'applied', 'in_progress')),
                    applied_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                );
                """
                
                conn.execute(text(create_table_sql))
                
                # Create indexes
                logger.info("📊 Creating indexes...")
                
                # Index on user_id for fast user queries
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_tracked_urls_user_id 
                    ON tracked_urls(user_id);
                """))
                
                # Index on status for filtering
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_tracked_urls_status 
                    ON tracked_urls(status);
                """))
                
                # Index on created_at for ordering
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_tracked_urls_created_at 
                    ON tracked_urls(created_at DESC);
                """))
                
                # Index on is_active for filtering active URLs
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_tracked_urls_is_active 
                    ON tracked_urls(is_active);
                """))
                
                # Composite index for common query pattern
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_tracked_urls_user_active 
                    ON tracked_urls(user_id, is_active);
                """))
                
                # Create trigger for updated_at
                logger.info("⚡ Creating update trigger...")
                
                # Function to update updated_at timestamp
                conn.execute(text("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                """))
                
                # Trigger to automatically update updated_at
                conn.execute(text("""
                    DROP TRIGGER IF EXISTS update_tracked_urls_updated_at ON tracked_urls;
                    CREATE TRIGGER update_tracked_urls_updated_at
                        BEFORE UPDATE ON tracked_urls
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                """))
                
                logger.info("✅ URL tracking table created successfully!")
                logger.info("   📋 Table: tracked_urls")
                logger.info("   📊 Indexes: 5 indexes created")
                logger.info("   ⚡ Triggers: updated_at auto-update trigger")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create URL tracking table: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Creating URL tracking table...")
    success = create_url_tracking_table()
    if success:
        logger.info("🎉 URL tracking table setup complete!")
    else:
        logger.error("💥 URL tracking table setup failed!")
        exit(1) 