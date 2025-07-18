#!/usr/bin/env python3
"""
Create database tables for Smart Form Fill API
"""
import os
from sqlalchemy import create_engine
from models import Base, User
from loguru import logger

def create_tables(db_url: str = None):
    """Create all database tables using the provided database URL"""
    try:
        # Use provided URL or get from environment
        database_url = db_url or os.getenv("DATABASE_URL", "postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqTpIKa@interchange.proxy.rlwy.net:30153/railway")
        
        if not database_url:
            raise ValueError("Database URL is not provided")
            
        # Create engine
        engine = create_engine(database_url)
        
        # Verify connection
        with engine.connect() as connection:
            logger.info("✅ Database connection successful!")

        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Database tables verified/created successfully!")
        logger.info(f"   📊 Database: {database_url.split('@')[-1] if '@' in database_url else database_url}") # Avoid logging credentials
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to verify/create tables: {e}")
        return False

if __name__ == "__main__":
    create_tables() 