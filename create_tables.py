#!/usr/bin/env python3
"""
Create database tables for Smart Form Fill API
"""
import os
from sqlalchemy import create_engine
from db.models import Base, User
from loguru import logger

# Database URL
POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL", "postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup")

def create_tables():
    """Create all database tables"""
    try:
        # Create engine
        engine = create_engine(POSTGRES_DB_URL)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Database tables created successfully!")
        logger.info(f"   📊 Database: {POSTGRES_DB_URL}")
        logger.info(f"   📋 Tables created:")
        logger.info(f"      • users (User authentication)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        return False

if __name__ == "__main__":
    create_tables() 