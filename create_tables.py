#!/usr/bin/env python3
"""
Create database tables for Smart Form Fill API
"""
import os
from sqlalchemy import create_engine
from models import Base, User, UserToken, UserSession
from loguru import logger

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")

def create_tables():
    """Create all database tables"""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Database tables created successfully!")
        logger.info(f"   📊 Database: {DATABASE_URL}")
        logger.info(f"   📋 Tables created:")
        logger.info(f"      • users (User authentication)")
        logger.info(f"      • user_tokens (JWT tokens)")
        logger.info(f"      • user_sessions (Simple sessions)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        return False

if __name__ == "__main__":
    create_tables() 