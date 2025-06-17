import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
from loguru import logger

# Database URL - use PostgreSQL for production
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database on import
create_tables() 