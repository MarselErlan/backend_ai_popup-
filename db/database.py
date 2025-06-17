import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
from loguru import logger

# Database URL - use PostgreSQL for production  
POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL", "postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup")

# Create engine
engine = create_engine(
    POSTGRES_DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in POSTGRES_DB_URL else {}
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