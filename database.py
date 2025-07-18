import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment variable, with a fallback for local development
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqT@localhost:5432/ai_popup")

try:
    # Create engine and session
    engine = create_engine(DATABASE_URL)
    
    # Test the connection
    print(f"🔍 Testing database connection...")
    with engine.connect() as conn:
        print(f"✅ Database connection test successful!")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    print(f"✅ Database connection configured successfully")

except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print(f"🔍 DATABASE_URL format being used: {DATABASE_URL.split('@')[0]}@[HIDDEN]")
    raise

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 