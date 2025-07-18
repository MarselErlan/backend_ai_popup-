import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment variable, with a fallback for local development
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqTpIKa@interchange.proxy.rlwy.net:30153/railway")

try:
    # Create engine and session
    print(f"🔍 Attempting to connect to database...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
    
    # Test the connection
    print(f"🔍 Testing database connection...")
    with engine.connect() as conn:
        print(f"✅ Database connection test successful!")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    print(f"✅ Database connection configured successfully")

except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print(f"🔍 DATABASE_URL format being used: {DATABASE_URL.split('@')[0] if '@' in DATABASE_URL else 'Invalid URL'}@[HIDDEN]")
    
    # For debugging: try to parse URL to show what's wrong
    try:
        from urllib.parse import urlparse
        parsed = urlparse(DATABASE_URL)
        print(f"🔍 Parsed URL components:")
        print(f"   - Scheme: {parsed.scheme}")
        print(f"   - Username: {parsed.username}")
        print(f"   - Hostname: {parsed.hostname}")
        print(f"   - Port: {parsed.port}")
        print(f"   - Database: {parsed.path}")
    except Exception as parse_e:
        print(f"⚠️ Could not parse DATABASE_URL: {parse_e}")
    
    raise

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 