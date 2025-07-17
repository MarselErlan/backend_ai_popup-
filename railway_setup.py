#!/usr/bin/env python3
"""
🚀 Railway Production Setup Script
Sets up environment variables and initializes database for Railway deployment
"""

import os
import sys
import asyncio
from pathlib import Path
from loguru import logger

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_production_environment():
    """Setup production environment variables for Railway"""
    logger.info("🚀 Setting up Railway production environment...")
    
    # Required environment variables for Railway deployment
    required_vars = [
        "DATABASE_URL",  # Railway PostgreSQL URL
        "OPENAI_API_KEY",  # OpenAI API key
        "REDIS_URL",  # Railway Redis URL (optional, defaults to local)
    ]
    
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("📋 Please set these environment variables in Railway dashboard:")
        logger.info("   DATABASE_URL: Your Railway PostgreSQL connection string")
        logger.info("   OPENAI_API_KEY: Your OpenAI API key")
        logger.info("   REDIS_URL: Your Redis connection string (optional)")
        sys.exit(1)
    
    logger.info("✅ All required environment variables are set")
    
    # Log configuration (without sensitive data)
    database_url = os.getenv("DATABASE_URL", "")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    logger.info(f"📊 Database: {database_url.split('@')[0]}@***")
    logger.info(f"🔴 Redis: {redis_url.split('@')[0]}@***" if '@' in redis_url else f"🔴 Redis: {redis_url}")

async def initialize_database():
    """Initialize database tables for production"""
    logger.info("🗄️  Initializing production database...")
    
    try:
        # Import and run table creation
        from create_tables import create_tables
        create_tables()
        logger.info("✅ Database tables initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        sys.exit(1)

async def test_services():
    """Test that all services can initialize properly"""
    logger.info("🧪 Testing service initialization...")
    
    try:
        # Test database connection
        from app.services.document_service import DocumentService
        database_url = os.getenv("DATABASE_URL")
        doc_service = DocumentService(database_url)
        logger.info("✅ Database service initialized")
        
        # Test embedding service
        from app.services.embedding_service import EmbeddingService
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_service = EmbeddingService(redis_url=redis_url, openai_api_key=openai_api_key)
        logger.info("✅ Embedding service initialized")
        
        # Test LLM service
        from app.services.llm_service import SmartLLMService
        llm_service = SmartLLMService(openai_api_key=openai_api_key)
        logger.info("✅ LLM service initialized")
        
        logger.info("🎉 All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        sys.exit(1)

async def main():
    """Main setup function"""
    logger.info("🚀 Starting Railway Production Setup")
    logger.info("=" * 50)
    
    # Setup environment
    setup_production_environment()
    
    # Initialize database
    await initialize_database()
    
    # Test services
    await test_services()
    
    logger.info("=" * 50)
    logger.info("🎉 Railway setup complete!")
    logger.info("📋 Your backend is ready for production deployment")
    logger.info("🌐 Railway will automatically start your app using:")
    logger.info("   python -m uvicorn main:app --host 0.0.0.0 --port $PORT")

if __name__ == "__main__":
    asyncio.run(main()) 