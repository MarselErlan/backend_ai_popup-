#!/usr/bin/env python3
"""
🚀 SIMPLE DEPLOYMENT SETUP
"""

import os
import subprocess
import sys
from loguru import logger

def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        logger.info("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def setup_environment():
    """Setup environment variables"""
    logger.info("🔧 Setting up environment...")
    
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("# AI Form Filler Environment Variables\n")
            f.write("OPENAI_API_KEY=your-openai-api-key-here\n")
            f.write("JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production\n")
            f.write("DATABASE_URL=postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup\n")
            # Removed Supabase - using PostgreSQL instead
            # f.write("SUPABASE_URL=your-supabase-url\n")
            # f.write("SUPABASE_KEY=your-supabase-key\n")
        
        logger.info("✅ Created .env file - PLEASE UPDATE WITH YOUR API KEYS!")
        logger.warning("⚠️  IMPORTANT: Update .env file with your actual API keys before running!")
    else:
        logger.info("✅ .env file already exists")

def create_database():
    """Initialize database"""
    logger.info("🗄️  Initializing database...")
    try:
        from database import create_tables
        create_tables()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        sys.exit(1)

def main():
    """Main deployment setup"""
    logger.info("🚀 Starting AI Form Filler Deployment Setup")
    
    install_dependencies()
    setup_environment()
    create_database()
    
    logger.info("🎉 Deployment setup complete!")
    logger.info("📝 Next steps:")
    logger.info("   1. Update .env file with your API keys")
    logger.info("   2. Run: python -m uvicorn main:app --host 0.0.0.0 --port 8000")
    logger.info("   3. Test at: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 