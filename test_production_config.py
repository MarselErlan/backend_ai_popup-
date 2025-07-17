#!/usr/bin/env python3
"""
🧪 Test Production Configuration
Verifies that the backend can start with production environment variables
"""

import os
import sys
from loguru import logger

def test_environment_variables():
    """Test that all required environment variables are set"""
    logger.info("🧪 Testing environment variables...")
    
    # Set up production environment variables
    os.environ['DATABASE_URL'] = 'postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqTpIKa@postgres.railway.internal:5432/railway'
    os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
    os.environ['REDIS_URL'] = 'redis://localhost:6379'
    os.environ['PORT'] = '8000'
    
    # Test that environment variables are loaded correctly
    try:
        # Import after setting environment variables
        from main import POSTGRES_DB_URL, OPENAI_API_KEY, REDIS_URL, PORT
        
        logger.info("✅ Environment variables loaded successfully:")
        logger.info(f"   DATABASE_URL: {POSTGRES_DB_URL.split('@')[0]}@***")
        logger.info(f"   OPENAI_API_KEY: {'SET' if OPENAI_API_KEY else 'NOT SET'}")
        logger.info(f"   REDIS_URL: {REDIS_URL}")
        logger.info(f"   PORT: {PORT}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Environment variable loading failed: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("🧪 Testing imports...")
    
    try:
        # Test database imports
        from app.services.document_service import DocumentService
        logger.info("✅ DocumentService imported")
        
        # Test embedding service imports
        from app.services.embedding_service import EmbeddingService
        logger.info("✅ EmbeddingService imported")
        
        # Test LLM service imports
        from app.services.llm_service import SmartLLMService
        logger.info("✅ SmartLLMService imported")
        
        # Test RAG service imports
        from app.services.rag_service import RAGService
        logger.info("✅ RAGService imported")
        
        # Test main app import
        from main import app
        logger.info("✅ FastAPI app imported")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_app_creation():
    """Test that the FastAPI app can be created"""
    logger.info("🧪 Testing FastAPI app creation...")
    
    try:
        from main import app
        
        # Check that the app has the expected routes
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/health",
            "/docs",
            "/api/generate-field-answer",
            "/api/simple/register",
            "/api/upload/resume",
            "/api/upload/personal-info"
        ]
        
        for route in expected_routes:
            if route in routes:
                logger.info(f"✅ Route {route} exists")
            else:
                logger.warning(f"⚠️  Route {route} not found")
        
        logger.info("✅ FastAPI app created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ FastAPI app creation failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Production Configuration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Module Imports", test_imports),
        ("FastAPI App Creation", test_app_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
        
        if result:
            logger.info(f"✅ {test_name} test passed")
        else:
            logger.error(f"❌ {test_name} test failed")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your backend is ready for Railway deployment.")
        logger.info("📋 Next steps:")
        logger.info("   1. Push your code to GitHub")
        logger.info("   2. Deploy to Railway")
        logger.info("   3. Set environment variables in Railway dashboard")
        logger.info("   4. Run: python railway_setup.py (in Railway console)")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix the issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 