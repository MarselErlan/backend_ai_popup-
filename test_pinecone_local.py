#!/usr/bin/env python3
"""
Test Pinecone Integration Locally
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_pinecone_connection():
    """Test basic Pinecone connection"""
    print("🧪 Testing Pinecone Connection...")
    
    # Check environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY not found in environment")
        return False
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    print("✅ Environment variables found")
    
    try:
        from app.services.pinecone_vector_store import PineconeVectorStore
        
        # Initialize Pinecone
        print("🔄 Initializing Pinecone...")
        vector_store = PineconeVectorStore()
        
        # Test health check
        print("🔄 Testing health check...")
        health = vector_store.health_check()
        print(f"📊 Health check result: {health}")
        
        if health.get("status") == "healthy":
            print("✅ Pinecone connection successful!")
            return True
        else:
            print("❌ Pinecone health check failed")
            return False
            
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False

def test_embedding_service():
    """Test EmbeddingService with Pinecone"""
    print("\n🧪 Testing EmbeddingService with Pinecone...")
    
    try:
        from app.services.embedding_service import EmbeddingService
        
        # Initialize with Pinecone
        print("🔄 Initializing EmbeddingService...")
        embedding_service = EmbeddingService(use_pinecone=True)
        
        # Test document processing
        print("🔄 Testing document processing...")
        
        # Sample document
        sample_text = """
        John Doe
        Software Engineer
        5 years of experience in Python, FastAPI, and machine learning.
        Skills: Python, JavaScript, React, PostgreSQL, Redis
        Education: BS Computer Science, Stanford University
        """
        
        # Process document using the correct method
        result = embedding_service.process_document(
            document_id="test_doc_001",
            user_id="test_user",
            content=sample_text,
            reprocess=True
        )
        
        print(f"✅ Document processed: {result}")
        
        # Test search
        print("🔄 Testing vector search...")
        search_results = embedding_service.search_similar_by_document_type(
            query="Python developer with machine learning experience",
            user_id="test_user",
            document_type="resume",
            top_k=3
        )
        
        print(f"📊 Search results: {len(search_results)} found")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. Score: {result.get('score', 0):.3f} - {result.get('text', '')[:100]}...")
        
        print("✅ EmbeddingService test successful!")
        return True
        
    except Exception as e:
        print(f"❌ EmbeddingService test failed: {e}")
        return False

def test_smart_llm_service():
    """Test SmartLLMService with Pinecone"""
    print("\n🧪 Testing SmartLLMService with Pinecone...")
    
    try:
        from app.services.llm_service import SmartLLMService
        
        # Initialize SmartLLMService
        print("🔄 Initializing SmartLLMService...")
        llm_service = SmartLLMService()
        
        print("✅ SmartLLMService initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ SmartLLMService test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Pinecone Integration Tests\n")
    
    tests = [
        ("Pinecone Connection", test_pinecone_connection),
        ("EmbeddingService", test_embedding_service),
        ("SmartLLMService", test_smart_llm_service),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 Running: {test_name}")
        print('='*50)
        
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pinecone integration is working!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 