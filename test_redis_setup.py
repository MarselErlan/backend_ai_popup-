#!/usr/bin/env python3
"""
Test Redis Vector Store Setup
"""
import numpy as np
from app.services.vector_store import RedisVectorStore
from loguru import logger

def test_redis_vector_store():
    """Test Redis vector store functionality"""
    try:
        logger.info("🧪 Testing Redis Vector Store Setup...")
        
        # Initialize vector store
        vector_store = RedisVectorStore(
            redis_url="redis://localhost:6379",
            index_name="test_embeddings"
        )
        
        # Test health check
        health = vector_store.health_check()
        logger.info(f"Health Check: {health}")
        
        if health.get("status") != "healthy":
            logger.error("❌ Redis connection failed")
            return False
        
        # Test storing embeddings
        test_user_id = "test_user"
        test_document_id = "test_doc"
        
        # Create test embeddings
        test_chunks = [
            {
                "chunk_id": "chunk_1",
                "text": "This is a test document about software engineering",
                "embedding": np.random.rand(1536).astype(np.float32)
            },
            {
                "chunk_id": "chunk_2", 
                "text": "Python programming and machine learning",
                "embedding": np.random.rand(1536).astype(np.float32)
            }
        ]
        
        # Store embeddings
        vector_store.store_embeddings(test_document_id, test_user_id, test_chunks)
        logger.info("✅ Test embeddings stored successfully")
        
        # Test search
        query_embedding = np.random.rand(1536).astype(np.float32)
        results = vector_store.search_similar(
            query_embedding=query_embedding,
            user_id=test_user_id,
            top_k=5,
            min_score=0.1
        )
        
        logger.info(f"✅ Search test successful, found {len(results)} results")
        
        # Test document type search
        results_by_type = vector_store.search_similar_by_document_type(
            query_embedding=query_embedding,
            user_id=test_user_id,
            document_type="test",
            top_k=5,
            min_score=0.1
        )
        
        logger.info(f"✅ Document type search successful, found {len(results_by_type)} results")
        
        # Test stats
        stats = vector_store.get_document_stats(test_user_id)
        logger.info(f"✅ Stats test successful: {stats}")
        
        # Clean up test data
        vector_store.delete_document(test_document_id, test_user_id)
        logger.info("✅ Test data cleaned up")
        
        logger.info("🎉 All Redis Vector Store tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis Vector Store test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_redis_vector_store()
    exit(0 if success else 1) 