"""
Redis Vector Store for storing and searching document embeddings
Optimized for redis==5.0.8 with direct command execution
"""
import redis
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
import json
from app.services.vector_search import search_similar, search_similar_by_document_type

class RedisVectorStore:
    def __init__(self, redis_url: str = "redis://localhost:6379", index_name: str = "doc_vectors", vector_dim: int = 1536):
        """
        Initialize Redis Vector Store
        
        Args:
            redis_url: Redis connection URL
            index_name: Name of the search index
            vector_dim: Dimension of vectors (default 1536 for OpenAI embeddings)
        """
        self.redis_url = redis_url
        self.index_name = index_name
        self.vector_dim = vector_dim
        
        # Initialize Redis connection
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        
        # Create search index
        self._create_index()
    
    def _create_index(self):
        """Create RediSearch index for vector similarity search"""
        try:
            # Check if index already exists
            try:
                self.redis_client.execute_command("FT.INFO", self.index_name)
                logger.info(f"✅ Redis index '{self.index_name}' already exists")
                return
            except redis.exceptions.ResponseError as e:
                if "Unknown index name" not in str(e):
                    raise
            
            # Create new index with vector field
            cmd = [
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", f"{self.index_name}:",
                "SCHEMA",
                "document_id", "TEXT",
                "user_id", "TEXT", 
                "chunk_id", "TEXT",
                "text", "TEXT",
                "document_type", "TEXT",
                "embedding", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", str(self.vector_dim),
                "DISTANCE_METRIC", "COSINE"
            ]
            
            result = self.redis_client.execute_command(*cmd)
            logger.info(f"✅ Created Redis search index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"❌ Error creating Redis index: {e}")
            raise
    
    def store_embeddings(self, document_id: str, user_id: str, chunk_data: List[Dict[str, Any]], document_type: str = "resume"):
        """
        Store document embeddings in Redis using namespace pattern:
        doc_vectors:{user_id}:{document_type}:{chunk_id}
        """
        try:
            stored_count = 0
            for chunk in chunk_data:
                chunk_id = chunk.get("chunk_id", f"chunk_{stored_count}")
                text = chunk.get("text", "")
                embedding = chunk.get("embedding")
                if embedding is None:
                    logger.warning(f"No embedding for chunk {chunk_id}, skipping")
                    continue
                # Convert numpy array to bytes
                if isinstance(embedding, np.ndarray):
                    embedding_bytes = embedding.astype(np.float32).tobytes()
                else:
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                # New Redis key pattern: namespace/folder per user and doc type
                redis_key = f"{self.index_name}:{user_id}:{document_type}:{chunk_id}"
                doc_data = {
                    "document_id": document_id,
                    "user_id": user_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "document_type": document_type,
                    "embedding": embedding_bytes
                }
                self.redis_client.hset(redis_key, mapping=doc_data)
                stored_count += 1
            logger.info(f"✅ Stored {stored_count} chunks for user {user_id} ({document_type}) in namespace {self.index_name}:{user_id}:{document_type}:")
            return stored_count
        except Exception as e:
            logger.error(f"❌ Error storing embeddings: {e}")
            raise
    
    def delete_document(self, document_id: str, user_id: str, document_type: str = "resume"):
        """
        Delete all chunks for a user's document type (namespace):
        doc_vectors:{user_id}:{document_type}:*
        """
        try:
            pattern = f"{self.index_name}:{user_id}:{document_type}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                logger.info(f"🗑️ Deleted {deleted_count} chunks for user {user_id} ({document_type}) in namespace {self.index_name}:{user_id}:{document_type}:")
                return deleted_count
            else:
                logger.info(f"No chunks found for user {user_id} ({document_type}) in namespace {self.index_name}:{user_id}:{document_type}:")
                return 0
        except Exception as e:
            logger.error(f"❌ Error deleting document: {e}")
            raise
    
    def get_document_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about stored documents for a user"""
        try:
            # Search for all user documents
            cmd = ["FT.SEARCH", self.index_name, f"@user_id:{user_id}", "LIMIT", "0", "0"]
            result = self.redis_client.execute_command(*cmd)
            
            total_chunks = result[0] if result else 0
            
            return {
                "user_id": user_id,
                "total_chunks": total_chunks,
                "index_name": self.index_name
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting document stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis connection and index health"""
        try:
            # Test Redis connection
            ping_result = self.redis_client.ping()
            
            # Test index
            index_info = self.redis_client.execute_command("FT.INFO", self.index_name)
            
            return {
                "status": "healthy",
                "redis_ping": ping_result,
                "index_exists": True,
                "index_name": self.index_name
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def search_similar_by_document_type(self, query_embedding, user_id, document_type, top_k=5, min_score=0.3):
        """
        Search for similar vectors filtered by document type (resume or personal_info).
        Delegates to the standalone function in vector_search.py.
        """
        return search_similar_by_document_type(
            self.redis_client,
            self.index_name,
            query_embedding,
            user_id,
            document_type,
            top_k=top_k,
            min_score=min_score
        ) 