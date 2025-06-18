"""
Redis Vector Store for storing and searching document embeddings
Optimized for redis==5.0.8 with direct command execution
"""
import redis
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
import json

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
        Store document embeddings in Redis
        
        Args:
            document_id: Unique document identifier
            user_id: User identifier for filtering
            chunk_data: List of chunks with embeddings
            document_type: Type of document (resume, personal_info, etc.)
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
                
                # Create Redis key
                redis_key = f"{self.index_name}:{document_id}:{chunk_id}"
                
                # Store document data
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
            
            logger.info(f"✅ Stored {stored_count} chunks for document {document_id}")
            return stored_count
            
        except Exception as e:
            logger.error(f"❌ Error storing embeddings: {e}")
            raise
    
    def search_similar(self, query_embedding: np.ndarray, user_id: str, top_k: int = 5, min_score: float = 0.7, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector
            user_id: User ID to filter results
            top_k: Number of top results to return
            min_score: Minimum similarity score
            document_type: Optional document type filter
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Convert query embedding to bytes
            if isinstance(query_embedding, np.ndarray):
                query_bytes = query_embedding.astype(np.float32).tobytes()
            else:
                query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Build search query
            base_query = f"@user_id:{user_id}"
            if document_type:
                base_query += f" @document_type:{document_type}"
            
            # Execute vector search using KNN syntax
            knn_query = f"({base_query})=>[KNN {top_k} @embedding $query_vector AS vector_score]"
            cmd = [
                "FT.SEARCH", self.index_name,
                knn_query,
                "PARAMS", "2", "query_vector", query_bytes,
                "SORTBY", "vector_score",
                "LIMIT", "0", str(top_k),
                "RETURN", "4", "document_id", "chunk_id", "text", "vector_score",
                "DIALECT", "2"
            ]
            
            result = self.redis_client.execute_command(*cmd)
            
            # Parse results
            results = []
            if len(result) > 1:
                num_results = result[0]
                for i in range(1, len(result), 2):
                    if i + 1 < len(result):
                        doc_key = result[i].decode() if isinstance(result[i], bytes) else result[i]
                        doc_data = result[i + 1]
                        
                        # Extract fields
                        doc_dict = {}
                        for j in range(0, len(doc_data), 2):
                            if j + 1 < len(doc_data):
                                key = doc_data[j].decode() if isinstance(doc_data[j], bytes) else doc_data[j]
                                value = doc_data[j + 1].decode() if isinstance(doc_data[j + 1], bytes) else doc_data[j + 1]
                                doc_dict[key] = value
                        
                        score = float(doc_dict.get("vector_score", 0))
                        if score >= min_score:
                            results.append({
                                "document_id": doc_dict.get("document_id", ""),
                                "chunk_id": doc_dict.get("chunk_id", ""),
                                "text": doc_dict.get("text", ""),
                                "score": score
                            })
            
            logger.info(f"🔍 Found {len(results)} similar chunks for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching vectors: {e}")
            return []
    
    def search_similar_by_document_type(self, query_embedding: np.ndarray, user_id: str, document_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors filtered by document type"""
        return self.search_similar(query_embedding, user_id, top_k, document_type=document_type)
    
    def delete_document(self, document_id: str, user_id: str):
        """Delete all chunks for a document"""
        try:
            # Find all keys for this document
            pattern = f"{self.index_name}:{document_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                logger.info(f"🗑️ Deleted {deleted_count} chunks for document {document_id}")
                return deleted_count
            else:
                logger.info(f"No chunks found for document {document_id}")
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