"""
Redis Vector Store Service for managing document embeddings
"""
from typing import List, Dict, Any, Optional
import numpy as np
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis import Redis
from redis.commands.search.query import Query
from loguru import logger

class RedisVectorStore:
    """Redis Vector Store for managing document embeddings"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "document_embeddings",
        vector_dim: int = 1536  # OpenAI embedding dimension
    ):
        self.redis_client = Redis.from_url(redis_url, decode_responses=True)
        self.index_name = index_name
        self.vector_dim = vector_dim
        self._create_index()
    
    def _create_index(self):
        """Create Redis search index if it doesn't exist"""
        try:
            # Define schema
            schema = (
                TextField("$.document_id", as_name="document_id"),
                TextField("$.chunk_id", as_name="chunk_id"),
                TextField("$.text", as_name="text"),
                VectorField("$.embedding", 
                          "HNSW", {
                              "TYPE": "FLOAT32",
                              "DIM": self.vector_dim,
                              "DISTANCE_METRIC": "COSINE"
                          }, as_name="embedding")
            )
            
            # Create index
            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(
                    prefix=["doc:"],
                    index_type=IndexType.JSON
                )
            )
            logger.info(f"Created Redis search index: {self.index_name}")
        except Exception as e:
            if "Index already exists" not in str(e):
                logger.error(f"Error creating Redis index: {e}")
                raise
    
    def store_embeddings(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Store document chunks and their embeddings
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of dictionaries containing:
                - chunk_id: Unique identifier for the chunk
                - text: Text content of the chunk
                - embedding: Numpy array of the chunk's embedding
        """
        pipe = self.redis_client.pipeline()
        
        for chunk in chunks:
            key = f"doc:{document_id}:{chunk['chunk_id']}"
            data = {
                "document_id": document_id,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": chunk["embedding"].astype(np.float32).tobytes()
            }
            pipe.json().set(key, "$", data)
        
        pipe.execute()
        logger.info(f"Stored {len(chunks)} embeddings for document {document_id}")
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Query vector to search for
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries containing:
                - document_id: ID of the matching document
                - chunk_id: ID of the matching chunk
                - text: Text content of the chunk
                - score: Similarity score
        """
        # Prepare query
        query_vector = query_embedding.astype(np.float32).tobytes()
        q = (
            Query(f"*=>[KNN {top_k} @embedding $query_vector AS score]")
            .dialect(2)
            .return_fields("document_id", "chunk_id", "text", "score")
            .paging(0, top_k)
            .sort_by("score", asc=False)
        )
        
        # Execute search
        results = self.redis_client.ft(self.index_name).search(
            q, 
            query_params={"query_vector": query_vector}
        )
        
        # Process results
        matches = []
        for doc in results.docs:
            if float(doc.score) < min_score:
                continue
            matches.append({
                "document_id": doc.document_id,
                "chunk_id": doc.chunk_id,
                "text": doc.text,
                "score": float(doc.score)
            })
        
        return matches
    
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document"""
        pattern = f"doc:{document_id}:*"
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)
            logger.info(f"Deleted {len(keys)} chunks for document {document_id}") 