"""
Embedding Service for document vector processing
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import openai
from loguru import logger

from app.services.vector_store import RedisVectorStore
from app.services.document_service import DocumentService

class EmbeddingService:
    """Service for processing documents into vector embeddings"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        openai_api_key: Optional[str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ):
        # Initialize OpenAI
        openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize vector store
        self.vector_store = RedisVectorStore(redis_url=redis_url)
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info("Initialized Embedding Service")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk with overlap
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Clean up chunk
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
            
        return chunks
    
    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a list of texts using OpenAI API"""
        try:
            # Get embeddings in batches
            embeddings = []
            for i in range(0, len(texts), 100):  # Process 100 at a time
                batch = texts[i:i + 100]
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [np.array(e.embedding) for e in response.data]
                embeddings.extend(batch_embeddings)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
    
    def process_document(
        self,
        document_id: str,
        content: str,
        reprocess: bool = False
    ) -> None:
        """
        Process document content into vector embeddings
        
        Args:
            document_id: Unique identifier for the document
            content: Text content to process
            reprocess: Whether to reprocess existing embeddings
        """
        try:
            # Delete existing embeddings if reprocessing
            if reprocess:
                self.vector_store.delete_document(document_id)
            
            # Split into chunks
            chunks = self._chunk_text(content)
            if not chunks:
                logger.warning(f"No chunks generated for document {document_id}")
                return
                
            # Get embeddings
            embeddings = self._get_embeddings(chunks)
            
            # Prepare chunks with embeddings
            chunk_data = []
            for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_data.append({
                    "chunk_id": f"chunk_{i}",
                    "text": text,
                    "embedding": embedding
                })
            
            # Store in Redis
            self.vector_store.store_embeddings(document_id, chunk_data)
            
            logger.info(f"Successfully processed document {document_id} into {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            raise
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks
        
        Args:
            query: Text query to search for
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
        """
        try:
            # Get query embedding
            query_embedding = self._get_embeddings([query])[0]
            
            # Search vector store
            results = self.vector_store.search_similar(
                query_embedding,
                top_k=top_k,
                min_score=min_score
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            raise 