"""
Pinecone Vector Store for production-grade vector similarity search
"""
import os
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available - install with: pip install pinecone-client")


class PineconeVectorStore:
    """Production-grade vector store using Pinecone"""
    
    def __init__(self, api_key: str = None, environment: str = "us-east-1", index_name: str = "smart-form-fill"):
        """
        Initialize Pinecone Vector Store
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (default: us-east-1)
            index_name: Name of the Pinecone index
        """
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment
        self.index_name = index_name
        self.vector_dim = 1536  # OpenAI embedding dimension
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Create or connect to index
        self._setup_index()
        
        logger.info(f"✅ Pinecone Vector Store initialized: {index_name}")
    
    def _setup_index(self):
        """Create or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.vector_dim,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.environment
                    )
                )
                logger.info(f"✅ Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"✅ Connected to existing Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Pinecone index: {e}")
            raise
    
    def store_embeddings(self, document_id: str, user_id: str, chunk_data: List[Dict[str, Any]], document_type: str = "resume"):
        """
        Store document embeddings in Pinecone
        """
        try:
            vectors_to_upsert = []
            
            for chunk in chunk_data:
                chunk_id = chunk.get("chunk_id", f"chunk_{uuid.uuid4().hex[:8]}")
                text = chunk.get("text", "")
                embedding = chunk.get("embedding")
                
                if embedding is None:
                    logger.warning(f"No embedding for chunk {chunk_id}, skipping")
                    continue
                
                # Convert numpy array to list if needed
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                # Create unique vector ID
                vector_id = f"{user_id}_{document_type}_{document_id}_{chunk_id}"
                
                # Prepare metadata
                metadata = {
                    "user_id": user_id,
                    "document_id": document_id,
                    "document_type": document_type,
                    "chunk_id": chunk_id,
                    "text": text[:1000],  # Pinecone metadata limit
                }
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"✅ Stored {len(vectors_to_upsert)} vectors in Pinecone for user {user_id} ({document_type})")
            return len(vectors_to_upsert)
            
        except Exception as e:
            logger.error(f"❌ Error storing embeddings in Pinecone: {e}")
            raise
    
    def search_similar_by_document_type(self, query_embedding, user_id, document_type, top_k=5, min_score=0.3):
        """
        Search for similar vectors in Pinecone filtered by document type
        """
        try:
            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Search with filters
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={
                    "user_id": user_id,
                    "document_type": document_type
                }
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                if match.score >= min_score:
                    formatted_results.append({
                        'id': match.id,
                        'score': float(match.score),
                        'text': match.metadata.get('text', ''),
                        'document_id': match.metadata.get('document_id', ''),
                        'chunk_id': match.metadata.get('chunk_id', ''),
                    })
            
            logger.info(f"✅ Pinecone search found {len(formatted_results)} results for user {user_id} ({document_type})")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Pinecone search failed: {e}")
            return []
    
    def delete_document(self, document_id: str, user_id: str, document_type: str = "resume"):
        """Delete document vectors from Pinecone"""
        try:
            # Delete by filter
            self.index.delete(
                filter={
                    "user_id": user_id,
                    "document_id": document_id,
                    "document_type": document_type
                }
            )
            logger.info(f"✅ Deleted document {document_id} from Pinecone")
            
        except Exception as e:
            # Don't fail if document doesn't exist
            if "Namespace not found" in str(e) or "404" in str(e):
                logger.info(f"📝 Document {document_id} not found in Pinecone (this is OK for new documents)")
            else:
                logger.error(f"❌ Error deleting document from Pinecone: {e}")
                raise
    
    def get_document_stats(self, user_id: str) -> Dict[str, Any]:
        """Get document statistics (limited in Pinecone)"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            logger.error(f"❌ Error getting Pinecone stats: {e}")
            return {}
    
    def delete_all_documents(self, user_id: str, document_type: str = None):
        """
        Delete all documents for a user, optionally filtered by document type
        """
        try:
            # Build filter for user and optionally document type
            if document_type:
                filter_dict = {
                    "user_id": {"$eq": user_id},
                    "document_type": {"$eq": document_type}
                }
            else:
                filter_dict = {
                    "user_id": {"$eq": user_id}
                }
            
            # Delete vectors matching the filter
            delete_response = self.index.delete(filter=filter_dict)
            logger.info(f"✅ Deleted documents for user {user_id} ({document_type or 'all types'}) from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting documents for user {user_id} from Pinecone: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check Pinecone connection health"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "status": "healthy",
                "service": "pinecone",
                "index_name": self.index_name,
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "pinecone",
                "error": str(e)
            } 