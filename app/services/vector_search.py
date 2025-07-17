import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

# Standalone vector search functions for reuse

def search_similar(redis_client, index_name: str, query_embedding: np.ndarray, user_id: str, top_k: int = 5, min_score: float = 0.3, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search for similar vectors in Redis vector store.
    Args:
        redis_client: Redis connection
        index_name: Name of the search index
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
        # Execute vector search using correct KNN syntax for Redis Search
        knn_query = f"*=>[KNN {top_k * 3} @embedding $query_vector AS vector_score]"
        cmd = [
            "FT.SEARCH", index_name,
            knn_query,
            "PARAMS", "2", "query_vector", query_bytes,
            "SORTBY", "vector_score",
            "LIMIT", "0", str(top_k),
            "RETURN", "6", "document_id", "chunk_id", "text", "user_id", "document_type", "vector_score",
            "DIALECT", "2"
        ]
        result = redis_client.execute_command(*cmd)
        # Parse results
        results = []
        if len(result) > 1:
            num_results = result[0]
            for i in range(1, len(result), 2):
                if i + 1 < len(result):
                    doc_key = result[i].decode() if isinstance(result[i], bytes) else result[i]
                    doc_data = result[i + 1]
                    doc_dict = {}
                    for j in range(0, len(doc_data), 2):
                        if j + 1 < len(doc_data):
                            key = doc_data[j].decode() if isinstance(doc_data[j], bytes) else doc_data[j]
                            value = doc_data[j + 1].decode() if isinstance(doc_data[j + 1], bytes) else doc_data[j + 1]
                            doc_dict[key] = value
                    score = float(doc_dict.get("vector_score", 0))
                    doc_user_id = doc_dict.get("user_id", "")
                    doc_type = doc_dict.get("document_type", "")
                    if doc_user_id != user_id:
                        continue
                    if document_type and doc_type != document_type:
                        continue
                    if score >= min_score:
                        results.append({
                            "document_id": doc_dict.get("document_id", ""),
                            "chunk_id": doc_dict.get("chunk_id", ""),
                            "text": doc_dict.get("text", ""),
                            "score": score
                        })
                        if len(results) >= top_k:
                            break
        logger.info(f"🔍 Found {len(results)} similar chunks for user {user_id}")
        return results
    except Exception as e:
        logger.error(f"❌ Error searching vectors: {e}")
        return []

def search_similar_by_document_type(redis_client, index_name: str, query_embedding: np.ndarray, user_id: str, document_type: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict[str, Any]]:
    """Search for similar vectors filtered by document type"""
    return search_similar(redis_client, index_name, query_embedding, user_id, top_k, min_score, document_type=document_type) 