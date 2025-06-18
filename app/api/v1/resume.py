"""
Resume API endpoints
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.core.auth import get_session_user
from app.models.user import User

router = APIRouter()

# Initialize services
document_service = DocumentService()
embedding_service = EmbeddingService()

@router.post("/reembed")
async def reembed_resume(
    document_id: int,
    user: User = Depends(get_session_user)
) -> Dict[str, Any]:
    """
    Re-process resume document into vector embeddings
    """
    try:
        # Get document content
        document = document_service.get_resume(document_id, user.id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update status
        document_service.update_resume_status(document_id, "processing")
        
        try:
            # Process document
            embedding_service.process_document(
                document_id=str(document_id),
                content=document.file_content.decode(),
                reprocess=True
            )
            
            # Update status
            document_service.update_resume_status(document_id, "completed")
            
            return {
                "status": "success",
                "message": "Document re-processed successfully"
            }
            
        except Exception as e:
            # Update status and re-raise
            document_service.update_resume_status(document_id, "failed")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing document: {str(e)}"
        )

@router.get("/search")
async def search_resume(
    query: str,
    top_k: int = 5,
    min_score: float = 0.7,
    user: User = Depends(get_session_user)
) -> List[Dict[str, Any]]:
    """
    Search resume content using vector similarity
    """
    try:
        results = embedding_service.search_similar(
            query=query,
            top_k=top_k,
            min_score=min_score
        )
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        ) 