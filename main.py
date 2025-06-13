#!/usr/bin/env python3
"""
Smart Form Fill API - Vector Database Management + Form Auto-Fill Pipeline
Comprehensive API for managing resume/personal info vector databases and form filling
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

# Import form filling services
from app.services.form_pipeline import FormPipeline
from app.services.cache_service import CacheService

# Import database-based extractors
from resume_extractor_db import ResumeExtractorDB
from personal_info_extractor_db import PersonalInfoExtractorDB
from app.services.document_service import DocumentService

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smart_form_filler")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize services  
cache_service = CacheService()

# Initialize database extractors as singletons for better performance
logger.info("🔧 Initializing database extractors...")
try:
    resume_extractor_db = ResumeExtractorDB(
        openai_api_key=OPENAI_API_KEY,
        database_url=DATABASE_URL,
        use_hf_fallback=True
    )
    personal_info_extractor_db = PersonalInfoExtractorDB(
        openai_api_key=OPENAI_API_KEY,
        database_url=DATABASE_URL,
        use_hf_fallback=True
    )
    logger.info("✅ Database extractors initialized successfully")
except Exception as e:
    logger.error(f"❌ Database extractors initialization failed: {e}")
    # Set to None to handle gracefully in endpoints
    resume_extractor_db = None
    personal_info_extractor_db = None

# Clear Redis cache on startup for fresh analysis
def clear_redis_cache_on_startup():
    """Clear Redis cache on server startup to ensure fresh form analysis"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()  # Test connection
        
        keys = r.keys("*")
        if keys:
            r.flushall()
            logger.info(f"🧹 Cleared {len(keys)} cached entries on startup")
        else:
            logger.info("📭 No cache entries found on startup")
            
    except redis.ConnectionError:
        logger.warning("⚠️  Redis not available - cache clearing skipped")
    except Exception as e:
        logger.warning(f"⚠️  Cache clearing failed: {e}")

# Clear cache on startup
clear_redis_cache_on_startup()

# Initialize form pipeline for auto-filling
try:
    form_pipeline = FormPipeline(
        openai_api_key=OPENAI_API_KEY,
        db_url=DATABASE_URL,
        cache_service=cache_service
    )
    logger.info("✅ Form pipeline initialized successfully")
except Exception as e:
    logger.error(f"❌ Form pipeline initialization failed: {e}")
    form_pipeline = None

# Pydantic models
class ReembedResponse(BaseModel):
    status: str
    message: str
    processing_time: float
    database_info: Dict[str, Any]

# Form pipeline models
class PipelineRequest(BaseModel):
    url: HttpUrl
    user_data: Dict[str, Any] = {}
    force_refresh: bool = False
    submit: bool = False
    manual_submit: bool = True
    headless: bool = False
    use_documents: bool = True

class PipelineResponse(BaseModel):
    status: str
    url: str
    pipeline_status: str
    steps: Dict[str, Any]
    message: str

# Initialize FastAPI app
app = FastAPI(
    title="Smart Form Fill API",
    description="Database-driven Vector Database Management + Form Auto-Fill Pipeline",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document service for database operations
document_service = DocumentService(DATABASE_URL)

# ============================================================================
# VECTOR DATABASE ENDPOINTS
# ============================================================================

@app.post("/api/v1/resume/reembed", response_model=ReembedResponse)
async def reembed_resume_from_database(user_id: str = Query("default", description="User ID for multi-user support")):
    """Re-embed resume from database using consolidated extractor"""
    try:
        if resume_extractor_db is None:
            raise HTTPException(status_code=503, detail="Resume extractor service unavailable")
            
        start_time = datetime.now()
        
        # Set user ID for this operation
        resume_extractor_db.user_id = user_id
        
        # Process resume from database
        result = resume_extractor_db.process_resume()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ReembedResponse(
            status="success",
            message="Resume re-embedding completed from database",
            processing_time=processing_time,
            database_info=result.get("database_info", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database re-embedding failed: {str(e)}")

@app.post("/api/v1/personal-info/reembed", response_model=ReembedResponse)
async def reembed_personal_info_from_database(user_id: str = Query("default", description="User ID for multi-user support")):
    """Re-embed personal info from database using consolidated extractor"""
    try:
        if personal_info_extractor_db is None:
            raise HTTPException(status_code=503, detail="Personal info extractor service unavailable")
            
        start_time = datetime.now()
        
        # Set user ID for this operation
        personal_info_extractor_db.user_id = user_id
        
        # Process personal info from database
        result = personal_info_extractor_db.process_personal_info()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ReembedResponse(
            status="success",
            message="Personal info re-embedding completed from database",
            processing_time=processing_time,
            database_info=result.get("database_info", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database re-embedding failed: {str(e)}")

@app.get("/api/v1/documents/status")
async def get_documents_status(user_id: str = Query(None, description="User ID for multi-user support")):
    """Get status of stored documents in database"""
    try:
        # Get overall stats
        stats = document_service.get_document_stats()
        
        # Get user-specific documents
        resume_doc = document_service.get_active_resume_document(user_id)
        personal_info_doc = document_service.get_active_personal_info_document(user_id)
        
        user_documents = {
            "resume": {
                "exists": resume_doc is not None,
                "filename": resume_doc.filename if resume_doc else None,
                "file_size": resume_doc.file_size if resume_doc else None,
                "processing_status": resume_doc.processing_status if resume_doc else None,
                "last_processed": resume_doc.last_processed_at.isoformat() if resume_doc and resume_doc.last_processed_at else None,
                "created_at": resume_doc.created_at.isoformat() if resume_doc else None
            },
            "personal_info": {
                "exists": personal_info_doc is not None,
                "filename": personal_info_doc.filename if personal_info_doc else None,
                "content_length": len(personal_info_doc.content) if personal_info_doc else None,
                "processing_status": personal_info_doc.processing_status if personal_info_doc else None,
                "last_processed": personal_info_doc.last_processed_at.isoformat() if personal_info_doc and personal_info_doc.last_processed_at else None,
                "created_at": personal_info_doc.created_at.isoformat() if personal_info_doc else None
            }
        }
        
        return {
            "status": "success",
            "user_id": user_id,
            "user_documents": user_documents,
            "system_stats": stats,
            "source": "database"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# ============================================================================
# FORM AUTO-FILL PIPELINE ENDPOINTS
# ============================================================================

@app.post("/api/run-pipeline", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest) -> Dict[str, Any]:
    """
    🚀 MAIN ENDPOINT: Run complete form filling pipeline (analyze → fill → submit)
    
    Features:
    - Automatic form analysis and filling
    - Vector database integration for user data
    - Manual submission mode (keeps browser open)
    - Document-based user data loading
    """
    if not form_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="Form pipeline service not available. Check environment variables."
        )
    
    try:
        url = str(request.url)
        user_data = request.user_data.copy()
        
        # Load user data from vector databases if requested
        if request.use_documents:
            logger.info("📄 Loading user data from vector databases")
            
            try:
                # Search resume for relevant info using database extractor
                if resume_extractor_db:
                    resume_search_result = resume_extractor_db.search_resume("professional experience skills education", k=5)
                    resume_text = ""
                    if resume_search_result and "results" in resume_search_result:
                        resume_text = " ".join([r.get("content", "") for r in resume_search_result["results"]])
                else:
                    resume_text = ""
                
                # Search personal info for contact details using database extractor
                if personal_info_extractor_db:
                    personal_search_result = personal_info_extractor_db.search_personal_info("contact information work authorization salary", k=3)
                    personal_text = ""
                    if personal_search_result and "results" in personal_search_result:
                        personal_text = " ".join([r.get("content", "") for r in personal_search_result["results"]])
                else:
                    personal_text = ""
                
                # Combine vector database data
                vector_data = {
                    "resume_content": resume_text,
                    "personal_info": personal_text,
                    "data_source": "vector_databases"
                }
                
                # Merge with provided user_data
                user_data.update(vector_data)
                logger.info("✅ Vector database data loaded successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load vector database data: {e}")
        
        # Run the form pipeline
        if request.manual_submit:
            logger.info("🖱️ Manual submission mode - browser will stay open")
            
            # For manual submission, we'll use the form filler directly
            form_filler = form_pipeline.form_filler
            form_filler.headless = request.headless
            
            # First analyze the form
            analysis_result = await form_pipeline.form_analyzer.analyze_form(url, request.force_refresh)
            
            if analysis_result["status"] != "success":
                return PipelineResponse(
                    status="error",
                    url=url,
                    pipeline_status="failed",
                    steps={"analysis": {"status": "failed", "error": analysis_result.get("error")}},
                    message=f"Form analysis failed: {analysis_result.get('error', 'Unknown error')}"
                )
            
            # Fill the form with manual submission
            fill_result = await form_filler.auto_fill_form(
                url=url,
                user_data=user_data,
                submit=False,  # Never auto-submit in manual mode
                manual_submit=True
            )
            
            return PipelineResponse(
                status="success" if fill_result["status"] == "success" else "error",
                url=url,
                pipeline_status="completed_manual" if fill_result["status"] == "success" else "failed",
                steps={
                    "analysis": {"status": "success", "timestamp": analysis_result.get("timestamp")},
                    "filling": {
                        "status": fill_result["status"],
                        "filled_fields": fill_result.get("filled_fields", 0),
                        "screenshot": fill_result.get("screenshot")
                    }
                },
                message="Form filled successfully. Browser kept open for manual submission." if fill_result["status"] == "success" else f"Form filling failed: {fill_result.get('error', 'Unknown error')}"
            )
        
        else:
            # Regular pipeline execution
            form_pipeline.form_filler.headless = request.headless
            
            result = await form_pipeline.run_complete_pipeline(
                url=url,
                user_data=user_data,
                force_refresh=request.force_refresh,
                submit_form=request.submit,
                preview_only=False
            )
            
            return PipelineResponse(
                status="success" if result["pipeline_status"] == "completed" else "error",
                url=url,
                pipeline_status=result["pipeline_status"],
                steps=result.get("steps", {}),
                message=result.get("error", "Pipeline completed successfully") if result["pipeline_status"] != "completed" else "Pipeline completed successfully"
            )
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.post("/api/analyze-form")
async def analyze_form(request: PipelineRequest) -> Dict[str, Any]:
    """
    Analyze a form without filling it
    """
    if not form_pipeline:
        raise HTTPException(status_code=503, detail="Form pipeline service not available")
    
    try:
        url = str(request.url)
        result = await form_pipeline.form_analyzer.analyze_form(url, request.force_refresh)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Form analysis failed: {str(e)}")

# ============================================================================
# HEALTH AND INFO ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Form Fill API - Database-driven Vector Database + Form Auto-Fill",
        "version": "3.0.0",
        "features": [
            "Database-driven vector database management",
            "Consolidated resume and personal info extractors",
            "OpenAI + Hugging Face embedding fallback",
            "Intelligent form analysis and auto-filling",
            "LangChain-powered document processing",
            "FAISS vector search capabilities"
        ],
        "endpoints": {
            "vector_db": "/api/v1/",
            "form_pipeline": "/api/run-pipeline",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with database connectivity"""
    try:
        # Test database connection
        db_status = "healthy"
        try:
            document_service.get_document_stats()
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # Test extractor status
        extractor_status = {
            "resume_extractor": "healthy" if resume_extractor_db is not None else "unavailable",
            "personal_info_extractor": "healthy" if personal_info_extractor_db is not None else "unavailable"
        }
        
        overall_status = "healthy" if db_status == "healthy" and all(status == "healthy" for status in extractor_status.values()) else "degraded"
        
        return {
            "status": overall_status,
            "message": "Smart Form Fill API v3.0 - Database-driven",
            "timestamp": datetime.now().isoformat(),
            "database": db_status,
            "extractors": extractor_status,
            "services": {
                "vector_databases": "ready" if overall_status in ["healthy", "degraded"] else "unavailable",
                "form_pipeline": "ready" if form_pipeline else "unavailable"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

 
