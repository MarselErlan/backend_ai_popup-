#!/usr/bin/env python3
"""
Smart Form Fill API - Field-by-Field Form Filling
OPTIMIZED VERSION with Performance Enhancements
"""

import os
import asyncio
from functools import lru_cache
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

# Import form filling services
from app.services.form_filler_optimized import FormFillerOptimized

# Import database-based extractors
from resume_extractor_optimized import ResumeExtractorOptimized
from personal_info_extractor_optimized import PersonalInfoExtractorOptimized
from app.services.document_service import DocumentService

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smart_form_filler")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# 🚀 PERFORMANCE OPTIMIZATION: Global singletons with proper lifecycle management
_resume_extractor = None
_personal_info_extractor = None
_form_filler = None
_document_service = None

@lru_cache(maxsize=1)
def get_document_service():
    """Cached document service singleton"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService(DATABASE_URL)
    return _document_service

@lru_cache(maxsize=1)
def get_resume_extractor():
    """Cached resume extractor singleton"""
    global _resume_extractor
    if _resume_extractor is None:
        _resume_extractor = ResumeExtractorOptimized(
            openai_api_key=OPENAI_API_KEY,
            database_url=DATABASE_URL,
            use_hf_fallback=True
        )
    return _resume_extractor

@lru_cache(maxsize=1)
def get_personal_info_extractor():
    """Cached personal info extractor singleton"""
    global _personal_info_extractor
    if _personal_info_extractor is None:
        _personal_info_extractor = PersonalInfoExtractorOptimized(
            openai_api_key=OPENAI_API_KEY,
            database_url=DATABASE_URL,
            use_hf_fallback=True
        )
    return _personal_info_extractor

@lru_cache(maxsize=1)
def get_form_filler():
    """Cached form filler singleton"""
    global _form_filler
    if _form_filler is None:
        _form_filler = FormFillerOptimized(
            openai_api_key=OPENAI_API_KEY,
            resume_extractor=get_resume_extractor(),
            personal_info_extractor=get_personal_info_extractor(),
            headless=True
        )
    return _form_filler

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("🚀 Starting Smart Form Fill API (OPTIMIZED)")
    
    # Pre-warm the singletons
    try:
        logger.info("🔧 Pre-warming services...")
        get_document_service()
        get_resume_extractor()
        get_personal_info_extractor()
        get_form_filler()
        logger.info("✅ All services pre-warmed successfully")
    except Exception as e:
        logger.error(f"❌ Service pre-warming failed: {e}")
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down Smart Form Fill API")

# Pydantic models
class ReembedResponse(BaseModel):
    status: str
    message: str
    processing_time: float
    database_info: Dict[str, Any]

class FieldAnswerRequest(BaseModel):
    label: str
    url: str
    user_id: Optional[str] = "default"

class FieldAnswerResponse(BaseModel):
    answer: str
    data_source: str
    reasoning: str
    status: str
    performance_metrics: Optional[Dict[str, Any]] = None

# Initialize FastAPI app with lifecycle management
app = FastAPI(
    title="Smart Form Fill API (OPTIMIZED)",
    description="High-Performance Field-by-Field Intelligent Form Filling",
    version="4.1.0-optimized",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# OPTIMIZED MAIN FIELD ANSWER ENDPOINT
# ============================================================================

@app.post("/api/generate-field-answer", response_model=FieldAnswerResponse)
async def generate_field_answer(request: FieldAnswerRequest) -> FieldAnswerResponse:
    """
    ⚡ OPTIMIZED: Generate intelligent answer for form field with performance metrics
    """
    start_time = datetime.now()
    
    # 🖥️  CONSOLE LOGGING - Frontend Request Data
    print("=" * 80)
    print("⚡ OPTIMIZED REQUEST - /api/generate-field-answer")
    print("=" * 80)
    print(f"📥 Request Data:")
    print(f"   • Field Label: '{request.label}'")
    print(f"   • Page URL: '{request.url}'")
    print(f"   • User ID: '{request.user_id}'")
    print(f"   • Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        form_filler = get_form_filler()
        
        logger.info(f"🎯 Generating answer for field: '{request.label}' on {request.url}")
        
        # Create a mock field object for the existing logic
        mock_field = {
            "field_purpose": request.label,
            "name": request.label,
            "selector": "#mock-field",
            "field_type": "text"
        }
        
        # Use the optimized field generation logic
        result = await form_filler.generate_field_values_optimized([mock_field], {}, request.user_id)
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Field generation failed: {result.get('error', 'Unknown error')}")
        
        # Extract the answer from the result
        field_mappings = result.get("values", [])
        if not field_mappings:
            raise HTTPException(status_code=500, detail="No field mapping generated")
        
        field_mapping = field_mappings[0]
        answer = field_mapping.get("value", "")
        data_source = field_mapping.get("data_source", "unknown")
        reasoning = field_mapping.get("reasoning", "No reasoning provided")
        
        # Handle skip actions
        if field_mapping.get("action") == "skip" or not answer:
            answer = ""
            data_source = "skipped"
            reasoning = "Field skipped - unable to generate appropriate answer"
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Generated answer: '{answer}' (source: {data_source}) in {processing_time:.2f}s")
        
        # Performance metrics
        performance_metrics = {
            "processing_time_seconds": processing_time,
            "optimization_enabled": True,
            "cache_hits": result.get("cache_hits", 0),
            "database_queries": result.get("database_queries", 0)
        }
        
        # 🖥️  CONSOLE LOGGING - Response Data
        print("📤 OPTIMIZED Response Data:")
        print(f"   • Generated Answer: '{answer}'")
        print(f"   • Data Source: {data_source}")
        print(f"   • Processing Time: {processing_time:.2f}s")
        print(f"   • Reasoning: {reasoning}")
        print("=" * 80)
        
        return FieldAnswerResponse(
            answer=answer,
            data_source=data_source,
            reasoning=reasoning,
            status="success",
            performance_metrics=performance_metrics
        )
    
    except HTTPException:
        raise
    except Exception as e:
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.error(f"❌ Field answer generation failed: {e} (in {processing_time:.2f}s)")
        print(f"❌ ERROR: {str(e)} (in {processing_time:.2f}s)")
        print("=" * 80)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# OPTIMIZED VECTOR DATABASE ENDPOINTS
# ============================================================================

@app.post("/api/v1/resume/reembed", response_model=ReembedResponse)
async def reembed_resume_from_database(user_id: str = Query("default", description="User ID for multi-user support")):
    """⚡ OPTIMIZED: Re-embed resume from database using cached extractor"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("⚡ OPTIMIZED REQUEST - /api/v1/resume/reembed")
    print("=" * 80)
    print(f"📥 Request Data:")
    print(f"   • User ID: '{user_id}'")
    print(f"   • Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        resume_extractor = get_resume_extractor()
        resume_extractor.user_id = user_id  # Set user context
        
        logger.info(f"🔄 Re-embedding resume from database for user: {user_id}")
        
        # Process resume using optimized extractor
        result = resume_extractor.process_resume_optimized()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if result.get("status") == "success":
            logger.info(f"✅ Resume re-embedding completed successfully in {processing_time:.2f}s")
            
            response_data = {
                "status": "success",
                "message": f"Resume re-embedded successfully from database in {processing_time:.2f}s",
                "processing_time": processing_time,
                "database_info": {
                    "user_id": user_id,
                    "chunks_processed": result.get("chunks", 0),
                    "embeddings_created": result.get("embeddings", 0),
                    "vector_dimension": result.get("dimension", 0),
                    "optimization_enabled": True
                }
            }
            
            print("📤 Response Data:")
            print(f"   • Status: SUCCESS")
            print(f"   • Processing Time: {processing_time:.2f}s")
            print(f"   • Chunks: {result.get('chunks', 0)}")
            print("=" * 80)
            
            return ReembedResponse(**response_data)
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Re-embedding failed"))
    
    except Exception as e:
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.error(f"❌ Resume re-embedding failed: {e} (in {processing_time:.2f}s)")
        raise HTTPException(status_code=500, detail=f"Re-embedding failed: {str(e)}")

@app.post("/api/v1/personal-info/reembed", response_model=ReembedResponse)
async def reembed_personal_info_from_database(user_id: str = Query("default", description="User ID for multi-user support")):
    """⚡ OPTIMIZED: Re-embed personal info from database using cached extractor"""
    start_time = datetime.now()
    
    try:
        personal_info_extractor = get_personal_info_extractor()
        personal_info_extractor.user_id = user_id
        
        logger.info(f"🔄 Re-embedding personal info from database for user: {user_id}")
        
        result = personal_info_extractor.process_personal_info_optimized()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if result.get("status") == "success":
            return ReembedResponse(
                status="success",
                message=f"Personal info re-embedded successfully in {processing_time:.2f}s",
                processing_time=processing_time,
                database_info={
                    "user_id": user_id,
                    "chunks_processed": result.get("chunks", 0),
                    "optimization_enabled": True
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Re-embedding failed"))
    
    except Exception as e:
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.error(f"❌ Personal info re-embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Re-embedding failed: {str(e)}")

@app.get("/api/v1/documents/status")
async def get_documents_status(user_id: str = Query(None, description="User ID for multi-user support")):
    """⚡ OPTIMIZED: Get documents status using cached service"""
    try:
        document_service = get_document_service()
        
        # Get document status from database
        status_data = document_service.get_documents_status(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "documents": status_data,
            "optimization_enabled": True,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with optimization info"""
    return {
        "message": "Smart Form Fill API - OPTIMIZED VERSION",
        "version": "4.1.0-optimized",
        "status": "operational",
        "performance_enhancements": [
            "Singleton pattern with cached services",
            "Connection pooling",
            "Pre-warmed services",
            "Optimized vector operations",
            "Performance metrics tracking"
        ],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check with performance metrics"""
    try:
        # Quick health check on cached services
        document_service = get_document_service()
        
        return {
            "status": "healthy",
            "version": "4.1.0-optimized",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "document_service": "healthy",
                "resume_extractor": "cached",
                "personal_info_extractor": "cached",
                "form_filler": "cached"
            },
            "optimization_status": "enabled"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)  # Disable reload for better performance

 
