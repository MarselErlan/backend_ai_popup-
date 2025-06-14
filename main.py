#!/usr/bin/env python3
"""
Smart Form Fill API - Field-by-Field Form Filling
Simple API for generating individual field answers using intelligent data retrieval
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
from app.services.form_filler import FormFiller

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

# Initialize form filler for intelligent field generation
try:
    form_filler = FormFiller(
        openai_api_key=OPENAI_API_KEY,
        cache_service=None,  # No more Redis cache
        headless=True
    )
    logger.info("✅ Form filler initialized successfully")
except Exception as e:
    logger.error(f"❌ Form filler initialization failed: {e}")
    form_filler = None

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

# Initialize FastAPI app
app = FastAPI(
    title="Smart Form Fill API",
    description="Field-by-Field Intelligent Form Filling with Vector Database Integration",
    version="4.0.0",
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
# MAIN FIELD ANSWER ENDPOINT
# ============================================================================

@app.post("/api/generate-field-answer", response_model=FieldAnswerResponse)
async def generate_field_answer(request: FieldAnswerRequest) -> FieldAnswerResponse:
    """
    Generate intelligent answer for a specific form field using 3-tier data retrieval:
    1. Resume vector database (professional info)
    2. Personal info vector database (personal details)
    3. AI generation (when data is insufficient)
    """
    # 🖥️  CONSOLE LOGGING - Frontend Request Data
    print("=" * 80)
    print("🔵 FRONTEND REQUEST - /api/generate-field-answer")
    print("=" * 80)
    print(f"📥 Request Data:")
    print(f"   • Field Label: '{request.label}'")
    print(f"   • Page URL: '{request.url}'")
    print(f"   • User ID: '{request.user_id}'")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        if form_filler is None:
            raise HTTPException(status_code=503, detail="Form filler service unavailable")
        
        logger.info(f"🎯 Generating answer for field: '{request.label}' on {request.url}")
        
        # Create a mock field object for the existing logic
        mock_field = {
            "field_purpose": request.label,
            "name": request.label,
            "selector": "#mock-field",
            "field_type": "text"
        }
        
        # Use the existing intelligent field generation logic
        result = await form_filler._generate_field_values([mock_field], {})
        
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
        
        logger.info(f"✅ Generated answer: '{answer}' (source: {data_source})")
        
        # 🖥️  CONSOLE LOGGING - Response Data
        print("📤 Response Data:")
        print(f"   • Generated Answer: '{answer}'")
        print(f"   • Data Source: {data_source}")
        print(f"   • Reasoning: {reasoning}")
        print("=" * 80)
        
        return FieldAnswerResponse(
            answer=answer,
            data_source=data_source,
            reasoning=reasoning,
            status="success"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Field answer generation failed: {e}")
        print(f"❌ ERROR: {str(e)}")
        print("=" * 80)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# VECTOR DATABASE ENDPOINTS
# ============================================================================

@app.post("/api/v1/resume/reembed", response_model=ReembedResponse)
async def reembed_resume_from_database(user_id: str = Query("default", description="User ID for multi-user support")):
    """Re-embed resume from database using consolidated extractor"""
    # 🖥️  CONSOLE LOGGING - Frontend Request Data
    print("=" * 80)
    print("🟡 FRONTEND REQUEST - /api/v1/resume/reembed")
    print("=" * 80)
    print(f"📥 Request Data:")
    print(f"   • User ID: '{user_id}'")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
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
        
        # 🖥️  CONSOLE LOGGING - Response Data
        print("📤 Response Data:")
        print(f"   • Status: {result.get('status')}")
        print(f"   • Processing Time: {processing_time:.2f}s")
        print(f"   • Database Info: {result.get('database_info', {})}")
        print("=" * 80)
        
        return ReembedResponse(
            status="success",
            message="Resume re-embedding completed from database",
            processing_time=processing_time,
            database_info=result.get("database_info", {})
        )
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("=" * 80)
        raise HTTPException(status_code=500, detail=f"Database re-embedding failed: {str(e)}")

@app.post("/api/v1/personal-info/reembed", response_model=ReembedResponse)
async def reembed_personal_info_from_database(user_id: str = Query("default", description="User ID for multi-user support")):
    """Re-embed personal info from database using consolidated extractor"""
    # 🖥️  CONSOLE LOGGING - Frontend Request Data
    print("=" * 80)
    print("🟠 FRONTEND REQUEST - /api/v1/personal-info/reembed")
    print("=" * 80)
    print(f"📥 Request Data:")
    print(f"   • User ID: '{user_id}'")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
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
        
        # 🖥️  CONSOLE LOGGING - Response Data
        print("📤 Response Data:")
        print(f"   • Status: {result.get('status')}")
        print(f"   • Processing Time: {processing_time:.2f}s")
        print(f"   • Database Info: {result.get('database_info', {})}")
        print("=" * 80)
        
        return ReembedResponse(
            status="success",
            message="Personal info re-embedding completed from database",
            processing_time=processing_time,
            database_info=result.get("database_info", {})
        )
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("=" * 80)
        raise HTTPException(status_code=500, detail=f"Database re-embedding failed: {str(e)}")

@app.get("/api/v1/documents/status")
async def get_documents_status(user_id: str = Query(None, description="User ID for multi-user support")):
    """Get status of documents in the database"""
    # 🖥️  CONSOLE LOGGING - Frontend Request Data
    print("=" * 80)
    print("🟢 FRONTEND REQUEST - /api/v1/documents/status")
    print("=" * 80)
    print(f"📥 Request Data:")
    print(f"   • User ID: '{user_id or 'default'}'")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Get resume documents status
        resume_docs = document_service.get_user_resume_documents(user_id or "default")
        personal_docs = document_service.get_user_personal_info_documents(user_id or "default")
        
        response_data = {
            "status": "success",
            "user_id": user_id or "default",
            "resume_documents": {
                "count": len(resume_docs),
                "documents": [
                    {
                        "id": doc.id,
                        "filename": doc.filename,
                        "file_size": doc.file_size,
                        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None
                    } for doc in resume_docs
                ]
            },
            "personal_info_documents": {
                "count": len(personal_docs),
                "documents": [
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "content_length": len(doc.content) if doc.content else 0,
                        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None
                    } for doc in personal_docs
                ]
            }
        }
        
        # 🖥️  CONSOLE LOGGING - Response Data
        print("📤 Response Data:")
        print(f"   • Resume Documents: {len(resume_docs)} found")
        print(f"   • Personal Info Documents: {len(personal_docs)} found")
        print(f"   • Total Files: {len(resume_docs) + len(personal_docs)}")
        print("=" * 80)
        
        return response_data
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("=" * 80)
        raise HTTPException(status_code=500, detail=f"Failed to get documents status: {str(e)}")

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint with service information"""
    return {
        "service": "Smart Form Fill API",
        "version": "4.0.0",
        "description": "Field-by-Field Intelligent Form Filling",
        "features": [
            "Individual field answer generation",
            "3-tier intelligent data retrieval",
            "Vector database integration",
            "Multi-user support",
            "Database-driven document management"
        ],
        "endpoints": {
            "main": "/api/generate-field-answer",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services_status = {
        "form_filler": form_filler is not None,
        "resume_extractor": resume_extractor_db is not None,
        "personal_info_extractor": personal_info_extractor_db is not None,
        "database": True  # Assume database is healthy for now
    }
    
    all_healthy = all(services_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": services_status,
        "version": "4.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

 
