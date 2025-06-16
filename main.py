#!/usr/bin/env python3
"""
Smart Form Fill API - Field-by-Field Form Filling
OPTIMIZED VERSION with Performance Enhancements
"""
# %%
import os
import asyncio
from functools import lru_cache
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends, Request, UploadFile, File, Form
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from httpcore import request
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session
import io

# Import auth components
from database import get_db
from models import User
from auth import create_access_token, get_current_user, get_current_user_id

# Import form filling services
from app.services.form_filler_optimized import OptimizedFormFiller

# Import database-based extractors
from resume_extractor_optimized import ResumeExtractorOptimized
from personal_info_extractor_optimized import PersonalInfoExtractorOptimized
from app.services.document_service import DocumentService
from app.models.document_models import ResumeDocument, PersonalInfoDocument

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
        _form_filler = OptimizedFormFiller(
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

# ============================================================================
# USER AUTHENTICATION MODELS
# ============================================================================

class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime
    is_active: bool

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

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
async def generate_field_answer(
    field_request: FieldAnswerRequest,
    user_id: str = Depends(get_current_user_id)
) -> FieldAnswerResponse:
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
        
        logger.info(f"🎯 Generating answer for field: '{field_request.label}' on {field_request.url}")
        logger.info(f"👤 Authenticated user: {user_id}")
        
        # Create a mock field object for the existing logic
        mock_field = {
            "field_purpose": field_request.label,
            "name": field_request.label,
            "selector": "#mock-field",
            "field_type": "text"
        }
        
        # Use the authenticated user_id (no fallback to default)
        result = await form_filler.generate_field_values_optimized([mock_field], {}, user_id)
        
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
# DEMO ENDPOINT (No Authentication Required)
# ============================================================================

@app.post("/api/demo/generate-field-answer", response_model=FieldAnswerResponse)
async def demo_generate_field_answer(field_request: FieldAnswerRequest) -> FieldAnswerResponse:
    """
    🎯 DEMO: Generate field answer without authentication (uses default user data)
    
    This endpoint is for testing/demo purposes only.
    For production use, users should register and use the authenticated endpoint.
    """
    try:
        logger.info(f"🎯 DEMO: Generating answer for field: '{field_request.label}' on {field_request.url}")
        logger.info(f"👤 Using demo user: default")
        
        # Get the form filler service
        form_filler = get_form_filler()
        
        # Create a mock field object for the existing logic
        mock_field = {
            "field_purpose": field_request.label,
            "name": field_request.label,
            "selector": "#mock-field",
            "field_type": "text"
        }
        
        # Use default user for demo
        result = await form_filler.generate_field_values_optimized([mock_field], {}, "default")
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Form filling failed: {result.get('error', 'Unknown error')}")
        
        # Extract the answer from the result
        field_answer = result["values"][0] if result.get("values") else {}
        answer = field_answer.get("value", "Unable to generate answer")
        data_source = field_answer.get("data_source", "unknown")
        reasoning = field_answer.get("reasoning", "No reasoning provided")
        
        # Get performance metrics
        performance_metrics = {
            "processing_time_seconds": result.get("processing_time", 0),
            "optimization_enabled": True,
            "cache_hits": result.get("cache_analytics", {}).get("cache_hit_rate", 0),
            "early_exit": result.get("early_exit", False),
            "tier_exit": result.get("tier_exit", 3),
            "tiers_used": result.get("tiers_used", 3)
        }
        
        logger.info(f"✅ DEMO Generated answer: '{answer}' (source: {data_source}) in {performance_metrics.get('processing_time_seconds', 0):.2f}s")
        
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
        logger.error(f"❌ DEMO Error generating field answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# USER AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/api/auth/register", response_model=TokenResponse)
async def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """🔐 Register new user"""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        new_user = User(email=user_data.email)
        new_user.set_password(user_data.password)
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create access token
        access_token = create_access_token(new_user.id)
        
        logger.info(f"✅ New user registered: {user_data.email}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(
                id=new_user.id,
                email=new_user.email,
                created_at=new_user.created_at,
                is_active=new_user.is_active
            )
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login", response_model=TokenResponse)
async def login_user(user_data: UserLogin, db: Session = Depends(get_db)):
    """🔐 Login user"""
    try:
        # Find user by email
        user = db.query(User).filter(User.email == user_data.email, User.is_active == True).first()
        if not user or not user.verify_password(user_data.password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Create access token
        access_token = create_access_token(user.id)
        
        logger.info(f"✅ User logged in: {user_data.email}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                created_at=user.created_at,
                is_active=user.is_active
            )
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """👤 Get current user info"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
        is_active=current_user.is_active
    )

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

@app.get("/api/v1/documents/status/all")
async def get_documents_status_legacy(user_id: str = Query(None, description="User ID for multi-user support")):
    """⚡ LEGACY: Get documents status using cached service (for backward compatibility)"""
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
# DOCUMENT UPLOAD & CRUD ENDPOINTS
# ============================================================================

# Additional Pydantic models for document operations
class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    document_id: int
    filename: str
    processing_time: float
    file_size: int
    replaced_previous: bool  # Indicates if a previous document was replaced

class DocumentInfoResponse(BaseModel):
    id: int
    filename: str
    file_size: Optional[int] = None
    content_length: Optional[int] = None
    processing_status: str
    is_active: bool
    created_at: str
    last_processed_at: Optional[str] = None
    user_id: Optional[str] = None

# ============================================================================
# RESUME DOCUMENT ENDPOINTS (ONE PER USER)
# ============================================================================

@app.post("/api/v1/resume/upload", response_model=DocumentUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    """
    📄 Upload resume document to database (One per user - replaces existing)
    
    Supports: PDF, DOCX, DOC, TXT files
    """
    start_time = datetime.now()
    
    # Validate file type
    allowed_types = {
        'application/pdf': ['.pdf'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
        'application/msword': ['.doc'],
        'text/plain': ['.txt']
    }
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Supported: PDF, DOCX, DOC, TXT"
        )
    
    try:
        # Check if user already has a resume
        document_service = get_document_service()
        existing_resume = document_service.get_active_resume_document(user_id)
        had_previous = existing_resume is not None
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 10MB")
        
        # Save to database (this will automatically deactivate previous resume)
        document_id = document_service.save_resume_document(
            filename=file.filename,
            file_content=file_content,
            content_type=file.content_type,
            user_id=user_id
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        message = f"Resume '{file.filename}' uploaded successfully"
        if had_previous:
            message += f" (replaced previous resume)"
        
        logger.info(f"✅ Resume uploaded successfully: {file.filename} (ID: {document_id})")
        
        return DocumentUploadResponse(
            status="success",
            message=message,
            document_id=document_id,
            filename=file.filename,
            processing_time=processing_time,
            file_size=len(file_content),
            replaced_previous=had_previous
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Resume upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/v1/resume")
async def get_user_resume(
    user_id: str = Depends(get_current_user_id)
):
    """📄 Get user's resume document info (One per user)"""
    try:
        document_service = get_document_service()
        document = document_service.get_active_resume_document(user_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="No resume found for this user")
        
        return DocumentInfoResponse(
            id=document.id,
            filename=document.filename,
            file_size=document.file_size,
            processing_status=document.processing_status,
            is_active=document.is_active,
            created_at=document.created_at.isoformat() if document.created_at else "",
            last_processed_at=document.last_processed_at.isoformat() if document.last_processed_at else None,
            user_id=document.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get resume document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.get("/api/v1/resume/download")
async def download_user_resume(
    user_id: str = Depends(get_current_user_id)
):
    """⬇️ Download user's resume document"""
    try:
        document_service = get_document_service()
        document = document_service.get_active_resume_document(user_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="No resume found for this user")
        
        return StreamingResponse(
            io.BytesIO(document.file_content),
            media_type=document.content_type,
            headers={"Content-Disposition": f"attachment; filename={document.filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to download resume: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.delete("/api/v1/resume")
async def delete_user_resume(
    user_id: str = Depends(get_current_user_id)
):
    """🗑️ Delete user's resume document"""
    try:
        document_service = get_document_service()
        with document_service.get_session() as session:
            result = session.query(ResumeDocument).filter(
                ResumeDocument.user_id == user_id,
                ResumeDocument.is_active == True
            ).update({"is_active": False})
            
            if result == 0:
                raise HTTPException(status_code=404, detail="No resume found for this user")
            
            session.commit()
            
            return {"status": "success", "message": "Resume deleted successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete resume: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

# ============================================================================
# PERSONAL INFO DOCUMENT ENDPOINTS (ONE PER USER)
# ============================================================================

@app.post("/api/v1/personal-info/upload", response_model=DocumentUploadResponse)
async def upload_personal_info(
    content: str = Form(..., description="Personal information content"),
    user_id: str = Depends(get_current_user_id)
):
    """
    📝 Upload personal information to database (One per user - replaces existing)
    
    Personal info is stored as text content, not files.
    Examples: Contact details, work authorization, salary expectations, preferences
    """
    start_time = datetime.now()
    
    try:
        if len(content.strip()) == 0:
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        if len(content) > 50000:  # 50KB text limit
            raise HTTPException(status_code=400, detail="Content too large. Maximum: 50KB")
        
        # Check if user already has personal info
        document_service = get_document_service()
        existing_info = document_service.get_active_personal_info_document(user_id)
        had_previous = existing_info is not None
        
        # Save to database (this will automatically deactivate previous personal info)
        document_id = document_service.save_personal_info_document(
            filename="personal_info.txt",
            content=content,
            user_id=user_id
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        message = "Personal info uploaded successfully"
        if had_previous:
            message += " (replaced previous personal info)"
        
        logger.info(f"✅ Personal info uploaded successfully (ID: {document_id})")
        
        return DocumentUploadResponse(
            status="success",
            message=message,
            document_id=document_id,
            filename="personal_info.txt",
            processing_time=processing_time,
            file_size=len(content.encode('utf-8')),
            replaced_previous=had_previous
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Personal info upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/v1/personal-info")
async def get_user_personal_info(
    user_id: str = Depends(get_current_user_id)
):
    """📝 Get user's personal info document info (One per user)"""
    try:
        document_service = get_document_service()
        document = document_service.get_active_personal_info_document(user_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="No personal info found for this user")
        
        return DocumentInfoResponse(
            id=document.id,
            filename=document.filename,
            content_length=len(document.content) if document.content else 0,
            processing_status=document.processing_status,
            is_active=document.is_active,
            created_at=document.created_at.isoformat() if document.created_at else "",
            last_processed_at=document.last_processed_at.isoformat() if document.last_processed_at else None,
            user_id=document.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get personal info document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.get("/api/v1/personal-info/download")
async def download_user_personal_info(
    user_id: str = Depends(get_current_user_id)
):
    """⬇️ Download user's personal info document"""
    try:
        document_service = get_document_service()
        document = document_service.get_active_personal_info_document(user_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="No personal info found for this user")
        
        return StreamingResponse(
            io.BytesIO(document.content.encode('utf-8')),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={document.filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to download personal info: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.delete("/api/v1/personal-info")
async def delete_user_personal_info(
    user_id: str = Depends(get_current_user_id)
):
    """🗑️ Delete user's personal info document"""
    try:
        document_service = get_document_service()
        with document_service.get_session() as session:
            result = session.query(PersonalInfoDocument).filter(
                PersonalInfoDocument.user_id == user_id,
                PersonalInfoDocument.is_active == True
            ).update({"is_active": False})
            
            if result == 0:
                raise HTTPException(status_code=404, detail="No personal info found for this user")
            
            session.commit()
            
            return {"status": "success", "message": "Personal info deleted successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete personal info: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

# ============================================================================
# USER DOCUMENT STATUS ENDPOINT
# ============================================================================

@app.get("/api/v1/documents/status")
async def get_user_documents_status(
    user_id: str = Depends(get_current_user_id)
):
    """📊 Get user's document status (One resume + One personal info per user)"""
    try:
        document_service = get_document_service()
        
        # Get user's documents
        resume_doc = document_service.get_active_resume_document(user_id)
        personal_info_doc = document_service.get_active_personal_info_document(user_id)
        
        status_data = {
            "user_id": user_id,
            "resume": None,
            "personal_info": None,
            "summary": {
                "has_resume": resume_doc is not None,
                "has_personal_info": personal_info_doc is not None,
                "documents_ready": False,
                "resume_status": "none",
                "personal_info_status": "none"
            }
        }
        
        # Add resume info if exists
        if resume_doc:
            status_data["resume"] = {
                "id": resume_doc.id,
                "filename": resume_doc.filename,
                "file_size": resume_doc.file_size,
                "processing_status": resume_doc.processing_status,
                "created_at": resume_doc.created_at.isoformat() if resume_doc.created_at else None,
                "last_processed_at": resume_doc.last_processed_at.isoformat() if resume_doc.last_processed_at else None
            }
            status_data["summary"]["resume_status"] = resume_doc.processing_status
        
        # Add personal info if exists
        if personal_info_doc:
            status_data["personal_info"] = {
                "id": personal_info_doc.id,
                "content_length": len(personal_info_doc.content) if personal_info_doc.content else 0,
                "processing_status": personal_info_doc.processing_status,
                "created_at": personal_info_doc.created_at.isoformat() if personal_info_doc.created_at else None,
                "last_processed_at": personal_info_doc.last_processed_at.isoformat() if personal_info_doc.last_processed_at else None
            }
            status_data["summary"]["personal_info_status"] = personal_info_doc.processing_status
        
        # Determine if documents are ready
        resume_ready = resume_doc and resume_doc.processing_status == "completed"
        personal_info_ready = personal_info_doc and personal_info_doc.processing_status == "completed"
        status_data["summary"]["documents_ready"] = resume_ready and personal_info_ready
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": status_data
        }
    
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# ============================================================================
# DEMO UPLOAD ENDPOINTS (No Authentication Required)
# ============================================================================

@app.post("/api/demo/resume/upload", response_model=DocumentUploadResponse)
async def demo_upload_resume(
    file: UploadFile = File(...)
):
    """
    🎯 DEMO: Upload resume without authentication (uses default user)
    
    For testing/demo purposes only.
    """
    start_time = datetime.now()
    
    # Same validation as authenticated endpoint
    allowed_types = {
        'application/pdf': ['.pdf'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
        'application/msword': ['.doc'],
        'text/plain': ['.txt']
    }
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Supported: PDF, DOCX, DOC, TXT"
        )
    
    try:
        # Check if default user already has a resume
        document_service = get_document_service()
        existing_resume = document_service.get_active_resume_document("default")
        had_previous = existing_resume is not None
        
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 10MB")
        
        # Save to database with default user
        document_id = document_service.save_resume_document(
            filename=file.filename,
            file_content=file_content,
            content_type=file.content_type,
            user_id="default"  # Demo user
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        message = f"Demo resume '{file.filename}' uploaded successfully"
        if had_previous:
            message += " (replaced previous demo resume)"
        
        logger.info(f"✅ DEMO: Resume uploaded successfully: {file.filename} (ID: {document_id})")
        
        return DocumentUploadResponse(
            status="success",
            message=message,
            document_id=document_id,
            filename=file.filename,
            processing_time=processing_time,
            file_size=len(file_content),
            replaced_previous=had_previous
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Demo resume upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/demo/resume")
async def demo_get_resume():
    """
    🎯 DEMO: Get resume document info without authentication (uses default user)
    """
    try:
        document_service = get_document_service()
        document = document_service.get_active_resume_document("default")
        
        if not document:
            raise HTTPException(status_code=404, detail="No demo resume found")
        
        return {
            "status": "success",
            "data": {
                "id": document.id,
                "filename": document.filename,
                "file_size": document.file_size,
                "processing_status": document.processing_status,
                "is_active": document.is_active,
                "created_at": document.created_at.isoformat() if document.created_at else "",
                "last_processed_at": document.last_processed_at.isoformat() if document.last_processed_at else None,
                "user_id": document.user_id
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get demo resume: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resume: {str(e)}")

@app.get("/api/demo/resume/download")
async def demo_download_resume():
    """
    🎯 DEMO: Download resume document without authentication (uses default user)
    """
    try:
        document_service = get_document_service()
        document = document_service.get_active_resume_document("default")
        
        if not document:
            raise HTTPException(status_code=404, detail="No demo resume found")
        
        return StreamingResponse(
            io.BytesIO(document.file_content),
            media_type=document.content_type,
            headers={"Content-Disposition": f"attachment; filename={document.filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to download demo resume: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.post("/api/demo/personal-info/upload", response_model=DocumentUploadResponse)
async def demo_upload_personal_info(
    content: str = Form(...)
):
    """
    🎯 DEMO: Upload personal info without authentication (uses default user)
    """
    start_time = datetime.now()
    
    try:
        if len(content.strip()) == 0:
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        if len(content) > 50000:  # 50KB text limit
            raise HTTPException(status_code=400, detail="Content too large. Maximum: 50KB")
        
        # Check if default user already has personal info
        document_service = get_document_service()
        existing_info = document_service.get_active_personal_info_document("default")
        had_previous = existing_info is not None
        
        # Save to database with default user
        document_id = document_service.save_personal_info_document(
            filename="demo_personal_info.txt",
            content=content,
            user_id="default"  # Demo user
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        message = "Demo personal info uploaded successfully"
        if had_previous:
            message += " (replaced previous demo personal info)"
        
        logger.info(f"✅ DEMO: Personal info uploaded successfully (ID: {document_id})")
        
        return DocumentUploadResponse(
            status="success",
            message=message,
            document_id=document_id,
            filename="demo_personal_info.txt",
            processing_time=processing_time,
            file_size=len(content.encode('utf-8')),
            replaced_previous=had_previous
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Demo personal info upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/demo/personal-info")
async def demo_get_personal_info():
    """
    🎯 DEMO: Get personal info document without authentication (uses default user)
    """
    try:
        document_service = get_document_service()
        document = document_service.get_active_personal_info_document("default")
        
        if not document:
            raise HTTPException(status_code=404, detail="No demo personal info found")
        
        return {
            "status": "success",
            "data": {
                "id": document.id,
                "filename": document.filename,
                "content_length": len(document.content) if document.content else 0,
                "processing_status": document.processing_status,
                "is_active": document.is_active,
                "created_at": document.created_at.isoformat() if document.created_at else "",
                "last_processed_at": document.last_processed_at.isoformat() if document.last_processed_at else None,
                "user_id": document.user_id
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get demo personal info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get personal info: {str(e)}")

@app.get("/api/demo/personal-info/download")
async def demo_download_personal_info():
    """
    🎯 DEMO: Download personal info document without authentication (uses default user)
    """
    try:
        document_service = get_document_service()
        document = document_service.get_active_personal_info_document("default")
        
        if not document:
            raise HTTPException(status_code=404, detail="No demo personal info found")
        
        return StreamingResponse(
            io.BytesIO(document.content.encode('utf-8')),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={document.filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to download demo personal info: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/api/demo/documents/status")
async def demo_get_documents_status():
    """
    🎯 DEMO: Get documents status without authentication (uses default user)
    """
    try:
        document_service = get_document_service()
        
        # Get default user's documents
        resume_doc = document_service.get_active_resume_document("default")
        personal_info_doc = document_service.get_active_personal_info_document("default")
        
        status_data = {
            "user_id": "default",
            "resume": None,
            "personal_info": None,
            "summary": {
                "has_resume": resume_doc is not None,
                "has_personal_info": personal_info_doc is not None,
                "documents_ready": False,
                "resume_status": "none",
                "personal_info_status": "none"
            }
        }
        
        # Add resume info if exists
        if resume_doc:
            status_data["resume"] = {
                "id": resume_doc.id,
                "filename": resume_doc.filename,
                "file_size": resume_doc.file_size,
                "processing_status": resume_doc.processing_status,
                "created_at": resume_doc.created_at.isoformat() if resume_doc.created_at else None,
                "last_processed_at": resume_doc.last_processed_at.isoformat() if resume_doc.last_processed_at else None
            }
            status_data["summary"]["resume_status"] = resume_doc.processing_status
        
        # Add personal info if exists
        if personal_info_doc:
            status_data["personal_info"] = {
                "id": personal_info_doc.id,
                "content_length": len(personal_info_doc.content) if personal_info_doc.content else 0,
                "processing_status": personal_info_doc.processing_status,
                "created_at": personal_info_doc.created_at.isoformat() if personal_info_doc.created_at else None,
                "last_processed_at": personal_info_doc.last_processed_at.isoformat() if personal_info_doc.last_processed_at else None
            }
            status_data["summary"]["personal_info_status"] = personal_info_doc.processing_status
        
        # Determine if documents are ready
        resume_ready = resume_doc and resume_doc.processing_status == "completed"
        personal_info_ready = personal_info_doc and personal_info_doc.processing_status == "completed"
        status_data["summary"]["documents_ready"] = resume_ready and personal_info_ready
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": status_data
        }
    
    except Exception as e:
        logger.error(f"❌ Demo status check failed: {e}")
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

 
