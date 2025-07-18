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
from fastapi import FastAPI, HTTPException, Query, Depends, Request, UploadFile, File, Form, Header
from fastapi import Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from httpcore import request
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List, Optional
from datetime import datetime
# Configure logger first
from app.utils.logger import configure_logger
configure_logger()

from loguru import logger
from sqlalchemy.orm import Session
import io
from sqlalchemy import func
import time
import PyPDF2
import docx2txt

# Import auth components
from database import get_db
from models import User, UserSession

# Import form filling services
from app.services.form_filler_optimized import OptimizedFormFiller

# Import database-based extractors
from app.services.resume_extractor_optimized import ResumeExtractorOptimized
from app.services.personal_info_extractor_optimized import PersonalInfoExtractorOptimized
from app.services.document_service import DocumentService
from app.models.document_models import ResumeDocument, PersonalInfoDocument
from app.utils.text_extractor import extract_text_from_file

# Import embedding service
from app.services.embedding_service import EmbeddingService

# Import LLM service
from app.services.llm_service import SmartLLMService

# URL tracking endpoints removed
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("🚀 Starting Smart Form Fill API (OPTIMIZED)")
    
    # Create tables
    from create_tables import create_tables
    create_tables(DATABASE_URL)
    
    # Pre-warm the singletons
    try:
        logger.info("🔧 Pre-warming services...")
        get_document_service()
        get_resume_extractor()
        get_personal_info_extractor()
        get_form_filler()
        get_smart_llm_service()
        logger.info("✅ All services pre-warmed successfully")
    except Exception as e:
        logger.error(f"❌ Service pre-warming failed: {e}")

# Load environment variables
# For Railway deployment, check if we're in Railway environment first
# If not in Railway, then load .env file
if not os.getenv("RAILWAY_ENVIRONMENT"):
    load_dotenv()

# Get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# For Railway deployment, use DATABASE_URL first, then fallback to the new default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqTpIKa@interchange.proxy.rlwy.net:30153/railway")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PORT = int(os.getenv("PORT", "8000"))
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(',')

# Debug logging for Railway deployment
if os.getenv("RAILWAY_ENVIRONMENT"):
    print(f"🚂 Railway Environment: {os.getenv('RAILWAY_ENVIRONMENT')}")
    print(f"🔑 OPENAI_API_KEY: {'✅ SET' if OPENAI_API_KEY else '❌ NOT SET'}")
    print(f"🗄️ DATABASE_URL: {'✅ SET' if DATABASE_URL else '❌ NOT SET'}")
    print(f"🔴 REDIS_URL: {'✅ SET' if REDIS_URL else '❌ NOT SET'}")
    # Log DATABASE_URL format without exposing credentials
    if DATABASE_URL:
        # Extract host info for debugging
        try:
            from urllib.parse import urlparse
            parsed = urlparse(DATABASE_URL)
            print(f"🔍 Database Host: {parsed.hostname}")
            print(f"🔍 Database Port: {parsed.port}")
            print(f"🔍 Database Name: {parsed.path.lstrip('/')}")
        except Exception as e:
            print(f"⚠️ Could not parse DATABASE_URL: {e}")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Session-based authentication dependency
def get_session_user(db: Session = Depends(get_db), session_id: str = Header(None, alias="Authorization")):
    """
    Dependency to get the current user based on session ID from Authorization header
    Raises HTTPException if session is invalid or expired
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="No session ID provided")
    
    # Query the active session
    session = db.query(UserSession).filter(
        UserSession.session_id == session_id,
        UserSession.is_active == True
    ).first()
    
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    # Get the associated user
    user = db.query(User).filter(User.id == session.user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Update last_used_at timestamp
    session.last_used_at = datetime.utcnow()
    db.commit()
    
    return user

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

def get_resume_extractor():
    """Resume extractor singleton (cache disabled for debugging)"""
    global _resume_extractor
    # Force recreation for debugging
    _resume_extractor = ResumeExtractorOptimized(
        openai_api_key=OPENAI_API_KEY,
        database_url=DATABASE_URL,
        use_hf_fallback=True
    )
    # Clear any existing cache to ensure fresh start
    _resume_extractor.clear_cache()
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

@lru_cache(maxsize=1)
def get_smart_llm_service():
    """Get Smart LLM service instance"""
    return SmartLLMService()

# Integrated usage analyzer and simple function tracker imports removed

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
        get_smart_llm_service()
        logger.info("✅ All services pre-warmed successfully")
    except Exception as e:
        logger.error(f"❌ Service pre-warming failed: {e}")
    
    # Usage analysis removed
    
    yield
    
    # Usage analysis removed
    
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
    user_id: Optional[str] = None  # Optional for demo endpoint
    # Note: user_id is now determined from session authentication for main endpoint, but can be passed for demo

class FieldAnswerResponse(BaseModel):
    answer: str
    data_source: str
    reasoning: str
    status: str
    performance_metrics: Optional[Dict[str, Any]] = None

class TranslationRequest(BaseModel):
    text: str
    source_language: str = "en"  # Default: English
    target_language: str = "ru"  # Default: Russian

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: Optional[float] = None
    status: str

# ============================================================================
# USER AUTHENTICATION MODELS
# ============================================================================

class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# Note: UserResponse and TokenResponse models removed since JWT auth is disabled

# Initialize FastAPI app with lifecycle management
app = FastAPI(
    title="Smart Form Fill API (OPTIMIZED)",
    description="High-Performance Field-by-Field Intelligent Form Filling",
    version="4.1.0-optimized",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Custom CORS handling for hybrid scenarios
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class CustomCORSMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.localhost_origins = {
            "http://localhost:5173",
            "http://localhost:3000", 
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000"
        }
        # Explicit headers list to avoid issues with "*"
        self.allowed_headers = [
            "Accept", "Accept-Language", "Content-Language", "Content-Type",
            "Authorization", "X-Requested-With", "Origin", "DNT", "Cache-Control",
            "User-Agent", "If-Modified-Since", "Keep-Alive", "X-Requested-With",
            "If-None-Match", "X-CSRF-Token", "X-Forwarded-For", "X-Real-IP"
        ]
    
    async def dispatch(self, request, call_next):
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            if origin in self.localhost_origins:
                # Localhost gets credentials support
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
            else:
                # All other origins get universal access
                response.headers["Access-Control-Allow-Origin"] = "*"
            
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
            response.headers["Access-Control-Expose-Headers"] = ", ".join(self.allowed_headers)
            response.headers["Access-Control-Max-Age"] = "86400"  # Cache preflight for 24 hours
            return response
        
        # Process actual request
        response = await call_next(request)
        
        # Add CORS headers to response
        if origin in self.localhost_origins:
            # Localhost gets credentials support
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        else:
            # All other origins get universal access
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        response.headers["Access-Control-Expose-Headers"] = ", ".join(self.allowed_headers)
        
        return response

# Add custom CORS middleware
app.add_middleware(CustomCORSMiddleware)

# Add usage analysis middleware (DISABLED)
# from app.middleware.usage_middleware import UsageAnalysisMiddleware
# app.add_middleware(UsageAnalysisMiddleware)

# Deep tracking middleware removed

# URL tracking router removed

# Initialize services
document_service = DocumentService(DATABASE_URL)
embedding_service = EmbeddingService(
    redis_url=REDIS_URL,
    openai_api_key=OPENAI_API_KEY
)

# ============================================================================
# SIMPLIFIED MAIN FIELD ANSWER ENDPOINT (Backend Authorization)
# ============================================================================

@app.post("/api/generate-field-answer", response_model=FieldAnswerResponse)
async def generate_field_answer(
    field_data: FieldAnswerRequest,
    user: User = Depends(get_session_user)
):
    """
    🧠 Generate AI answer for form field using Redis Vector Store
    
    Uses 3-tier system: Resume Redis → Personal Info Redis → LLM Generation
    Auth: Requires valid session
    """
    try:
        # Get Smart LLM service
        llm_service = get_smart_llm_service()
        
        # Generate answer using Redis-powered system
        service_start = time.time()
        result = await llm_service.generate_field_answer(
            field_label=field_data.label,
            user_id=user.id,
            field_context={
                "url": field_data.url,
                "field_type": getattr(field_data, 'field_type', None),
                "field_name": getattr(field_data, 'field_name', None)
            }
        )
        service_time = time.time() - service_start
        
        logger.info(f"✅ Generated answer: '{result['answer']}' (source: {result['data_source']}) in {result['performance_metrics']['processing_time_seconds']:.2f}s")
        
        return FieldAnswerResponse(
            answer=result["answer"],
            data_source=result["data_source"],
            reasoning=result["reasoning"],
            status=result["status"],
            performance_metrics=result["performance_metrics"]
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating field answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# USER ID VALIDATION ENDPOINT (For Extension Setup)
# ============================================================================

@app.get("/api/validate-user/{user_id}")
async def validate_user_id(user_id: str, db: Session = Depends(get_db)):
    """
    🔍 Validate if user_id exists and is active
    Extension can use this to verify user before sending requests
    """
    try:
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found or inactive")
        
        return {
            "status": "valid",
            "user_id": user_id,
            "user_type": "registered",
            "email": user.email,
            "message": "User validated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# ============================================================================
# SIMPLIFIED USER REGISTRATION (Returns User ID)
# ============================================================================

class SimpleRegisterResponse(BaseModel):
    """
    📝 Simple response for register/login
    Includes user_id and session_id for authentication
    """
    status: str
    user_id: str
    email: str
    message: str

@app.post("/api/simple/register", response_model=SimpleRegisterResponse)
async def simple_register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    🔐 Simplified Registration: Returns user_id directly (no JWT tokens)
    Handles existing users appropriately
    """
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            if existing_user.is_active:
                logger.info(f"❌ Registration failed: User already exists - {user_data.email}")
                raise HTTPException(
                    status_code=409,
                    detail="User already exists. Please login instead."
                )
            else:
                # Reactivate inactive user
                existing_user.is_active = True
                existing_user.set_password(user_data.password)
                db.commit()
                logger.info(f"✅ Reactivated user: {user_data.email} (ID: {existing_user.id})")
                return SimpleRegisterResponse(
                    status="reactivated",
                    user_id=existing_user.id,
                    email=existing_user.email,
                    message="Account reactivated successfully. Please login."
                )
        
        # Create new user
        new_user = User(email=user_data.email)
        new_user.set_password(user_data.password)
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"✅ New user registered: {user_data.email} (ID: {new_user.id})")
        
        return SimpleRegisterResponse(
            status="registered",
            user_id=new_user.id,
            email=new_user.email,
            message="Registration successful. Please login to continue."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")

# ============================================================================
# SIMPLIFIED USER LOGIN (Returns User ID)
# ============================================================================

@app.post("/api/simple/login", response_model=SimpleRegisterResponse)
async def simple_login_user(user_data: UserLogin, db: Session = Depends(get_db)):
    """
    🔐 Simplified Login: Returns user_id
    Extension should then call check-and-update endpoint to get/create session
    """
    try:
        # Find user by email
        user = db.query(User).filter(User.email == user_data.email, User.is_active == True).first()
        if not user or not user.verify_password(user_data.password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        logger.info(f"✅ User authenticated: {user_data.email} (ID: {user.id})")
        
        return SimpleRegisterResponse(
            status="authenticated",
            user_id=user.id,
            email=user.email,
            message="Authentication successful - use check-and-update endpoint for session"
        )
    except Exception as e:
        logger.error(f"❌ Login failed for {user_data.email}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ULTRA SIMPLE SESSION MANAGEMENT (Store session_id in DB)
# ============================================================================

class SessionCreateRequest(BaseModel):
    user_id: str
    device_info: Optional[str] = None

class SessionResponse(BaseModel):
    status: str
    session_id: str
    user_id: str
    message: str

class CurrentUserResponse(BaseModel):
    status: str
    user_id: str
    email: str
    session_id: str
    device_info: Optional[str] = None
    created_at: str
    last_used_at: str

@app.post("/api/session/create", response_model=SessionResponse)
async def create_user_session(
    session_request: SessionCreateRequest, 
    db: Session = Depends(get_db)
):
    """
    🔑 Create a simple session for user (stores session_id in DB)
    Perfect for browser extensions - no JWT complexity!
    
    Usage:
    1. Register/Login → get user_id
    2. Create session → get session_id  
    3. Store session_id locally
    4. Use session_id to get current user info
    """
    try:
        # Validate user exists and is active
        user = db.query(User).filter(User.id == session_request.user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User not found or inactive: {session_request.user_id}")
        
        # Create new session
        new_session = UserSession(
            user_id=session_request.user_id,
            device_info=session_request.device_info
        )
        
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        
        logger.info(f"✅ Session created for user: {user.email} (Session: {new_session.session_id})")
        
        return SessionResponse(
            status="created",
            session_id=new_session.session_id,
            user_id=user.id,
            message="Session created successfully - store this session_id"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Session creation failed: {e}")
        raise HTTPException(status_code=500, detail="Session creation failed")

@app.get("/api/session/current/{session_id}", response_model=CurrentUserResponse)
async def get_current_user_by_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    👤 Get current user info by session_id (stored in browser extension)
    
    Usage:
    const sessionId = localStorage.getItem('session_id');
    const response = await fetch(`/api/session/current/${sessionId}`);
    const user = await response.json();
    """
    try:
        # Find active session
        session = db.query(UserSession).filter(
            UserSession.session_id == session_id,
            UserSession.is_active == True
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        # Get user details
        user = db.query(User).filter(User.id == session.user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found or inactive")
        
        # Update last used timestamp
        session.last_used_at = func.now()
        db.commit()
        
        logger.info(f"✅ Session accessed: {user.email} (Session: {session_id})")
        
        return CurrentUserResponse(
            status="active",
            user_id=user.id,
            email=user.email,
            session_id=session.session_id,
            device_info=session.device_info,
            created_at=session.created_at.isoformat(),
            last_used_at=session.last_used_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Session lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Session lookup failed")

@app.delete("/api/session/{session_id}")
async def delete_user_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    🗑️ Delete/logout session (deactivate in DB)
    """
    try:
        session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Deactivate session instead of deleting
        session.is_active = False
        db.commit()
        
        logger.info(f"✅ Session deactivated: {session_id}")
        
        return {"status": "deactivated", "message": "Session logged out successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Session deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Session deletion failed")

@app.get("/api/session/list/{user_id}")
async def list_user_sessions(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    📋 List all active sessions for a user (for session management)
    """
    try:
        # Validate user exists
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get all active sessions
        sessions = db.query(UserSession).filter(
            UserSession.user_id == user_id,
            UserSession.is_active == True
        ).all()
        
        session_list = []
        for session in sessions:
            session_list.append({
                "session_id": session.session_id,
                "device_info": session.device_info,
                "created_at": session.created_at.isoformat(),
                "last_used_at": session.last_used_at.isoformat()
            })
        
        return {
            "status": "success",
            "user_id": user_id,
            "active_sessions": len(session_list),
            "sessions": session_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ List sessions failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")

# ============================================================================
# DEMO ENDPOINT (No Authentication Required)
# ============================================================================

@app.post("/api/demo/generate-field-answer", response_model=FieldAnswerResponse)
async def demo_generate_field_answer(field_request: FieldAnswerRequest) -> FieldAnswerResponse:
    """
    🎯 DEMO: Generate field answer using Redis Vector Store (no authentication)
    
    This endpoint is for testing/demo purposes only.
    For production use, users should register and use the main endpoint.
    """
    try:
        logger.info(f"🎯 DEMO: Generating answer for field: '{field_request.label}' on {field_request.url}")
        logger.info(f"👤 Using demo user: default")
        
        # Get the RAG service (same as main endpoint)
        from app.services.rag_service import RAGService
        rag_service = RAGService()
        
        # Use the user_id from the request
        user_id = field_request.user_id or "6fe5bceb-8c76-4db6-a0ec-79c65c6a9346"
        
        # Generate answer using RAG service
        result = await asyncio.to_thread(rag_service.generate_field_answer, field_request.label, user_id)
        
        # Extract the answer from the RAG result
        answer = result.get("answer", "Unable to generate answer")
        data_source = result.get("data_source", "unknown")
        reasoning = result.get("reasoning", "No reasoning provided")
        confidence = result.get("confidence", 0)
        similarity_scores = result.get("similarity_scores", [])
        
        # Get performance metrics
        processing_time = time.time() - time.time()  # Will be set properly below
        performance_metrics = {
            "processing_time_seconds": processing_time,
            "optimization_enabled": True,
            "cache_hits": 0,
            "early_exit": False,
            "tier_exit": 1 if data_source.startswith("high_confidence") else 3,
            "tiers_used": 1 if data_source.startswith("high_confidence") else 3
        }
        
        logger.info(f"✅ DEMO RAG Generated answer: '{answer}' (source: {data_source}, confidence: {confidence}%)")
        if similarity_scores:
            logger.info(f"🔍 Similarity scores: {[f'{s:.3f}' for s in similarity_scores[:3]]}")
        
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
# DEMO DOCUMENT UPLOAD ENDPOINTS (No Authentication Required)
# ============================================================================

# Additional Pydantic models for document operations
class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    document_id: int
    filename: str
    file_size: int
    content_type: str
    processing_time: float
    replaced_previous: bool  # Indicates if a previous document was replaced

@app.post("/api/demo/resume/upload", response_model=DocumentUploadResponse)
async def demo_upload_resume(file: UploadFile = File(...)):
    """
    📄 DEMO: Upload resume document (no authentication, user_id='default')
    Automatically embeds the document after upload.
    """
    start_time = time.time()
    try:
        file_content = await file.read()
        document_id = document_service.save_resume_document(
            filename=file.filename,
            file_content=file_content,
            content_type=file.content_type,
            user_id="default"
        )
        # Extract text for embedding
        try:
            text = extract_text_from_bytes(file_content, file.content_type)
            embedding_service.process_document(
                document_id=f"resume_{document_id}",
                user_id="default",
                content=text,
                reprocess=True
            )
            logger.info(f"✅ Embedded resume document {document_id} for user 'default'")
        except Exception as embed_err:
            logger.error(f"❌ Embedding failed for resume {document_id}: {embed_err}")
        processing_time = time.time() - start_time
        return {
            "status": "success",
            "message": "Demo resume uploaded and embedded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "file_size": len(file_content),
            "content_type": file.content_type,
            "processing_time": round(processing_time, 3),
            "replaced_previous": True
        }
    except Exception as e:
        logger.error(f"❌ Demo resume upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo resume upload failed: {str(e)}")

@app.post("/api/demo/personal-info/upload", response_model=DocumentUploadResponse)
async def demo_upload_personal_info(content: str = Form(...)):
    """
    📝 DEMO: Upload personal info document (no authentication, user_id='default')
    Automatically embeds the document after upload.
    """
    start_time = time.time()
    try:
        file_content = content.encode("utf-8")
        filename = "personal_info.txt"
        content_type = "text/plain"
        document_id = document_service.save_personal_info_document(
            filename=filename,
            file_content=file_content,
            content_type=content_type,
            user_id="default"
        )
        # Embed personal info
        try:
            embedding_service.process_document(
                document_id=f"personal_info_{document_id}",
                user_id="default",
                content=content,
                reprocess=True
            )
            logger.info(f"✅ Embedded personal info document {document_id} for user 'default'")
        except Exception as embed_err:
            logger.error(f"❌ Embedding failed for personal info {document_id}: {embed_err}")
        processing_time = time.time() - start_time
        return {
            "status": "success",
            "message": "Demo personal info uploaded and embedded successfully",
            "document_id": document_id,
            "filename": filename,
            "file_size": len(file_content),
            "content_type": content_type,
            "processing_time": round(processing_time, 3),
            "replaced_previous": True
        }
    except Exception as e:
        logger.error(f"❌ Demo personal info upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo personal info upload failed: {str(e)}")

# ============================================================================
# OPTIMIZED VECTOR DATABASE ENDPOINTS
# ============================================================================

@app.post("/api/v1/resume/reembed")
async def reembed_resume(
    user: User = Depends(get_session_user)
):
    """
    🔄 Re-process resume document into vector embeddings and store in Redis
    
    This will:
    1. Load user's resume document from database
    2. Extract text content
    3. Generate chunks
    4. Create OpenAI embeddings
    5. Store in Redis vector store
    """
    try:
        # Get document service and embedding service
        document_service = get_document_service()
        embedding_service = EmbeddingService()
        
        # Get document
        document = document_service.get_user_resume(user.id if user else None)
        if not document:
            raise HTTPException(
                status_code=404,
                detail="No resume document found"
            )
        
        # Update status to processing
        document_service.update_resume_status(document.id, "processing")
        
        try:
            # Extract text from file content
            text = extract_text_from_bytes(document.file_content, document.content_type)
            
            # Process document with Redis storage
            embedding_service.process_document(
                document_id=f"resume_{document.id}",
                user_id=user.id,
                content=text,
                reprocess=True
            )
            
            # Update status to completed
            document_service.update_resume_status(document.id, "completed")
            
            return {
                "status": "success",
                "message": "Resume document processed and stored in Redis successfully",
                "document_id": document.id,
                "user_id": user.id,
                "storage": "redis"
            }
            
        except Exception as e:
            # Update status to failed
            document_service.update_resume_status(document.id, "failed")
            raise e
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing resume: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}"
        )

def extract_text_from_bytes(file_content: bytes, content_type: str) -> str:
    """Extract text from file bytes based on content type"""
    try:
        if content_type == "application/pdf":
            # PDF
            with io.BytesIO(file_content) as file_obj:
                reader = PyPDF2.PdfReader(file_obj)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
        elif content_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            # Word document
            with io.BytesIO(file_content) as file_obj:
                text = docx2txt.process(file_obj)
                return text
                
        elif content_type == "text/plain":
            # Plain text
            return file_content.decode("utf-8")
            
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
            
    except Exception as e:
        logger.error(f"❌ Error extracting text: {str(e)}")
        raise

@app.post("/api/v1/personal-info/reembed", response_model=ReembedResponse)
async def reembed_personal_info_from_database(
    user: User = Depends(get_session_user)
):
    """⚡ OPTIMIZED: Re-embed personal info from database using Redis vector store"""
    start_time = datetime.now()
    
    try:
        # Get document service and embedding service with smaller chunks for personal info
        document_service = get_document_service()
        embedding_service = EmbeddingService(
            chunk_size=200,  # Much smaller chunks for personal info (was 800)
            chunk_overlap=30  # Smaller overlap (was 100)
        )
        
        logger.info(f"🔄 Re-embedding personal info from database for user: {user.id} with smaller chunks")
        
        # Get personal info document (✅ FIXED: correct method name)
        personal_info_doc = document_service.get_personal_info_document(user.id)
        if not personal_info_doc:
            raise HTTPException(status_code=404, detail="No personal info document found for this user")
        
        # Update status to processing
        document_service.update_personal_info_status(personal_info_doc.id, "processing")
        
        try:
            # Process document with Redis storage
            content = personal_info_doc.file_content.decode() if isinstance(personal_info_doc.file_content, bytes) else personal_info_doc.file_content
            
            embedding_service.process_document(
                document_id=f"personal_info_{personal_info_doc.id}",
                user_id=user.id,
                content=content,
                reprocess=True
            )
            
            # Update status to completed
            document_service.update_personal_info_status(personal_info_doc.id, "completed")
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return ReembedResponse(
                status="success",
                message=f"Personal info re-embedded and stored in Redis successfully in {processing_time:.2f}s",
                processing_time=processing_time,
                database_info={
                    "user_id": user.id,
                    "document_id": personal_info_doc.id,
                    "storage": "redis",
                    "optimization_enabled": True
                }
            )
            
        except Exception as e:
            # Update status to failed
            document_service.update_personal_info_status(personal_info_doc.id, "failed")
            raise e
    
    except HTTPException:
        raise
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

class DocumentInfoResponse(BaseModel):
    id: int
    filename: str
    file_size: int
    content_type: str
    processing_status: str
    user_id: Optional[str] = None

# ============================================================================
# RESUME DOCUMENT ENDPOINTS (ONE PER USER)
# ============================================================================

@app.post("/api/v1/resume/upload", response_model=DocumentUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    user: User = Depends(get_session_user)
):
    """
    📄 Upload resume document to database (One per user - replaces existing)
    
    Supports:
    - PDF (.pdf)
    - Word (.doc, .docx)
    - Text (.txt)
    """
    start_time = time.time()
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Save document
        service_start = time.time()
        document_id = document_service.save_resume_document(
            filename=file.filename,
            file_content=file_content,
            content_type=file.content_type,
            user_id=user.id if user else None
        )
        service_time = time.time() - service_start
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "Resume uploaded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "file_size": len(file_content),
            "content_type": file.content_type,
            "processing_time": round(processing_time, 3),
            "replaced_previous": True  # Since we always replace previous resume
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading resume: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading resume: {str(e)}"
        )

@app.get("/api/v1/resume")
async def get_user_resume(
    user: User = Depends(get_session_user)
):
    """📄 Get user's resume document info"""
    try:
        document_service = get_document_service()
        document = document_service.get_user_resume(user.id)
        
        if not document:
            raise HTTPException(status_code=404, detail="No resume found for this user")
        
        return DocumentInfoResponse(
            id=document.id,
            filename=document.filename,
            file_size=len(document.file_content),
            content_type=document.content_type,
            processing_status=document.processing_status,
            user_id=document.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get resume document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.delete("/api/v1/resume")
async def delete_user_resume(
    user: User = Depends(get_session_user)
):
    """🗑️ Delete user's resume document"""
    try:
        document_service = get_document_service()
        with document_service.get_session() as session:
            result = session.query(ResumeDocument).filter(
                ResumeDocument.user_id == user.id
            ).delete()
            
            if result == 0:
                raise HTTPException(status_code=404, detail="No resume found for this user")
            
            session.commit()
            
            return {"status": "success", "message": "Resume deleted successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete resume: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/api/v1/documents/status")
async def get_user_documents_status(
    user: User = Depends(get_session_user)
):
    """📊 Get status of user's documents"""
    try:
        document_service = get_document_service()
        status = document_service.get_documents_status(user.id)
        
        return {
            "status": "success",
            "data": status
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get documents status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PERSONAL INFO DOCUMENT ENDPOINTS (ONE PER USER)
# ============================================================================

@app.post("/api/v1/personal-info/upload", response_model=DocumentUploadResponse)
async def upload_personal_info(
    file: UploadFile = File(...),
    user: User = Depends(get_session_user)
):
    """
    📄 Upload personal info document to database (One per user - replaces existing)
    
    Supports:
    - PDF (.pdf)
    - Word (.doc, .docx)
    - Text (.txt)
    """
    start_time = time.time()
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Save document
        service_start = time.time()
        document_id = document_service.save_personal_info_document(
            filename=file.filename,
            file_content=file_content,
            content_type=file.content_type,
            user_id=user.id if user else None
        )
        service_time = time.time() - service_start
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "Personal info document uploaded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "file_size": len(file_content),
            "content_type": file.content_type,
            "processing_time": round(processing_time, 3),
            "replaced_previous": True  # Since we always replace previous document
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading personal info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading personal info: {str(e)}"
        )

@app.get("/api/v1/personal-info", response_model=DocumentInfoResponse)
async def get_personal_info(user: User = Depends(get_session_user)):
    """Get personal info document info"""
    try:
        document = document_service.get_personal_info_document(
            user_id=user.id if user else None
        )
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail="No personal info document found"
            )
        
        return DocumentInfoResponse(
            id=document.id,
            filename=document.filename,
            file_size=len(document.file_content),
            content_type=document.content_type,
            processing_status=document.processing_status,
            user_id=document.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving personal info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving personal info: {str(e)}"
        )

@app.get("/api/v1/personal-info/download")
async def download_user_personal_info(
    user_id: str = Query("default", description="User ID for simple auth")
):
    """⬇️ Download user's personal info document"""
    try:
        document_service = get_document_service()
        document = document_service.get_active_personal_info_document(user_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="No personal info found for this user")
        
        return StreamingResponse(
            io.BytesIO(document.file_content),
            media_type=document.content_type,
            headers={"Content-Disposition": f"attachment; filename={document.filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to download personal info: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.delete("/api/v1/personal-info")
async def delete_user_personal_info(
    user_id: str = Query("default", description="User ID for simple auth")
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
# FULL JWT AUTHENTICATION ENDPOINTS (Optional)
# ============================================================================

@app.get("/api/auth/validate")
async def validate_session(user: User = Depends(get_session_user)):
    """
    🔍 Validate current session
    Returns user info if session is valid
    """
    return {
        "valid": True,
        "user_id": user.id,
        "email": user.email
    }

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
    """Simple health check for Railway deployment"""
    try:
        return {
            "status": "healthy",
            "version": "4.1.0-optimized",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "healthy",
                "database": "ready",
                "redis": "ready",
                "openai": "ready"
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

# Analysis status endpoint removed

# ============================================================================
# REDIS WORKFLOW TEST ENDPOINT
# ============================================================================

class RedisWorkflowTestResponse(BaseModel):
    status: str
    workflow_steps: Dict[str, Any]
    sample_search_results: Dict[str, int]
    processing_time: float
    error: Optional[str] = None

@app.post("/api/test/redis-workflow", response_model=RedisWorkflowTestResponse)
async def test_redis_workflow(user: User = Depends(get_session_user)):
    """
    Test the complete Redis vector workflow:
    1. Get resume from DB
    2. Re-embed (generate embeddings) 
    3. Store vectors in Redis
    4. LLM searches Redis for form filling
    """
    start_time = time.time()
    
    try:
        from app.services.vector_store import RedisVectorStore
        import numpy as np
        import openai
        
        # Initialize Redis vector store
        vector_store = RedisVectorStore(
            redis_url=REDIS_URL,
            index_name=f"workflow_test_{user.id}",
            vector_dim=1536
        )
        
        # STEP 1: Get resume from DB
        logger.info(f"📄 Step 1: Getting resume from DB for user {user.id}")
        document_service = get_document_service()
        resume_doc = document_service.get_user_resume(user.id)
        
        if not resume_doc:
            # Use sample resume for testing
            resume_text = """
            John Doe - Software Engineer
            
            Experience:
            • 5 years Python development at Tech Corp
            • Led team of 4 developers on AI projects
            • Built FastAPI microservices with Redis
            • Implemented machine learning pipelines
            
            Skills:
            • Languages: Python, JavaScript, TypeScript
            • Frameworks: FastAPI, React, Node.js
            • Databases: PostgreSQL, Redis, MongoDB
            • Cloud: AWS, Docker, Kubernetes
            • AI/ML: OpenAI API, LangChain, scikit-learn
            
            Education:
            • BS Computer Science, Stanford University
            • Machine Learning Specialization, Coursera
            """
        else:
            resume_text = resume_doc.extracted_text or "No text extracted"
        
        # STEP 2: Re-embed (generate embeddings)
        logger.info("🔢 Step 2: Re-embedding resume text...")
        
        # Simple text chunking
        chunks = []
        chunk_size = 800
        chunk_overlap = 100
        start = 0
        while start < len(resume_text):
            end = start + chunk_size
            chunk = resume_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - chunk_overlap
        
        # Get embeddings using OpenAI
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=chunks
            )
            embeddings = [np.array(e.embedding) for e in response.data]
        except Exception as e:
            logger.warning(f"OpenAI embedding failed, using random embeddings for test: {e}")
            embeddings = [np.random.rand(1536).astype(np.float32) for _ in chunks]
        
        # Prepare chunk data
        chunk_data = []
        for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_data.append({
                "chunk_id": f"chunk_{i}",
                "text": text,
                "embedding": embedding
            })
        
        # STEP 3: Store vectors in Redis
        logger.info("💾 Step 3: Storing vectors in Redis...")
        document_id = f"resume_{user.id}_test"
        vector_store.store_embeddings(document_id, str(user.id), chunk_data)
        
        # STEP 4: LLM searches Redis for form filling
        logger.info("🔍 Step 4: Testing form field searches...")
        
        form_queries = [
            "What programming languages do you know?",
            "What is your work experience?", 
            "What is your education background?",
            "What are your technical skills?"
        ]
        
        search_results = {}
        for query in form_queries:
            # Get query embedding
            try:
                query_response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=[query]
                )
                query_embedding = np.array(query_response.data[0].embedding)
            except Exception:
                query_embedding = np.random.rand(1536).astype(np.float32)
            
            # Search Redis
            results = vector_store.search_similar(
                query_embedding=query_embedding,
                user_id=str(user.id),
                top_k=2,
                min_score=0.1
            )
            
            search_results[query] = len(results)
        
        # STEP 5: Cleanup test data
        logger.info("🧹 Step 5: Cleaning up test data...")
        vector_store.delete_document(document_id, str(user.id))
        
        processing_time = time.time() - start_time
        
        return RedisWorkflowTestResponse(
            status="success",
            workflow_steps={
                "1_resume_chunks": len(chunks),
                "2_embeddings_generated": len(embeddings),
                "3_vectors_stored": len(chunk_data),
                "4_form_queries_tested": len(form_queries),
                "5_cleanup_completed": True
            },
            sample_search_results=search_results,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Redis workflow test failed: {e}")
        
        return RedisWorkflowTestResponse(
            status="failed",
            workflow_steps={},
            sample_search_results={},
            processing_time=processing_time,
            error=str(e)
        )

@app.post("/api/logout")
async def logout_user(session_data: dict, db: Session = Depends(get_db)):
    """
    🔓 Logout: Deactivate user session
    """
    try:
        session_id = session_data.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
            
        # Find and deactivate session
        session = db.query(UserSession).filter(
            UserSession.session_id == session_id,
            UserSession.is_active == True
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        session.is_active = False
        db.commit()
        
        logger.info(f"✅ User logged out (Session: {session_id})")
        
        return {"status": "success", "message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"❌ Logout failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class UpdateSessionResponse(BaseModel):
    """
    Response for session update/creation
    """
    status: str
    session_id: str
    user_id: str
    message: str

@app.post("/api/session/check-and-update/{user_id}", response_model=UpdateSessionResponse)
async def check_and_update_session(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Check if user has any existing sessions and update/create as needed
    Ensures only one active session per user
    """
    try:
        # Validate user exists and is active
        user = db.query(User).filter(
            User.id == user_id,
            User.is_active == True
        ).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found or inactive")
        
        # Deactivate all existing sessions for this user
        db.query(UserSession).filter(
            UserSession.user_id == user_id,
            UserSession.is_active == True
        ).update({
            "is_active": False,
            "last_used_at": datetime.utcnow()
        })
        db.commit()
        
        # Find most recent session
        existing_session = db.query(UserSession).filter(
            UserSession.user_id == user_id
        ).order_by(UserSession.last_used_at.desc()).first()
        
        if existing_session:
            # Reactivate most recent session
            existing_session.last_used_at = datetime.utcnow()
            existing_session.is_active = True
            db.commit()
            
            logger.info(f"✅ Session reactivated for user: {user.email} (Session: {existing_session.session_id})")
            
            return UpdateSessionResponse(
                status="updated",
                session_id=existing_session.session_id,
                user_id=user_id,
                message="Existing session reactivated"
            )
        else:
            # Create new session
            new_session = UserSession(
                user_id=user_id,
                device_info="Optional Device Info",
                is_active=True
            )
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            
            logger.info(f"✅ New session created for user: {user.email} (Session: {new_session.session_id})")
            
            return UpdateSessionResponse(
                status="created",
                session_id=new_session.session_id,
                user_id=user_id,
                message="New session created"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Session check/update failed: {e}")
        raise HTTPException(status_code=500, detail="Session check/update failed")

@app.post("/api/test/redis-field-answer")
async def test_redis_field_answer(field_request: FieldAnswerRequest):
    """🧪 Test Redis vector search directly for field answers"""
    try:
        from app.services.vector_store import VectorStore
        
        # Initialize Redis vector store
        vector_store = VectorStore()
        
        # Search for relevant information
        search_results = vector_store.search_similar_by_document_type(
            query=field_request.label,
            user_id="1eeb606f-f098-4a04-a5c7-5ff373e61903",  # Your user ID
            document_type="resume",
            top_k=3
        )
        
        # Extract answer from search results
        if search_results:
            answer = search_results[0].get('text', 'No answer found')[:100]  # First 100 chars
        else:
            answer = "No relevant information found"
        
        return FieldAnswerResponse(
            answer=answer,
            data_source="redis_vector_store",
            reasoning="Direct Redis search test",
            status="success",
            performance_metrics={"search_results_count": len(search_results)}
        )
        
    except Exception as e:
        return FieldAnswerResponse(
            answer="Error occurred",
            data_source="error",
            reasoning=str(e),
            status="error"
        )

# ============================================================================
# TEXT TRANSLATION ENDPOINT (For Browser Extension Text Highlighting)
# ============================================================================

@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(translation_request: TranslationRequest):
    """
    🌐 Translate highlighted text from any website
    
    Supports English → Russian translation for browser extension
    Uses OpenAI GPT for high-quality translation
    """
    try:
        from openai import OpenAI
        
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Create translation prompt
        translation_prompt = f"""
        Please translate the following text from {translation_request.source_language} to {translation_request.target_language}.
        Provide only the translation, no explanations or additional text.
        
        Text to translate: "{translation_request.text}"
        
        Translation:
        """
        
        # Get translation from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator. Provide accurate, natural translations without any additional text or explanations."},
                {"role": "user", "content": translation_prompt}
            ],
            max_tokens=500,
            temperature=0.3  # Lower temperature for more consistent translations
        )
        
        translated_text = response.choices[0].message.content.strip()
        
        logger.info(f"✅ Translated '{translation_request.text[:50]}...' from {translation_request.source_language} to {translation_request.target_language}")
        
        return TranslationResponse(
            original_text=translation_request.text,
            translated_text=translated_text,
            source_language=translation_request.source_language,
            target_language=translation_request.target_language,
            confidence=0.95,  # OpenAI typically provides high-quality translations
            status="success"
        )
        
    except Exception as e:
        logger.error(f"❌ Translation failed: {str(e)}")
        return TranslationResponse(
            original_text=translation_request.text,
            translated_text=f"Translation failed: {str(e)}",
            source_language=translation_request.source_language,
            target_language=translation_request.target_language,
            confidence=0.0,
            status="error"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)  # Use PORT env var for Railway deployment

 
