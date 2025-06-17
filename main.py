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
from models import User, UserSession
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
# SIMPLIFIED MAIN FIELD ANSWER ENDPOINT (Backend Authorization)
# ============================================================================

@app.post("/api/generate-field-answer", response_model=FieldAnswerResponse)
async def generate_field_answer(
    field_request: FieldAnswerRequest,
    request: Request,
    db: Session = Depends(get_db)
) -> FieldAnswerResponse:
    """
    ⚡ SIMPLIFIED: Generate intelligent answer for form field with backend authorization
    Frontend only sends: question + user_id (no JWT tokens needed)
    """
    start_time = datetime.now()
    
    # Extract user_id from the request
    user_id = field_request.user_id or "default"
    
    # Backend authorization - validate user exists and is active
    if user_id != "default":
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(status_code=401, detail=f"Invalid or inactive user: {user_id}")
        logger.info(f"✅ User validated: {user.email} (ID: {user_id})")
    else:
        logger.info(f"🎯 Using demo user: default")
    
    # 🖥️  CONSOLE LOGGING - Frontend Request Data + Headers
    print("=" * 80)
    print("⚡ SIMPLIFIED REQUEST - /api/generate-field-answer")
    print("=" * 80)
    print(f"📥 Request Data:")
    print(f"   • Field Label: '{field_request.label}'")
    print(f"   • Page URL: '{field_request.url}'")
    print(f"   • User ID: '{user_id}'")
    print(f"   • Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔒 Headers Debug:")
    print(f"   • Content-Type: '{request.headers.get('content-type', 'NOT SENT')}'")
    print(f"   • User-Agent: '{request.headers.get('user-agent', 'NOT SENT')}'")
    print("=" * 80)
    
    try:
        form_filler = get_form_filler()
        
        logger.info(f"🎯 Generating answer for field: '{field_request.label}' on {field_request.url}")
        logger.info(f"👤 Authorized user: {user_id}")
        
        # Create a mock field object for the existing logic
        mock_field = {
            "field_purpose": field_request.label,
            "name": field_request.label,
            "selector": "#mock-field",
            "field_type": "text"
        }
        
        # Use the validated user_id
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
            "database_queries": result.get("database_queries", 0),
            "backend_authorization": True
        }
        
        # 🖥️  CONSOLE LOGGING - Response Data
        print("📤 SIMPLIFIED Response Data:")
        print(f"   • Generated Answer: '{answer}'")
        print(f"   • Data Source: {data_source}")
        print(f"   • Processing Time: {processing_time:.2f}s")
        print(f"   • Authorization: Backend Validated")
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
# USER ID VALIDATION ENDPOINT (For Extension Setup)
# ============================================================================

@app.get("/api/validate-user/{user_id}")
async def validate_user_id(user_id: str, db: Session = Depends(get_db)):
    """
    🔍 Validate if user_id exists and is active
    Extension can use this to verify user before sending requests
    """
    try:
        if user_id == "default":
            return {
                "status": "valid",
                "user_id": "default",
                "user_type": "demo",
                "message": "Demo user - no registration required"
            }
        
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
    status: str
    user_id: str
    email: str
    message: str

@app.post("/api/simple/register", response_model=SimpleRegisterResponse)
async def simple_register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    🔐 Simplified Registration: Returns user_id directly (no JWT tokens)
    Extension can store the user_id and use it for all requests
    """
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            if existing_user.is_active:
                return SimpleRegisterResponse(
                    status="exists",
                    user_id=existing_user.id,
                    email=existing_user.email,
                    message="User already exists - you can use this user_id"
                )
            else:
                # Reactivate inactive user
                existing_user.is_active = True
                existing_user.set_password(user_data.password)
                db.commit()
                return SimpleRegisterResponse(
                    status="reactivated",
                    user_id=existing_user.id,
                    email=existing_user.email,
                    message="User reactivated successfully"
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
            message="User registered successfully - save this user_id for future requests"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Simple registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

# ============================================================================
# SIMPLIFIED USER LOGIN (Returns User ID)  
# ============================================================================

@app.post("/api/simple/login", response_model=SimpleRegisterResponse)
async def simple_login_user(user_data: UserLogin, db: Session = Depends(get_db)):
    """
    🔐 Simplified Login: Returns user_id directly (no JWT tokens)
    Extension can store the user_id and use it for all requests
    """
    try:
        # Find user by email
        user = db.query(User).filter(User.email == user_data.email, User.is_active == True).first()
        if not user or not user.verify_password(user_data.password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        logger.info(f"✅ User logged in: {user_data.email} (ID: {user.id})")
        
        return SimpleRegisterResponse(
            status="authenticated",
            user_id=user.id,
            email=user.email,
            message="Login successful - save this user_id for future requests"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Simple login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

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
        from sqlalchemy import func
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
    🎯 DEMO: Generate field answer without authentication (uses default user data)
    
    This endpoint is for testing/demo purposes only.
    For production use, users should register and use the main endpoint.
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
# FULL JWT AUTHENTICATION ENDPOINTS (Optional)
# ============================================================================

@app.post("/api/auth/register", response_model=TokenResponse)
async def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """🔐 Full JWT Registration (for users who want JWT tokens)"""
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
    """🔐 Full JWT Login (for users who want JWT tokens)"""
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
    """👤 Get current user info (JWT-based)"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
        is_active=current_user.is_active
    )

# ============================================================================
# JWT-BASED AUTHENTICATED ENDPOINT (For Full Auth Users)
# ============================================================================

@app.post("/api/auth/generate-field-answer", response_model=FieldAnswerResponse)
async def auth_generate_field_answer(
    field_request: FieldAnswerRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id)
) -> FieldAnswerResponse:
    """
    🔐 JWT-AUTHENTICATED: Generate field answer with full JWT authentication
    For users who prefer the complete JWT token-based authentication system
    """
    start_time = datetime.now()
    
    try:
        form_filler = get_form_filler()
        
        logger.info(f"🔐 JWT AUTH: Generating answer for field: '{field_request.label}' on {field_request.url}")
        logger.info(f"👤 JWT Authenticated user: {user_id}")
        
        # Create a mock field object for the existing logic
        mock_field = {
            "field_purpose": field_request.label,
            "name": field_request.label,
            "selector": "#mock-field",
            "field_type": "text"
        }
        
        # Use the JWT-authenticated user_id
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
        
        logger.info(f"✅ JWT AUTH Generated answer: '{answer}' (source: {data_source}) in {processing_time:.2f}s")
        
        # Performance metrics
        performance_metrics = {
            "processing_time_seconds": processing_time,
            "optimization_enabled": True,
            "cache_hits": result.get("cache_hits", 0),
            "database_queries": result.get("database_queries", 0),
            "jwt_authentication": True
        }
        
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
        logger.error(f"❌ JWT AUTH Field answer generation failed: {e} (in {processing_time:.2f}s)")
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
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    """
    📝 Upload personal information document to database (One per user - replaces existing)
    
    Supports: PDF, DOCX, DOC, TXT files
    Examples: Contact details, work authorization, salary expectations, preferences
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
        # Check if user already has personal info
        document_service = get_document_service()
        existing_info = document_service.get_active_personal_info_document(user_id)
        had_previous = existing_info is not None
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 10MB")
        
        # Save to database (this will automatically deactivate previous personal info)
        document_id = document_service.save_personal_info_document(
            filename=file.filename,
            file_content=file_content,
            content_type=file.content_type,
            user_id=user_id
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        message = f"Personal info '{file.filename}' uploaded successfully"
        if had_previous:
            message += f" (replaced previous personal info)"
        
        logger.info(f"✅ Personal info uploaded successfully: {file.filename} (ID: {document_id})")
        
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
                "filename": personal_info_doc.filename,
                "file_size": personal_info_doc.file_size,
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
# DEMO ENDPOINT (No Authentication Required)
# ============================================================================

@app.post("/api/demo/generate-field-answer", response_model=FieldAnswerResponse)
async def demo_generate_field_answer(field_request: FieldAnswerRequest) -> FieldAnswerResponse:
    """
    🎯 DEMO: Generate field answer without authentication (uses default user data)
    
    This endpoint is for testing/demo purposes only.
    For production use, users should register and use the main endpoint.
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
# FULL JWT AUTHENTICATION ENDPOINTS (Optional)
# ============================================================================

@app.post("/api/auth/register", response_model=TokenResponse)
async def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """🔐 Full JWT Registration (for users who want JWT tokens)"""
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
    """🔐 Full JWT Login (for users who want JWT tokens)"""
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
    """👤 Get current user info (JWT-based)"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
        is_active=current_user.is_active
    )

# ============================================================================
# JWT-BASED AUTHENTICATED ENDPOINT (For Full Auth Users)
# ============================================================================

@app.post("/api/auth/generate-field-answer", response_model=FieldAnswerResponse)
async def auth_generate_field_answer(
    field_request: FieldAnswerRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id)
) -> FieldAnswerResponse:
    """
    🔐 JWT-AUTHENTICATED: Generate field answer with full JWT authentication
    For users who prefer the complete JWT token-based authentication system
    """
    start_time = datetime.now()
    
    try:
        form_filler = get_form_filler()
        
        logger.info(f"🔐 JWT AUTH: Generating answer for field: '{field_request.label}' on {field_request.url}")
        logger.info(f"👤 JWT Authenticated user: {user_id}")
        
        # Create a mock field object for the existing logic
        mock_field = {
            "field_purpose": field_request.label,
            "name": field_request.label,
            "selector": "#mock-field",
            "field_type": "text"
        }
        
        # Use the JWT-authenticated user_id
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
        
        logger.info(f"✅ JWT AUTH Generated answer: '{answer}' (source: {data_source}) in {processing_time:.2f}s")
        
        # Performance metrics
        performance_metrics = {
            "processing_time_seconds": processing_time,
            "optimization_enabled": True,
            "cache_hits": result.get("cache_hits", 0),
            "database_queries": result.get("database_queries", 0),
            "jwt_authentication": True
        }
        
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
        logger.error(f"❌ JWT AUTH Field answer generation failed: {e} (in {processing_time:.2f}s)")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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

 
