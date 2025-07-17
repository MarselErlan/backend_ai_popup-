"""
🧪 Pytest Configuration and Shared Fixtures
Centralized test configuration with reusable fixtures for all test categories
"""

import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from typing import Generator, Dict, Any
import json
import sqlite3
from datetime import datetime
import requests
import time

# Add the parent directory to the path so we can import from app/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import application modules
try:
    from app.services.document_service import DocumentService
    from app.services.embedding_service import EmbeddingService
    from app.services.llm_service import RedisLLMService
    from app.models.document_models import ResumeDocument, PersonalInfoDocument
    from database import get_db
    from models import User, UserSession
except ImportError as e:
    print(f"⚠️ Warning: Could not import app modules: {e}")
    print("Some fixtures may not work without proper app imports")

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line(
        "markers", "unit: Unit level tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: API or database integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: Full stack end-to-end workflow tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load testing"
    )
    config.addinivalue_line(
        "markers", "analysis: Code analysis and quality tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer than 5 seconds"
    )
    config.addinivalue_line(
        "markers", "api: Tests that require running API server"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        # Get the test file path relative to tests directory
        test_file_path = Path(item.fspath).relative_to(Path(__file__).parent)
        
        # Auto-mark based on directory structure
        if "unit/" in str(test_file_path):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(test_file_path):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.api)
        elif "e2e/" in str(test_file_path):
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.api)
            item.add_marker(pytest.mark.slow)
        elif "performance/" in str(test_file_path):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "analysis/" in str(test_file_path):
            item.add_marker(pytest.mark.analysis)

# ============================================================================
# SERVER AND API FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def api_base_url():
    """Base URL for API testing"""
    return "http://localhost:8000"

@pytest.fixture(scope="session")
def server_health_check(api_base_url):
    """Check if the API server is running before tests"""
    try:
        response = requests.get(f"{api_base_url}/health", timeout=5)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass
    
    pytest.skip("API server is not running. Start with: python main.py")

@pytest.fixture
def api_client(api_base_url, server_health_check):
    """HTTP client for API testing"""
    import requests
    
    class APIClient:
        def __init__(self, base_url):
            self.base_url = base_url
            self.session = requests.Session()
            
        def get(self, endpoint, **kwargs):
            return self.session.get(f"{self.base_url}{endpoint}", **kwargs)
            
        def post(self, endpoint, **kwargs):
            return self.session.post(f"{self.base_url}{endpoint}", **kwargs)
            
        def put(self, endpoint, **kwargs):
            return self.session.put(f"{self.base_url}{endpoint}", **kwargs)
            
        def delete(self, endpoint, **kwargs):
            return self.session.delete(f"{self.base_url}{endpoint}", **kwargs)
    
    return APIClient(api_base_url)

# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_db_url():
    """Test database URL (uses SQLite for testing)"""
    return "sqlite:///test_database.db"

@pytest.fixture
def temp_db(test_db_url):
    """Temporary database for testing"""
    # Create a temporary database file
    db_file = "test_database.db"
    
    # Clean up any existing test database
    if os.path.exists(db_file):
        os.remove(db_file)
    
    # Create database tables (simplified for testing)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create basic tables for testing
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            device_info TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    conn.close()
    
    yield test_db_url
    
    # Cleanup
    if os.path.exists(db_file):
        os.remove(db_file)

@pytest.fixture
def mock_db_session():
    """Mock database session for unit tests"""
    mock_session = Mock()
    mock_session.query = Mock()
    mock_session.add = Mock()
    mock_session.commit = Mock()
    mock_session.rollback = Mock()
    mock_session.close = Mock()
    return mock_session

# ============================================================================
# MOCK LLM AND AI SERVICE FIXTURES
# ============================================================================

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a mock AI response for testing purposes."
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }

@pytest.fixture
def mock_embedding_response():
    """Mock OpenAI embedding response"""
    return {
        "data": [
            {
                "embedding": [0.1] * 1536,  # Mock 1536-dimensional embedding
                "index": 0
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing"""
    mock_service = Mock(spec=RedisLLMService)
    
    # Mock async methods
    mock_service.generate_field_answer = AsyncMock(return_value={
        "answer": "Mock answer from LLM",
        "data_source": "mock_vector_store",
        "reasoning": "Mock reasoning for testing",
        "status": "success",
        "performance_metrics": {
            "processing_time_seconds": 0.1,
            "cache_hit": False,
            "tier_exit": 3
        }
    })
    
    return mock_service

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing"""
    mock_service = Mock(spec=EmbeddingService)
    
    mock_service.process_document = Mock(return_value={
        "status": "success",
        "chunks_processed": 5,
        "embeddings_generated": 5
    })
    
    mock_service.search_similar = Mock(return_value=[
        {
            "text": "Mock search result",
            "score": 0.95,
            "metadata": {"source": "test"}
        }
    ])
    
    return mock_service

# ============================================================================
# FILE AND DOCUMENT FIXTURES
# ============================================================================

@pytest.fixture
def temp_directory():
    """Temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing"""
    return """
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

Contact:
• Email: john.doe@email.com
• Phone: (555) 123-4567
• LinkedIn: linkedin.com/in/johndoe
"""

@pytest.fixture
def sample_personal_info_text():
    """Sample personal information for testing"""
    return """
Personal Information:
Name: John Doe
Email: john.doe@email.com
Phone: (555) 123-4567
Address: 123 Main St, San Francisco, CA 94105
LinkedIn: linkedin.com/in/johndoe
GitHub: github.com/johndoe

Preferences:
• Preferred work location: Remote or San Francisco Bay Area
• Salary expectation: $120,000 - $150,000
• Available start date: 2 weeks notice
• Work authorization: US Citizen
• Willing to relocate: No
"""

@pytest.fixture
def sample_pdf_file(temp_directory, sample_resume_text):
    """Create a sample PDF file for testing"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        pdf_path = temp_directory / "sample_resume.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        
        # Add text to PDF
        text_lines = sample_resume_text.strip().split('\n')
        y_position = 750
        
        for line in text_lines:
            if line.strip():
                c.drawString(50, y_position, line.strip())
                y_position -= 20
                if y_position < 50:
                    c.showPage()
                    y_position = 750
        
        c.save()
        return pdf_path
        
    except ImportError:
        # If reportlab is not available, create a dummy file
        pdf_path = temp_directory / "sample_resume.pdf"
        pdf_path.write_bytes(b"Mock PDF content for testing")
        return pdf_path

@pytest.fixture
def sample_text_file(temp_directory, sample_resume_text):
    """Create a sample text file for testing"""
    text_path = temp_directory / "sample_resume.txt"
    text_path.write_text(sample_resume_text)
    return text_path

@pytest.fixture
def mock_document_service():
    """Mock document service for testing"""
    mock_service = Mock(spec=DocumentService)
    
    # Mock document objects
    mock_resume = Mock(spec=ResumeDocument)
    mock_resume.id = 1
    mock_resume.filename = "test_resume.pdf"
    mock_resume.file_content = b"Mock PDF content"
    mock_resume.content_type = "application/pdf"
    mock_resume.processing_status = "completed"
    mock_resume.user_id = "test-user-123"
    
    mock_personal_info = Mock(spec=PersonalInfoDocument)
    mock_personal_info.id = 1
    mock_personal_info.filename = "personal_info.txt"
    mock_personal_info.file_content = b"Mock personal info content"
    mock_personal_info.content_type = "text/plain"
    mock_personal_info.processing_status = "completed"
    mock_personal_info.user_id = "test-user-123"
    
    # Configure mock methods
    mock_service.get_user_resume.return_value = mock_resume
    mock_service.get_personal_info_document.return_value = mock_personal_info
    mock_service.save_resume_document.return_value = 1
    mock_service.save_personal_info_document.return_value = 1
    
    return mock_service

# ============================================================================
# USER AND SESSION FIXTURES
# ============================================================================

@pytest.fixture
def test_user_data():
    """Test user data for registration/login"""
    return {
        "email": f"test.user.{int(time.time())}@example.com",
        "password": "TestPassword123!"
    }

@pytest.fixture
def mock_user():
    """Mock user object for testing"""
    mock_user = Mock(spec=User)
    mock_user.id = "test-user-123"
    mock_user.email = "test@example.com"
    mock_user.is_active = True
    mock_user.verify_password.return_value = True
    return mock_user

@pytest.fixture
def mock_session():
    """Mock user session for testing"""
    mock_session = Mock(spec=UserSession)
    mock_session.session_id = "test-session-123"
    mock_session.user_id = "test-user-123"
    mock_session.is_active = True
    mock_session.device_info = "Test Device"
    mock_session.created_at = datetime.now()
    mock_session.last_used_at = datetime.now()
    return mock_session

@pytest.fixture
def authenticated_user(api_client, test_user_data):
    """Create and authenticate a test user, return user_id and session_id"""
    # Register user
    register_response = api_client.post("/api/simple/register", json=test_user_data)
    
    if register_response.status_code == 409:
        # User already exists, try to login
        login_response = api_client.post("/api/simple/login", json=test_user_data)
        assert login_response.status_code == 200
        user_data = login_response.json()
    else:
        assert register_response.status_code == 200
        user_data = register_response.json()
    
    user_id = user_data["user_id"]
    
    # Create session
    session_response = api_client.post("/api/session/create", json={
        "user_id": user_id,
        "device_info": "Test Device"
    })
    assert session_response.status_code == 200
    session_data = session_response.json()
    
    return {
        "user_id": user_id,
        "session_id": session_data["session_id"],
        "email": test_user_data["email"]
    }

# ============================================================================
# PERFORMANCE AND TIMING FIXTURES
# ============================================================================

@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests"""
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
        
        def start_timer(self, name: str):
            self.start_times[name] = time.time()
        
        def end_timer(self, name: str):
            if name in self.start_times:
                duration = time.time() - self.start_times[name]
                self.metrics[name] = duration
                return duration
            return None
        
        def get_metrics(self) -> Dict[str, float]:
            return self.metrics.copy()
    
    return PerformanceTracker()

# ============================================================================
# TEST DATA AND UTILITIES
# ============================================================================

@pytest.fixture
def test_form_fields():
    """Common form fields for testing"""
    return [
        {
            "label": "What programming languages do you know?",
            "field_type": "text",
            "expected_keywords": ["Python", "JavaScript", "TypeScript"]
        },
        {
            "label": "What is your work experience?",
            "field_type": "textarea",
            "expected_keywords": ["5 years", "Tech Corp", "developer"]
        },
        {
            "label": "What is your email address?",
            "field_type": "email",
            "expected_keywords": ["john.doe@email.com"]
        }
    ]

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    mock_redis = Mock()
    mock_redis.ping.return_value = True
    mock_redis.set.return_value = True
    mock_redis.get.return_value = None
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    return mock_redis

@pytest.fixture
def test_report_data():
    """Sample test report data structure"""
    return {
        "test_run_summary": {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": 0.0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "success_rate": 0.0
        },
        "test_results": {},
        "performance_metrics": {},
        "system_info": {
            "python_version": sys.version,
            "pytest_version": pytest.__version__
        }
    }

# ============================================================================
# EVENT LOOP FIXTURE FOR ASYNC TESTS
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test"""
    yield
    
    # Clean up any test database files
    test_files = [
        "test_database.db",
        "test_database.db-journal",
        "test_*.json",
        "temp_*.txt"
    ]
    
    for pattern in test_files:
        import glob
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except OSError:
                pass

# ============================================================================
# PYTEST HOOKS FOR REPORTING
# ============================================================================

@pytest.fixture(autouse=True)
def test_execution_tracker(request):
    """Track test execution for reporting"""
    start_time = time.time()
    
    yield
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Store execution time as a custom attribute
    request.node.execution_time = execution_time

def pytest_runtest_makereport(item, call):
    """Generate custom test reports"""
    if call.when == "call":
        # Add execution time to the report
        execution_time = getattr(item, 'execution_time', 0)
        if hasattr(item, 'execution_time'):
            # You can add custom reporting logic here
            pass
