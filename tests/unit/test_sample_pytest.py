"""
🧪 Sample Pytest Test
Demonstrates pytest structure with fixtures, markers, and best practices
"""

import pytest
import time
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch


class TestSampleUnit:
    """Sample unit tests demonstrating pytest best practices"""
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic functionality"""
        # Arrange
        test_data = "Hello, World!"
        
        # Act
        result = test_data.upper()
        
        # Assert
        assert result == "HELLO, WORLD!"
        assert isinstance(result, str)
    
    @pytest.mark.unit
    def test_with_fixture(self, sample_resume_text):
        """Test using a fixture from conftest.py"""
        # Arrange & Act
        lines = sample_resume_text.strip().split('\n')
        
        # Assert
        assert len(lines) > 0
        assert "John Doe" in sample_resume_text
        assert "Software Engineer" in sample_resume_text
    
    @pytest.mark.unit
    def test_mock_service(self, mock_llm_service):
        """Test using mock service fixture"""
        # Arrange
        test_query = "What is your experience?"
        
        # Act - call the mock service
        # Note: In real tests, this would be called by the code under test
        mock_result = mock_llm_service.generate_field_answer.return_value
        
        # Assert
        assert mock_result["status"] == "success"
        assert "Mock answer" in mock_result["answer"]
        assert mock_result["performance_metrics"]["processing_time_seconds"] == 0.1
    
    @pytest.mark.unit
    def test_temp_file_fixture(self, sample_text_file):
        """Test using temporary file fixture"""
        # Arrange & Act
        content = sample_text_file.read_text()
        
        # Assert
        assert sample_text_file.exists()
        assert "John Doe" in content
        assert sample_text_file.suffix == ".txt"
    
    @pytest.mark.unit
    def test_performance_tracking(self, performance_tracker):
        """Test performance tracking fixture"""
        # Arrange
        operation_name = "test_operation"
        
        # Act
        performance_tracker.start_timer(operation_name)
        time.sleep(0.1)  # Simulate some work
        duration = performance_tracker.end_timer(operation_name)
        
        # Assert
        assert duration is not None
        assert duration >= 0.1
        assert operation_name in performance_tracker.get_metrics()
    
    @pytest.mark.unit
    @pytest.mark.parametrize("input_text,expected_length", [
        ("short", 5),
        ("medium length text", 18),
        ("", 0),
    ])
    def test_parametrized(self, input_text, expected_length):
        """Test with parametrized inputs"""
        # Act
        result = len(input_text)
        
        # Assert
        assert result == expected_length
    
    @pytest.mark.unit
    def test_exception_handling(self):
        """Test exception handling"""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="invalid literal"):
            int("not_a_number")
    
    @pytest.mark.unit
    @patch('time.time')
    def test_with_mock_patch(self, mock_time):
        """Test using mock patch decorator"""
        # Arrange
        mock_time.return_value = 1234567890.0
        
        # Act
        result = time.time()
        
        # Assert
        assert result == 1234567890.0
        mock_time.assert_called_once()


class TestSampleIntegration:
    """Sample integration tests (would be in integration/ folder normally)"""
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_api_health_check(self, api_client):
        """Test API health check endpoint"""
        # Act
        response = api_client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_root_endpoint(self, api_client):
        """Test root endpoint"""
        # Act
        response = api_client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "Smart Form Fill API" in data["message"]
        assert "version" in data


class TestSampleAsync:
    """Sample async tests"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test async functionality"""
        # Arrange
        async def sample_async_function():
            await asyncio.sleep(0.01)
            return "async result"
        
        # Act
        result = await sample_async_function()
        
        # Assert
        assert result == "async result"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_async_service(self, mock_llm_service):
        """Test async mock service"""
        # Arrange
        test_field = "What programming languages do you know?"
        test_user_id = "test-user-123"
        
        # Act
        result = await mock_llm_service.generate_field_answer(
            field_label=test_field,
            user_id=test_user_id,
            field_context={}
        )
        
        # Assert
        assert result["status"] == "success"
        assert "Mock answer" in result["answer"]
        mock_llm_service.generate_field_answer.assert_called_once_with(
            field_label=test_field,
            user_id=test_user_id,
            field_context={}
        )


class TestSampleSlow:
    """Sample slow tests for performance testing"""
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_slow_operation(self, performance_tracker):
        """Test that takes longer time (marked as slow)"""
        # Arrange
        operation_name = "slow_operation"
        
        # Act
        performance_tracker.start_timer(operation_name)
        time.sleep(2)  # Simulate slow operation
        duration = performance_tracker.end_timer(operation_name)
        
        # Assert
        assert duration >= 2.0
        assert duration < 3.0  # Should not be too slow
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage (example)"""
        # Arrange
        large_data = []
        
        # Act
        for i in range(10000):
            large_data.append(f"item_{i}")
        
        # Assert
        assert len(large_data) == 10000
        assert sys.getsizeof(large_data) > 0


# Example of test class with setup/teardown
class TestWithSetupTeardown:
    """Test class demonstrating setup and teardown"""
    
    def setup_method(self):
        """Setup before each test method"""
        self.test_data = {"initialized": True}
    
    def teardown_method(self):
        """Cleanup after each test method"""
        self.test_data = None
    
    @pytest.mark.unit
    def test_setup_data(self):
        """Test that setup data is available"""
        assert self.test_data is not None
        assert self.test_data["initialized"] is True
    
    @pytest.mark.unit
    def test_modify_data(self):
        """Test modifying setup data"""
        # Act
        self.test_data["modified"] = True
        
        # Assert
        assert self.test_data["modified"] is True


# Example of test fixtures at module level
@pytest.fixture
def module_specific_fixture():
    """Fixture specific to this test module"""
    return {"module": "test_sample_pytest", "data": "module_specific"}


class TestModuleFixture:
    """Test using module-specific fixture"""
    
    @pytest.mark.unit
    def test_module_fixture(self, module_specific_fixture):
        """Test using module-specific fixture"""
        assert module_specific_fixture["module"] == "test_sample_pytest"
        assert module_specific_fixture["data"] == "module_specific"


# Skip tests conditionally
@pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
@pytest.mark.unit
def test_unix_specific():
    """Test that only runs on Unix systems"""
    assert "/" in str(Path("/tmp"))


# Expected failure
@pytest.mark.xfail(reason="Known issue - will be fixed in next version")
@pytest.mark.unit
def test_expected_failure():
    """Test that is expected to fail"""
    assert False, "This test is expected to fail" 