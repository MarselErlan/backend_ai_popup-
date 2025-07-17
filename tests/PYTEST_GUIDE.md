# 🧪 Pytest Testing Guide

## 🚀 Quick Start

### Installation

```bash
# Install testing dependencies
pip install -r requirements-test.txt

# Or install core pytest packages
pip install pytest pytest-cov pytest-html pytest-asyncio
```

### Basic Usage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific category
pytest -m unit
pytest -m integration
pytest -m e2e

# Run quick tests (exclude slow ones)
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

## 📁 Test Organization

### Folder Structure

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── pytest.ini          # Pytest configuration
├── unit/               # Unit tests (@pytest.mark.unit)
├── integration/        # Integration tests (@pytest.mark.integration)
├── e2e/               # End-to-end tests (@pytest.mark.e2e)
├── performance/       # Performance tests (@pytest.mark.performance)
├── fixtures/          # Test data files
└── reports/           # Generated reports
```

### Test Markers

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.e2e          # End-to-end tests
@pytest.mark.performance  # Performance tests
@pytest.mark.analysis     # Code analysis tests
@pytest.mark.slow         # Tests taking >5 seconds
@pytest.mark.api          # Tests requiring API server
@pytest.mark.redis        # Tests requiring Redis
@pytest.mark.db           # Tests requiring database
```

## 🔧 Using Fixtures

### Built-in Fixtures (from conftest.py)

#### API Testing

```python
def test_api_endpoint(api_client):
    """Test using API client fixture"""
    response = api_client.get("/health")
    assert response.status_code == 200
```

#### Mock Services

```python
def test_llm_service(mock_llm_service):
    """Test using mock LLM service"""
    result = mock_llm_service.generate_field_answer.return_value
    assert result["status"] == "success"
```

#### File Handling

```python
def test_file_processing(sample_text_file, temp_directory):
    """Test using file fixtures"""
    content = sample_text_file.read_text()
    assert "John Doe" in content

    # Create additional files in temp directory
    new_file = temp_directory / "test.txt"
    new_file.write_text("test content")
```

#### Database Testing

```python
def test_database(temp_db, mock_db_session):
    """Test using database fixtures"""
    # temp_db provides a clean SQLite database
    # mock_db_session provides a mocked session
    pass
```

#### Performance Tracking

```python
def test_performance(performance_tracker):
    """Test with performance tracking"""
    performance_tracker.start_timer("operation")
    # ... do work ...
    duration = performance_tracker.end_timer("operation")
    assert duration < 1.0  # Should complete in under 1 second
```

## 🎯 Test Categories and Commands

### Unit Tests

```bash
# Run all unit tests
pytest -m unit

# Run unit tests with coverage
pytest -m unit --cov=app --cov-report=term-missing

# Run specific unit test file
pytest tests/unit/test_sample_pytest.py

# Run specific test method
pytest tests/unit/test_sample_pytest.py::TestSampleUnit::test_basic_functionality
```

### Integration Tests

```bash
# Run integration tests (requires server)
pytest -m integration

# Run integration tests without server check
python tests/scripts/run_all_tests.py --category integration --no-server-check
```

### End-to-End Tests

```bash
# Run E2E tests
pytest -m e2e

# Run E2E tests with detailed output
pytest -m e2e -v -s
```

### Performance Tests

```bash
# Run performance tests
pytest -m performance

# Run with benchmark output
pytest -m performance --benchmark-only
```

## 📊 Coverage Reporting

### Generate Coverage Reports

```bash
# HTML coverage report
pytest --cov=app --cov-report=html
# Report saved to htmlcov/index.html

# Terminal coverage report
pytest --cov=app --cov-report=term-missing

# JSON coverage report
pytest --cov=app --cov-report=json
# Report saved to coverage.json

# XML coverage report (for CI)
pytest --cov=app --cov-report=xml
```

### Coverage Configuration

```ini
# In pytest.ini or .coveragerc
[coverage:run]
source = app
omit =
    */tests/*
    */venv/*
    */migrations/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## 🏗️ CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt

      - name: Run tests with coverage
        run: |
          pytest --cov=app --cov-report=xml --junit-xml=junit.xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
```

### Using the Test Runner

```bash
# CI mode with full reporting
python tests/scripts/run_all_tests.py --ci

# Quick tests for development
python tests/scripts/run_all_tests.py --quick

# Specific category with coverage
python tests/scripts/run_all_tests.py --category unit --coverage
```

## 🔍 Advanced Features

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("", ""),
])
def test_uppercase(input, expected):
    assert input.upper() == expected
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Test Fixtures with Scope

```python
@pytest.fixture(scope="session")
def expensive_resource():
    """Created once per test session"""
    return create_expensive_resource()

@pytest.fixture(scope="function")  # Default
def fresh_data():
    """Created for each test function"""
    return create_fresh_data()
```

### Conditional Tests

```python
@pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
def test_unix_feature():
    pass

@pytest.mark.xfail(reason="Known bug - fix in progress")
def test_known_issue():
    assert False  # Expected to fail
```

### Custom Markers

```python
# In pytest.ini
markers =
    slow: marks tests as slow
    external: marks tests that require external services

# In test files
@pytest.mark.slow
@pytest.mark.external
def test_external_api():
    pass
```

## 📈 Reporting and Output

### HTML Reports

```bash
# Generate HTML report
pytest --html=tests/reports/report.html --self-contained-html
```

### JUnit XML (for CI)

```bash
# Generate JUnit XML for CI systems
pytest --junit-xml=tests/reports/junit.xml
```

### Custom Test Reports

```bash
# Using our custom test runner
python tests/scripts/run_all_tests.py --ci
# Generates multiple report formats in tests/reports/
```

## 🛠️ Debugging Tests

### Verbose Output

```bash
# Verbose output with test names
pytest -v

# Extra verbose with print statements
pytest -v -s

# Show local variables on failure
pytest --tb=long

# Drop into debugger on failure
pytest --pdb
```

### Filtering Tests

```bash
# Run tests matching pattern
pytest -k "test_api"

# Run tests NOT matching pattern
pytest -k "not slow"

# Combine markers and keywords
pytest -m "unit and not slow" -k "test_basic"
```

### Test Discovery

```bash
# Show what tests would be collected
pytest --collect-only

# Show test markers
pytest --markers
```

## 📝 Writing Good Tests

### Test Structure (Arrange-Act-Assert)

```python
def test_function():
    # Arrange - Set up test data
    user_data = {"name": "John", "age": 30}

    # Act - Perform the action
    result = process_user(user_data)

    # Assert - Check the result
    assert result["status"] == "success"
    assert result["processed_name"] == "JOHN"
```

### Test Naming Convention

```python
def test_should_return_success_when_valid_input():
    """Test names should be descriptive"""
    pass

def test_raises_error_when_invalid_email():
    """Test error conditions"""
    with pytest.raises(ValueError):
        validate_email("invalid-email")
```

### Using Fixtures Effectively

```python
@pytest.fixture
def user_service(mock_db_session):
    """Compose fixtures to build complex test scenarios"""
    return UserService(db_session=mock_db_session)

def test_user_creation(user_service, test_user_data):
    """Use multiple fixtures in tests"""
    user = user_service.create_user(test_user_data)
    assert user.email == test_user_data["email"]
```

## 🔧 Configuration Files

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
addopts = --strict-markers --tb=short --maxfail=10
```

### conftest.py Best Practices

- Put shared fixtures in conftest.py
- Use appropriate fixture scopes
- Clean up resources in fixtures
- Provide meaningful fixture names
- Document complex fixtures

## 🎯 Common Patterns

### Testing Exceptions

```python
def test_raises_specific_error():
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_raises("bad_input")
```

### Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_endpoint(api_client):
    response = await api_client.async_get("/endpoint")
    assert response.status_code == 200
```

### Mocking External Services

```python
@patch('app.services.external_api.requests.get')
def test_external_service(mock_get, api_client):
    mock_get.return_value.json.return_value = {"status": "ok"}
    response = api_client.get("/api/external")
    assert response.status_code == 200
```

## 📊 Performance Testing

### Using pytest-benchmark

```python
def test_performance_benchmark(benchmark):
    result = benchmark(expensive_function, arg1, arg2)
    assert result is not None
```

### Memory Usage Testing

```python
import psutil
import os

def test_memory_usage():
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Perform memory-intensive operation
    large_data = create_large_dataset()

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    # Assert memory usage is reasonable
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

---

## 🎉 Summary

This pytest setup provides:

- **🏗️ Professional structure** with organized folders and markers
- **🔧 Comprehensive fixtures** for all testing scenarios
- **📊 Multiple reporting formats** (HTML, XML, JSON, coverage)
- **⚡ Performance tracking** and benchmarking
- **🚀 CI/CD ready** with proper exit codes and reports
- **🎯 Flexible execution** with markers and filtering
- **🧪 Best practices** for maintainable tests

Use this guide to write effective, maintainable tests that ensure your Smart Form Fill API is production-ready!
