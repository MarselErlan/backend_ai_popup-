# 🔍 Real-Time Usage Analyzer

A comprehensive tool to analyze your FastAPI application usage in real-time, tracking which functions, classes, and endpoints are being used and which ones are not.

## 🚀 Features

- **Real-time endpoint monitoring** - Tracks API endpoint usage, response times, and status codes
- **Code discovery** - Automatically discovers all functions, classes, and endpoints in your project
- **Usage tracking** - Identifies which code is being used and which is not
- **Performance metrics** - Monitors response times and system metrics
- **Beautiful reports** - Generates HTML and JSON reports with detailed analytics
- **Database storage** - Stores analysis data in SQLite for historical tracking

## 📋 Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install requests psutil
```

## 🎯 Quick Start

### 1. Start Your FastAPI Application

First, make sure your FastAPI app is running:

```bash
uvicorn main:app --reload
```

### 2. Start Real-Time Monitoring

In a new terminal, start the analyzer:

```bash
python tests/analysis/realtime_usage_analyzer.py --monitor
```

### 3. Use Your Application

Now use your application normally:

- Visit your web interface
- Make API calls
- Use all the features you want to test

The analyzer will automatically:

- ✅ Test common endpoints
- 📊 Track response times
- 📈 Monitor system metrics
- 🔍 Record usage patterns

### 4. Stop and Generate Report

Press `Ctrl+C` to stop monitoring. The analyzer will automatically generate:

- 📄 **JSON Report**: `tests/reports/realtime_analysis_report_YYYYMMDD_HHMMSS.json`
- 🌐 **HTML Report**: `tests/reports/realtime_analysis_report_YYYYMMDD_HHMMSS.html`

## 📊 What You'll Get

### Real-Time Monitoring

```
🚀 Starting real-time monitoring...
📡 Monitoring API at http://localhost:8000
💡 Make sure your FastAPI app is running with: uvicorn main:app --reload
✅ Monitoring started! Use your application now...
📈 Status: {'duration_min': 2.1, 'endpoints_tested': 5, 'total_calls': 15, 'active_endpoints': 5, 'unused_endpoints': 12}
📊 System - CPU: 15.2%, Memory: 67.8%
```

### Comprehensive Reports

The HTML report includes:

1. **📊 Overview**

   - Total API calls made
   - Endpoints tested vs discovered
   - Functions and classes discovered
   - Unused code statistics

2. **📈 Code Coverage**

   - Endpoint coverage percentage
   - Visual metrics with color coding

3. **🌐 Endpoint Usage Details**

   - Response times for each endpoint
   - Status codes (200, 404, 500, etc.)
   - Call frequency and timing

4. **🚫 Unused Code Analysis**

   - Lists of unused endpoints
   - Recommendations for cleanup

5. **⚡ Performance Metrics**
   - Most/least used endpoints
   - Average response times
   - System resource usage

## 🛠️ Command Line Options

```bash
# Start monitoring with custom host/port
python tests/analysis/realtime_usage_analyzer.py --monitor --host localhost --port 8000

# Generate report from existing data
python tests/analysis/realtime_usage_analyzer.py --report

# Show help
python tests/analysis/realtime_usage_analyzer.py
```

## 📁 Output Files

All reports are saved to `tests/reports/`:

```
tests/reports/
├── realtime_analysis_report_20241223_143022.json    # Raw data
├── realtime_analysis_report_20241223_143022.html    # Visual report
├── realtime_analysis.db                             # SQLite database
└── logs/
    └── realtime_analyzer.log                        # Analysis logs
```

## 🎨 Example Use Cases

### 1. Code Cleanup

Find and remove unused endpoints:

```bash
# Start monitoring
python tests/analysis/realtime_usage_analyzer.py --monitor

# Use your app for 5-10 minutes
# Stop with Ctrl+C

# Check the HTML report for unused endpoints
# Remove or refactor unused code
```

### 2. Performance Analysis

Identify slow endpoints:

```bash
# Monitor during heavy usage
# Check "Performance Summary" in the HTML report
# Optimize slow endpoints
```

### 3. API Testing Coverage

Ensure all endpoints are tested:

```bash
# Run your test suite while monitoring
# Check endpoint coverage percentage
# Add tests for uncovered endpoints
```

## 🔧 Customization

### Adding More Endpoints to Test

Edit the `test_endpoints` list in the `_monitor_api_endpoints` method:

```python
test_endpoints = [
    ("/", "GET"),
    ("/health", "GET"),
    ("/docs", "GET"),
    ("/your-custom-endpoint", "GET"),
    ("/api/v2/new-feature", "POST"),
]
```

### Changing Monitoring Frequency

Adjust the sleep intervals in the monitoring methods:

```python
time.sleep(15)  # Check every 15 seconds (default)
time.sleep(5)   # Check every 5 seconds (more frequent)
```

## 🐛 Troubleshooting

### "API not available" messages

- Make sure your FastAPI app is running on the correct host:port
- Check if the app is accessible at `http://localhost:8000`

### No endpoints discovered

- Ensure your FastAPI routes use standard decorators (`@app.get`, `@app.post`, etc.)
- Check that the analyzer can read your Python files

### Database errors

- Make sure the `tests/reports/` directory is writable
- Delete `realtime_analysis.db` if it gets corrupted

## 📈 Integration with CI/CD

You can integrate this into your CI/CD pipeline:

```bash
# In your CI script
uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 5  # Wait for app to start

# Run your tests while monitoring
timeout 300 python tests/analysis/realtime_usage_analyzer.py --monitor &
python -m pytest tests/

# The monitor will automatically generate reports
```

## 🤝 Contributing

Feel free to enhance the analyzer by:

- Adding more monitoring capabilities
- Improving the report visualization
- Adding support for other frameworks
- Enhancing the code discovery logic

---

**Happy analyzing! 🔍📊**
