# 📊 Integrated Usage Analysis

**Real-time analysis built directly into your FastAPI application!**

This feature provides comprehensive usage analysis without needing to run separate monitoring tools. It automatically starts when you run your FastAPI app and generates detailed reports when you stop it.

## 🚀 Quick Start

### 1. Enable Analysis

```bash
# Enable analysis
python toggle_analysis.py on

# Or manually set environment variable
export ENABLE_USAGE_ANALYSIS=true
```

### 2. Start Your App

```bash
uvicorn main:app --reload
```

### 3. Use Your Application

- Visit endpoints in your browser
- Use your frontend application
- Make API calls
- The analyzer captures everything automatically!

### 4. Generate Report

```bash
# Press Ctrl+C to stop the server
# Reports are automatically generated and saved
```

## 🎛️ Toggle Analysis On/Off

### Using the Toggle Script (Recommended)

```bash
# Enable analysis
python toggle_analysis.py on
python toggle_analysis.py enable

# Disable analysis (for production)
python toggle_analysis.py off
python toggle_analysis.py disable
```

### Using Environment Variables

```bash
# Enable
export ENABLE_USAGE_ANALYSIS=true

# Disable
export ENABLE_USAGE_ANALYSIS=false
```

### Using .env File

```bash
# Add to your .env file
ENABLE_USAGE_ANALYSIS=true   # Enable
ENABLE_USAGE_ANALYSIS=false  # Disable
```

## 📊 What Gets Analyzed

### ✅ Real-time Monitoring

- **API Endpoints**: All HTTP requests (GET, POST, PUT, DELETE, etc.)
- **Response Times**: Millisecond-precision timing
- **Status Codes**: 200, 404, 500, etc.
- **User Agents**: Browser/client information
- **IP Addresses**: Client locations

### 🔍 Code Discovery

- **Functions**: All Python functions in your codebase with signatures (392 discovered)
- **Classes**: All Python classes with methods and inheritance (80 discovered)
- **Endpoints**: All FastAPI route definitions

### ⚡ Function Tracking (NEW!)

- **Execution Monitoring**: Track function calls with timing data
- **Input/Output Analysis**: Parameter types and example values
- **Success/Error Rates**: Monitor function reliability
- **Performance Metrics**: Execution times and bottlenecks

### 🏗️ Class Usage Tracking (NEW!)

- **Instantiation Monitoring**: Track class creation and constructor arguments
- **Method Usage**: Monitor which methods are called and how often
- **Usage Patterns**: Identify most-used methods per class
- **Instance Analytics**: Analyze constructor argument patterns

### 📈 Performance Metrics

- **Call Frequency**: Which endpoints/functions/classes are used most/least
- **Response Performance**: Average response times and execution times
- **Error Rates**: Status code distributions and function success rates
- **Usage Patterns**: When endpoints are called and functions are executed

## 📁 Report Locations

Reports are saved to `tests/reports/`:

```
tests/reports/
├── integrated_analysis_current.json    # Latest analysis data
├── integrated_analysis_current.html    # Beautiful web dashboard
├── integrated_analysis_backup_*.json   # Timestamped backups
└── integrated_analysis.db             # SQLite database
```

## 🔧 Function & Class Tracking

### Using Decorators (Optional)

For detailed function and class tracking, you can use the provided decorators:

```python
from app.utils.usage_decorators import track_function, track_class

# Track function calls
@track_function
def my_function(param1: str, param2: int) -> str:
    return f"{param1}_{param2}"

# Track class usage
@track_class
class MyService:
    def __init__(self, config: str):
        self.config = config

    def process_data(self, data: list) -> dict:
        return {"processed": len(data)}
```

### Automatic Discovery

Even without decorators, the analyzer automatically discovers:

- All functions and their signatures
- All classes and their methods
- Function and class locations in your codebase

### Demo Script

Run the function/class tracking demo:

```bash
python demo_function_class_tracking.py
```

## 🌐 View Reports

### HTML Dashboard

Open `tests/reports/integrated_analysis_current.html` in your browser for a beautiful interactive dashboard with:

- 📊 Usage overview and metrics
- 🎯 Code coverage statistics
- 📈 Most used endpoints table with detailed info
- ⚡ **Most Used Functions table** (NEW!) - function name, file, calls, timing, success rate, input/output types
- 🏗️ **Most Used Classes table** (NEW!) - class name, file, instantiations, method calls, constructor args
- 🚫 Unused code lists (endpoints, functions, classes)
- 💡 Toggle instructions

### JSON Data

Use `tests/reports/integrated_analysis_current.json` for:

- Programmatic analysis
- Custom reporting
- Data integration
- API consumption
- Function and class usage data

## 🔧 API Endpoints

### Check Analysis Status

```bash
curl http://localhost:8000/api/analysis/status
```

Response:

```json
{
  "status": "success",
  "analysis": {
    "enabled": true,
    "monitoring": true,
    "duration_minutes": 15.5,
    "endpoints_tracked": 12,
    "total_calls": 156,
    "endpoints_discovered": 45,
    "functions_discovered": 369,
    "classes_discovered": 77
  },
  "toggle_info": {
    "enable": "Set environment variable ENABLE_USAGE_ANALYSIS=true",
    "disable": "Set environment variable ENABLE_USAGE_ANALYSIS=false",
    "current": "enabled"
  }
}
```

## 🎯 Use Cases

### 🧪 Development & Testing

- **Endpoint Coverage**: See which APIs you're actually testing
- **Performance Bottlenecks**: Identify slow endpoints
- **Dead Code Detection**: Find unused endpoints and functions
- **Usage Patterns**: Understand how your app is being used

### 🚀 Pre-Production Analysis

- **Load Testing Insights**: Analyze performance under load
- **Feature Usage**: See which features are popular
- **Error Analysis**: Track error rates and patterns
- **Optimization Targets**: Focus optimization efforts

### 📊 Production Monitoring (Lightweight)

- **Health Monitoring**: Basic endpoint availability
- **Performance Tracking**: Response time trends
- **Usage Analytics**: API consumption patterns

> **Note**: For production, consider disabling detailed analysis to reduce overhead

## ⚡ Performance Impact

### Minimal Overhead

- **Memory**: ~2-5MB additional memory usage
- **CPU**: <1% additional CPU usage
- **Storage**: SQLite database with automatic cleanup
- **Network**: No external dependencies

### Memory Optimization

- **Fixed Filenames**: Reports overwrite previous versions
- **Automatic Cleanup**: Old database entries are pruned
- **Backup Limits**: Only 1 most recent backup kept (maximum memory savings)
- **Efficient Storage**: Compressed JSON and optimized database
- **Manual Cleanup**: `python cleanup_old_analysis_reports.py` for manual cleanup

## 🛠️ Technical Details

### Architecture

- **Middleware**: Captures all HTTP requests/responses
- **Background Threads**: Non-blocking analysis processing
- **SQLite Database**: Persistent storage with automatic cleanup
- **Code Discovery**: AST parsing for comprehensive code analysis

### Integration Points

1. **FastAPI Lifecycle**: Starts/stops with your app
2. **Middleware Layer**: Captures all requests transparently
3. **Background Processing**: Doesn't block your application
4. **Graceful Shutdown**: Generates reports on Ctrl+C

## 🔍 Troubleshooting

### Analysis Not Starting

```bash
# Check if enabled
python toggle_analysis.py on

# Verify environment
echo $ENABLE_USAGE_ANALYSIS

# Check logs
tail -f logs/app.log | grep "📊"
```

### Reports Not Generated

- Ensure you stop with **Ctrl+C** (not kill -9)
- Check `tests/reports/` directory exists
- Verify write permissions
- Look for error messages in logs

### Performance Issues

```bash
# Disable for production
python toggle_analysis.py off

# Or set lightweight mode
export ENABLE_USAGE_ANALYSIS=false
```

## 🆚 vs. External Tools

| Feature           | Integrated Analysis | External APM           | Separate Scripts      |
| ----------------- | ------------------- | ---------------------- | --------------------- |
| **Setup**         | ✅ Zero config      | ❌ Complex setup       | ⚠️ Manual process     |
| **Performance**   | ✅ Minimal overhead | ❌ High overhead       | ⚠️ Resource intensive |
| **Integration**   | ✅ Built-in         | ❌ External dependency | ❌ Separate process   |
| **Cost**          | ✅ Free             | ❌ Often paid          | ✅ Free               |
| **Customization** | ✅ Full control     | ❌ Limited             | ✅ Full control       |
| **Production**    | ✅ Toggle on/off    | ⚠️ Always on           | ❌ Manual management  |

## 🎉 Success Stories

### Development Workflow

```bash
# Start development
python toggle_analysis.py on
uvicorn main:app --reload

# Develop and test your application
# ... use your app, test features, etc ...

# Stop and review
# Press Ctrl+C
# Open tests/reports/integrated_analysis_current.html
```

### Results You'll See

- 📊 **"Your app has 45 endpoints, but only 12 are being used!"**
- ⚡ **"POST /api/generate-field-answer takes 0.245s on average"**
- 🎯 **"87% of your functions are discovered but not directly tested"**
- 🚫 **"These 15 endpoints are never called: GET /api/old-feature"**

## 🔮 Future Enhancements

- **Function-level tracing**: Track individual function calls
- **Database query analysis**: Monitor SQL performance
- **Real-time dashboard**: Live web interface
- **Alerting**: Notifications for performance issues
- **Export formats**: CSV, Excel, PDF reports
- **Integration**: Slack/Discord notifications

## 🧹 Memory Management

### Automatic Cleanup

The analyzer automatically manages memory by:

- **Limiting backups**: Keeps only 1 most recent backup file
- **Database pruning**: Removes old entries (keeps last 1000 per endpoint)
- **Fixed filenames**: Current reports replace previous versions

### Manual Cleanup

```bash
# View current reports and backup files
python cleanup_old_analysis_reports.py --show

# Clean up old backup files (interactive)
python cleanup_old_analysis_reports.py

# Help
python cleanup_old_analysis_reports.py --help
```

### Memory Usage

- **Typical usage**: 2-5MB additional memory
- **Backup files**: ~50KB each (only 1 kept)
- **Database**: ~10-20KB with automatic cleanup
- **Total footprint**: <1MB for most applications

## 📞 Support

Need help? Check:

1. **Logs**: Look for 📊 emoji in your application logs
2. **Status API**: `GET /api/analysis/status`
3. **Toggle Script**: `python toggle_analysis.py --help`
4. **Environment**: Verify `ENABLE_USAGE_ANALYSIS` setting
5. **Cleanup**: `python cleanup_old_analysis_reports.py --show`

---

**🎯 Ready to get insights into your FastAPI application? Start with `python toggle_analysis.py on`!**
