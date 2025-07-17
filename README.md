# Smart Form Fill API v4.0

Field-by-Field Intelligent Form Filling with Vector Database Integration

## Overview

This API has been completely redesigned to work with individual form fields instead of entire HTML pages. It's designed to integrate with React frontends that can send field labels one-by-one and receive intelligent answers.

## Key Features

- **🎯 Field-by-Field Processing**: Send one field label at a time, get intelligent answers
- **🧠 3-Tier Data Retrieval**:
  1. Resume vector database (professional info)
  2. Personal info vector database (personal details)
  3. AI generation (when data is insufficient)
- **📄 Database-Driven**: No more file system dependencies, everything stored in database
- **🚀 Simple Integration**: Easy to integrate with React frontends
- **👥 Multi-User Support**: User ID support for multiple users

## Main API Endpoint

### POST `/api/generate-field-answer`

Generate intelligent answer for a specific form field.

**Request:**

```json
{
  "label": "What's your occupation?",
  "url": "https://example.com/job-application",
  "user_id": "user123"
}
```

**Response:**

```json
{
  "answer": "Software Engineer",
  "data_source": "resume_vectordb",
  "reasoning": "Found occupation information from resume vector database",
  "status": "success"
}
```

## Frontend Integration Example

```javascript
const fieldLabel = getFieldLabel(currentInput); // e.g. "What's your occupation?"
const pageUrl = window.location.href;

fetch("http://localhost:8000/api/generate-field-answer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    label: fieldLabel,
    url: pageUrl,
    user_id: "your-user-id", // optional
  }),
})
  .then((res) => res.json())
  .then((data) => {
    currentInput.value = data.answer || "⚠️ No answer returned";
  })
  .catch((err) => {
    console.error("Error calling backend:", err);
    currentInput.value = "⚠️ Backend error";
  });
```

## Vector Database Management

### Resume Data

```bash
POST /api/v1/resume/reembed?user_id=your-user-id
```

### Personal Info Data

```bash
POST /api/v1/personal-info/reembed?user_id=your-user-id
```

### Check Document Status

```bash
GET /api/v1/documents/status?user_id=your-user-id
```

## How It Works

1. **Field Analysis**: When you send a field label, the system analyzes what type of information is needed
2. **Smart Search**: It searches your resume and personal info vector databases for relevant information
3. **Intelligent Generation**: If no relevant data is found, it generates professional, contextually appropriate answers
4. **Response**: Returns the best answer along with the data source and reasoning

## Data Sources Priority

1. **Resume Vector Database** (highest priority)

   - Professional experience, skills, education
   - Work history, achievements, technical expertise

2. **Personal Info Vector Database** (second priority)

   - Contact details, work authorization, salary expectations
   - Location preferences, personal information

3. **AI Generation** (fallback)
   - Professional content generation when data is missing
   - Context-aware answers based on job requirements

## Setup

1. **Install Redis Stack (required for vector storage):**

```bash
# On macOS with Homebrew
brew tap redis-stack/redis-stack
brew install redis-stack/redis-stack/redis-stack
redis-stack-server --daemonize yes

# On Ubuntu/Debian
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
sudo apt-get update
sudo apt-get install redis-stack-server
sudo systemctl start redis-stack-server
sudo systemctl enable redis-stack-server

# On Windows
# Download Redis Stack from: https://redis.io/download#redis-stack
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set environment variables:**

```bash
OPENAI_API_KEY=your-openai-api-key
DATABASE_URL=your-database-connection-string
REDIS_URL=redis://localhost:6379  # Optional, defaults to localhost:6379
```

4. **Upload your documents to the database (resume and personal info)**

5. **Run the server:**

```bash
python main.py
```

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Changes from v3.0

- ❌ **Removed**: Complex HTML form analysis
- ❌ **Removed**: File system dependencies
- ✅ **Added**: Simple field-by-field API
- ✅ **Added**: Direct React frontend integration
- ✅ **Simplified**: Single endpoint for field answers
- ✅ **Preserved**: All intelligent data retrieval logic
- ✅ **Preserved**: Redis vector storage for fast document search
