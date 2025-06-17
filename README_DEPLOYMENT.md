# 🚀 AI Form Filler - PostgreSQL Deployment Guide

## Quick Start (5 minutes)

### 1. Setup PostgreSQL Database

```bash
# Create PostgreSQL role and database
sudo -u postgres psql
CREATE ROLE ai_popup WITH LOGIN PASSWORD 'Erlan1824';
ALTER ROLE ai_popup CREATEDB;
CREATE DATABASE ai_popup OWNER ai_popup;
\q

# Test connection
psql -U ai_popup -d ai_popup -h localhost
```

### 2. Setup Application

```bash
# Clone and setup
git clone <your-repo>
cd backend_ai_popup

# Run deployment setup
python deploy.py
```

### 3. Configure API Keys

Edit `.env` file:

```env
OPENAI_API_KEY=sk-your-actual-openai-key
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this
DATABASE_URL=postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup
```

### 4. Initialize Database Tables

```bash
# Test PostgreSQL connection and create tables
python test_postgres_connection.py

# Or manually create tables
python create_tables.py
```

### 5. Run Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Test API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔐 Authentication Endpoints

### Register User

```bash
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

### Login User

```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

### Use with Authentication (Required for Production)

```bash
# Main endpoint - requires authentication for user-specific data
curl -X POST "http://localhost:8000/api/generate-field-answer" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{"label": "Your Name:", "url": "https://example.com"}'
```

### Demo Endpoint (No Authentication)

```bash
# Demo endpoint - uses default user data for testing
curl -X POST "http://localhost:8000/api/demo/generate-field-answer" \
  -H "Content-Type: application/json" \
  -d '{"label": "Your Name:", "url": "https://example.com"}'
```

## 🗄️ Database Schema

The application creates the following PostgreSQL tables:

- **users** - User authentication and profiles
- **user_tokens** - JWT token management
- **user_sessions** - Simple session management
- **resume_documents** - Resume file storage (binary + metadata)
- **personal_info_documents** - Personal info file storage
- **document_processing_logs** - Processing history and logs
- **forms** - Form URL tracking and application status

## 🌐 Production Deployment

### Option 1: Railway with PostgreSQL

1. Connect GitHub repo to Railway
2. Add PostgreSQL database addon
3. Set environment variables in Railway dashboard
4. Deploy automatically

### Option 2: Heroku with PostgreSQL

1. Create Heroku app
2. Add Heroku Postgres addon
3. Set config vars (environment variables)
4. Deploy with Git

### Option 3: DigitalOcean App Platform

1. Connect GitHub repo
2. Add PostgreSQL database
3. Configure environment variables
4. Deploy

### Option 4: VPS (Ubuntu) with PostgreSQL

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib python3 python3-pip

# Setup PostgreSQL
sudo -u postgres psql
CREATE ROLE ai_popup WITH LOGIN PASSWORD 'your-secure-password';
ALTER ROLE ai_popup CREATEDB;
CREATE DATABASE ai_popup OWNER ai_popup;
\q

# Clone and setup app
git clone <your-repo>
cd backend_ai_popup
python3 deploy.py

# Install PM2 for process management
npm install -g pm2

# Run with PM2
pm2 start "python -m uvicorn main:app --host 0.0.0.0 --port 8000" --name ai-form-filler

# Setup nginx reverse proxy (optional)
sudo apt install nginx
# Configure nginx to proxy to localhost:8000
```

## 📊 Features

✅ **User Authentication** (JWT tokens + simple sessions)  
✅ **Smart Form Filling** (3-tier AI system)  
✅ **Performance Optimization** (caching, early exit)  
✅ **Multi-user Support** (isolated data per user)  
✅ **PostgreSQL Database** (production-ready)  
✅ **Professional Answers** (job-focused AI responses)  
✅ **File Upload System** (resume + personal info)  
✅ **Vector Embeddings** (AI-powered document search)

## 🔧 Environment Variables

| Variable         | Description                     | Required | Default                                                   |
| ---------------- | ------------------------------- | -------- | --------------------------------------------------------- |
| `OPENAI_API_KEY` | OpenAI API key for AI responses | Yes      | -                                                         |
| `JWT_SECRET_KEY` | Secret key for JWT tokens       | Yes      | -                                                         |
| `DATABASE_URL`   | PostgreSQL connection string    | Yes      | `postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup` |

## 🚀 Ready for Production!

This system is designed to be **production-ready** with PostgreSQL as the database backend. The migration from Supabase to PostgreSQL provides:

- **Better Performance** - Local database queries
- **Full Control** - No external service dependencies
- **Cost Effective** - No monthly Supabase fees
- **Scalable** - PostgreSQL handles high loads well
- **Reliable** - Battle-tested database system

**Start getting users immediately!** 🎯

## 🔄 Migration from Supabase

If you're migrating from Supabase:

1. **Backup your Supabase data** (if any)
2. **Update environment variables** to use PostgreSQL
3. **Run migration script**: `python test_postgres_connection.py`
4. **Test all endpoints** to ensure everything works
5. **Update deployment configurations** to use PostgreSQL

The application now uses **PostgreSQL natively** instead of Supabase's REST API, providing better performance and control.
