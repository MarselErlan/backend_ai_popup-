# 🚀 AI Form Filler - Simple Deployment Guide

## Quick Start (5 minutes)

### 1. Setup

```bash
# Clone and setup
git clone <your-repo>
cd backend_ai_popup

# Run deployment setup
python deploy.py
```

### 2. Configure API Keys

Edit `.env` file:

```env
OPENAI_API_KEY=sk-your-actual-openai-key
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this
DATABASE_URL=sqlite:///./users.db
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
```

### 3. Run Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Test API

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

## 🌐 Production Deployment

### Option 1: Railway

1. Connect GitHub repo to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically

### Option 2: Heroku

1. Create Heroku app
2. Set config vars (environment variables)
3. Deploy with Git

### Option 3: DigitalOcean App Platform

1. Connect GitHub repo
2. Configure environment variables
3. Deploy

### Option 4: VPS (Ubuntu)

```bash
# Install Python 3.9+
sudo apt update
sudo apt install python3 python3-pip

# Clone repo
git clone <your-repo>
cd backend_ai_popup

# Setup
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

✅ **User Authentication** (JWT tokens)
✅ **Smart Form Filling** (3-tier AI system)
✅ **Performance Optimization** (caching, early exit)
✅ **Multi-user Support** (isolated data per user)
✅ **Simple Database** (SQLite for easy deployment)
✅ **Professional Answers** (job-focused AI responses)

## 🔧 Environment Variables

| Variable         | Description                     | Required                |
| ---------------- | ------------------------------- | ----------------------- |
| `OPENAI_API_KEY` | OpenAI API key for AI responses | Yes                     |
| `JWT_SECRET_KEY` | Secret key for JWT tokens       | Yes                     |
| `DATABASE_URL`   | Database connection string      | No (defaults to SQLite) |
| `SUPABASE_URL`   | Supabase project URL            | Yes                     |
| `SUPABASE_KEY`   | Supabase API key                | Yes                     |

## 🚀 Ready for Production!

This system is designed to be **simple to deploy** while being **powerful enough for real users**. The authentication system is minimal but secure, and the AI form filling is optimized for speed and accuracy.

**Start getting users immediately!** 🎯
