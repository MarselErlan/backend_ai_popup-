# 🚀 Railway Deployment Guide

## Overview

This guide will help you deploy your Smart Form Fill API to Railway with PostgreSQL and Redis.

## Prerequisites

- Railway account: https://railway.app
- GitHub repository with your code
- OpenAI API key

## Step 1: Create Railway Project

1. Go to https://railway.app and sign in
2. Click "New Project"
3. Connect your GitHub repository
4. Select this repository (`backend_ai_popup`)

## Step 2: Add PostgreSQL Database

1. In your Railway project dashboard, click "New Service"
2. Select "Database" → "PostgreSQL"
3. Railway will automatically create the database and provide `DATABASE_URL`

## Step 3: Add Redis (Optional)

1. In your Railway project dashboard, click "New Service"
2. Select "Database" → "Redis"
3. Railway will automatically create Redis and provide `REDIS_URL`

## Step 4: Configure Environment Variables

In your Railway project dashboard, go to your service → "Variables" tab:

```bash
# Required
DATABASE_URL=postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqTpIKa@postgres.railway.internal:5432/railway
OPENAI_API_KEY=your-openai-api-key-here

# Optional (Railway provides Redis)
REDIS_URL=redis://default:password@redis.railway.internal:6379

# Auto-provided by Railway
PORT=8000
```

## Step 5: Deploy

1. Railway will automatically detect your Python app
2. It will install dependencies from `requirements.txt`
3. It will start your app using the command in `Procfile`

## Step 6: Initialize Database

After deployment, you may need to initialize the database:

```bash
# Run this in Railway's terminal or locally with production DATABASE_URL
python railway_setup.py
```

## Step 7: Test Your API

Your API will be available at: `https://your-app-name.railway.app`

Test endpoints:

- Health check: `GET /health`
- API docs: `GET /docs`
- Generate field answer: `POST /api/generate-field-answer`

## Environment Variables Reference

### Required Variables

- `DATABASE_URL`: PostgreSQL connection string (provided by Railway)
- `OPENAI_API_KEY`: Your OpenAI API key

### Optional Variables

- `REDIS_URL`: Redis connection string (defaults to `redis://localhost:6379`)
- `PORT`: Port to run the server on (provided by Railway)

## Troubleshooting

### Database Connection Issues

If you get database connection errors:

1. Check that `DATABASE_URL` is correctly set
2. Ensure the database is running in Railway
3. Run the setup script: `python railway_setup.py`

### Redis Connection Issues

If Redis is not available:

1. The app will fall back to local Redis
2. Add Redis service in Railway for better performance
3. Set `REDIS_URL` environment variable

### OpenAI API Issues

If you get OpenAI API errors:

1. Check that `OPENAI_API_KEY` is correctly set
2. Ensure your OpenAI account has sufficient credits
3. Check API key permissions

## Production Optimizations

### Database

- Railway PostgreSQL is production-ready
- Automatic backups are enabled
- Connection pooling is handled

### Redis

- Railway Redis is production-ready
- Persistence is enabled
- Memory optimization is automatic

### API

- CORS is configured for production
- Error handling is production-ready
- Logging is configured

## Monitoring

Railway provides:

- Real-time logs
- Resource usage metrics
- Uptime monitoring
- Alerts

## Scaling

Railway supports:

- Horizontal scaling (multiple instances)
- Vertical scaling (more CPU/RAM)
- Auto-scaling based on load

## Security

- Environment variables are encrypted
- HTTPS is enabled by default
- Database connections are secure
- API keys are handled securely

## Commands

```bash
# Test locally with production database
export DATABASE_URL="postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqTpIKa@postgres.railway.internal:5432/railway"
export OPENAI_API_KEY="your-key-here"
python main.py

# Setup production environment
python railway_setup.py

# Run migrations
python create_tables.py

# Test API
curl https://your-app-name.railway.app/health
```

## Next Steps

1. Set up monitoring and alerts
2. Configure custom domain (optional)
3. Set up CI/CD pipeline
4. Add rate limiting
5. Set up backup strategies
