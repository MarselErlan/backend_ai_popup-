#!/usr/bin/env python3

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 Environment Variable Debug")
print("=" * 50)

# Check all environment variables
openai_key = os.getenv("OPENAI_API_KEY")
database_url = os.getenv("DATABASE_URL")
redis_url = os.getenv("REDIS_URL")
port = os.getenv("PORT")

print(f"OPENAI_API_KEY: {'✅ SET' if openai_key else '❌ NOT SET'}")
if openai_key:
    print(f"  Value: {openai_key[:10]}...{openai_key[-10:] if len(openai_key) > 20 else openai_key}")

print(f"DATABASE_URL: {'✅ SET' if database_url else '❌ NOT SET'}")
if database_url:
    print(f"  Value: {database_url[:30]}...{database_url[-20:] if len(database_url) > 50 else database_url}")

print(f"REDIS_URL: {'✅ SET' if redis_url else '❌ NOT SET'}")
if redis_url:
    print(f"  Value: {redis_url}")

print(f"PORT: {'✅ SET' if port else '❌ NOT SET'}")
if port:
    print(f"  Value: {port}")

print("\n🌍 All Environment Variables:")
for key, value in os.environ.items():
    if any(keyword in key.upper() for keyword in ['OPENAI', 'DATABASE', 'REDIS', 'PORT', 'RAILWAY']):
        print(f"  {key}: {value[:50]}{'...' if len(value) > 50 else ''}") 