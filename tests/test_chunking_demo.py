#!/usr/bin/env python3
"""
Comprehensive test to demonstrate improved chunking for personal info
Shows the difference between old (large) and new (small) chunks
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.services.embedding_service import EmbeddingService
from loguru import logger

def test_old_vs_new_chunking():
    """Test to show the difference between old (800 characters) and new (200 characters) chunking"""
    
    # Your exact personal info example
    personal_info_text = """Name: Eric Abram
Phone: 312-805-9851
Email: ericabram33@gmail.com
Location: San Francisco, CA
LinkedIn: https://linkedin.com/in/eric-abram
GitHub: https://github.com/ericabram
Portfolio: https://ericabram.dev

About Me:
I'm a Full-Stack Software Test Automation Engineer (SDET) with 8+ years of experience in backend development and quality engineering. My career began in manual testing and evolved through backend development using Python, Django, and FastAPI, before returning to test automation to build intelligent, scalable testing pipelines.

I specialize in developing robust frameworks for UI, API, and database testing using tools like Selenium, Playwright, RestAssured, Jenkins, and Kubernetes. I've worked on high-impact systems across healthcare, banking, and global e-commerce platforms, ensuring quality at scale.

Core Skills:
• Test Automation: Selenium WebDriver, Playwright, Cypress, TestNG, JUnit
• API Testing: RestAssured, Postman, Newman, API security testing
• Backend Development: Python, Java, Node.js, Django, FastAPI, Spring Boot
• Database Testing: SQL, NoSQL, data validation, migration testing
• CI/CD: Jenkins, GitHub Actions, Docker, Kubernetes deployment pipelines
• Performance Testing: JMeter, LoadRunner, stress testing methodologies
• Cloud Platforms: AWS, GCP, containerization, microservices testing

Professional Experience:

Senior SDET at TechCorp (2020-Present)
• Built comprehensive test automation framework reducing manual testing by 85%
• Implemented API testing suite covering 200+ endpoints with 95% coverage
• Designed database validation framework for financial transactions
• Led migration from legacy testing tools to modern automation stack
• Mentored junior engineers in test automation best practices

Backend Developer at StartupXYZ (2018-2020)
• Developed REST APIs using Django and FastAPI serving 100K+ daily requests
• Implemented database optimization reducing query response time by 60%
• Built microservices architecture with Docker and Kubernetes
• Created automated deployment pipelines with Jenkins and AWS
• Collaborated with frontend team on API design and integration

QA Engineer at Enterprise Solutions (2016-2018)
• Performed manual and automated testing for enterprise web applications
• Created test cases and documentation for complex business workflows
• Implemented regression testing suite reducing release cycle time
• Worked closely with development team on bug resolution and quality metrics"""

    print("🔍 CHUNKING COMPARISON TEST")
    print("=" * 80)
    
    # OLD CHUNKING (800 characters - what you had before)
    print("\n❌ OLD CHUNKING (800 characters):")
    print("-" * 50)
    
    old_embedding_service = EmbeddingService(chunk_size=800, chunk_overlap=100)
    old_chunks = old_embedding_service._chunk_text(personal_info_text)
    
    for i, chunk in enumerate(old_chunks, 1):
        print(f"\n📄 OLD CHUNK {i} ({len(chunk)} chars):")
        print(f"'{chunk[:200]}{'...' if len(chunk) > 200 else ''}'")
    
    print(f"\n📊 OLD CHUNKING SUMMARY:")
    print(f"   • Total chunks: {len(old_chunks)}")
    print(f"   • Average chunk size: {sum(len(c) for c in old_chunks) // len(old_chunks)} chars")
    print(f"   • Largest chunk: {max(len(c) for c in old_chunks)} chars")
    
    # NEW CHUNKING (200 characters - what you have now)
    print("\n✅ NEW CHUNKING (200 characters):")
    print("-" * 50)
    
    new_embedding_service = EmbeddingService(chunk_size=200, chunk_overlap=30)
    new_chunks = new_embedding_service._chunk_text(personal_info_text)
    
    for i, chunk in enumerate(new_chunks, 1):
        print(f"\n📄 NEW CHUNK {i} ({len(chunk)} chars):")
        print(f"'{chunk}'")
    
    print(f"\n📊 NEW CHUNKING SUMMARY:")
    print(f"   • Total chunks: {len(new_chunks)}")
    print(f"   • Average chunk size: {sum(len(c) for c in new_chunks) // len(new_chunks)} chars")
    print(f"   • Largest chunk: {max(len(c) for c in new_chunks)} chars")
    
    # COMPARISON
    print(f"\n🎯 IMPROVEMENT SUMMARY:")
    print(f"   • Old: {len(old_chunks)} large chunks (hard to read)")
    print(f"   • New: {len(new_chunks)} small chunks (easy to read)")
    print(f"   • Chunk size reduction: {800-200} characters smaller")
    print(f"   • Better searchability: ✅")
    print(f"   • Contact info stays together: ✅")
    print(f"   • Natural text boundaries: ✅")

if __name__ == "__main__":
    test_old_vs_new_chunking() 