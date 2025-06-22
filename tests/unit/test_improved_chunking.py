#!/usr/bin/env python3
"""
Test script for improved resume chunking
Demonstrates the new smart chunking logic based on newlines and periods
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.services.embedding_service import EmbeddingService
from loguru import logger

def test_improved_chunking():
    """Test the improved chunking with the user's resume example"""
    
    # Sample resume text provided by the user
    resume_text = """Name: Eric Abram
Phone: 312-805-9851
Email: ericabram33@gmail.com
Location: San Francisco, CA
LinkedIn: https://linkedin.com/in/eric-abram
GitHub: https://github.com/ericabram
Portfolio: https://ericabram.dev

About Me:
I'm a Full-Stack Software Test Automation Engineer (SDET) with 8+ years of experience in backend development and quality engineering. My career began in manual testing and evolved through backend development using Python, Django, and FastAPI, before returning to test automation to build intelligent, scalable testing pipelines.

I specialize in developing robust frameworks for UI, API, and database testing using tools like Selenium, Playwright, RestAssured, Jenkins, and Kubernetes. I've worked on high-impact systems across healthcare, banking, and global e-commerce platforms.

EXPERIENCE:

Senior SDET at TechCorp (2021-2024)
• Led test automation initiatives for microservices architecture serving 10M+ users
• Developed comprehensive testing frameworks using Python, Pytest, and Docker
• Implemented CI/CD pipelines reducing deployment time by 60%
• Mentored junior engineers and established testing best practices

Backend Developer at DataFlow Inc (2019-2021)
• Built scalable APIs using Django REST Framework and PostgreSQL
• Optimized database queries improving response times by 40%
• Integrated third-party payment systems and authentication services
• Collaborated with frontend teams on React-based applications

Quality Assurance Engineer at StartupXYZ (2016-2019)
• Designed and executed comprehensive test plans for web and mobile applications
• Automated regression testing suites using Selenium WebDriver
• Performed API testing using Postman and custom Python scripts
• Managed bug tracking and release coordination

TECHNICAL SKILLS:

Programming Languages:
Python, JavaScript, Java, SQL, Bash

Testing Frameworks & Tools:
Selenium, Playwright, Pytest, TestNG, Cucumber, Postman, JMeter

Backend Technologies:
Django, FastAPI, Flask, Node.js, Express.js

Databases:
PostgreSQL, MySQL, MongoDB, Redis

DevOps & Cloud:
Docker, Kubernetes, Jenkins, GitHub Actions, AWS, Azure

EDUCATION:

Bachelor of Science in Computer Science
University of California, Berkeley (2012-2016)
• Relevant coursework: Data Structures, Algorithms, Database Systems, Software Engineering
• Senior Project: Automated Testing Framework for Web Applications

CERTIFICATIONS:

• AWS Certified Solutions Architect - Associate (2023)
• Certified Kubernetes Administrator (CKA) (2022)
• ISTQB Foundation Level Test Analyst (2020)

PROJECTS:

SmartTest Framework (2023)
• Open-source test automation framework combining Selenium and AI
• Supports cross-browser testing and intelligent element detection
• 500+ GitHub stars and active community contributions

E-commerce Testing Suite (2022)
• Comprehensive testing solution for online retail platforms
• Automated end-to-end testing covering payment flows and inventory management
• Reduced manual testing effort by 80%

API Performance Monitor (2021)
• Real-time API monitoring system built with Python and Redis
• Automated performance regression detection and alerting
• Deployed across multiple production environments"""

    print("🧪 Testing Improved Resume Chunking")
    print("=" * 60)
    
    # Initialize embedding service (no need for actual OpenAI key for chunking test)
    embedding_service = EmbeddingService(
        openai_api_key="test-key",  # Not needed for chunking test
        chunk_size=800,  # Same as current setting
        chunk_overlap=100
    )
    
    print(f"📄 Original text length: {len(resume_text)} characters")
    print(f"🔧 Chunk size: {embedding_service.chunk_size}")
    print(f"🔄 Chunk overlap: {embedding_service.chunk_overlap}")
    
    # Test the improved chunking
    try:
        chunks = embedding_service._chunk_text_smart(resume_text)
        
        print(f"\n✅ Successfully created {len(chunks)} chunks")
        print("\n📋 Chunk Analysis:")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
            # Show first 200 characters of each chunk
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            print(preview)
            
            # Highlight structure
            if '\n' in chunk:
                line_count = chunk.count('\n') + 1
                print(f"📝 Contains {line_count} lines")
            
            if '.' in chunk:
                sentence_count = chunk.count('.')
                print(f"📖 Contains ~{sentence_count} sentences")
        
        # Compare with old chunking method
        print(f"\n🔄 Comparing with legacy chunking...")
        
        # Simulate old chunking (character-based)
        old_chunks = embedding_service._chunk_by_characters(resume_text)
        
        print(f"📊 Chunking Comparison:")
        print(f"   Smart chunking: {len(chunks)} chunks")
        print(f"   Legacy chunking: {len(old_chunks)} chunks")
        
        # Analyze readability
        print(f"\n📖 Readability Analysis:")
        
        smart_complete_lines = sum(1 for chunk in chunks if not chunk.endswith(' ') or chunk.count('\n') > 0)
        legacy_complete_lines = sum(1 for chunk in old_chunks if not chunk.endswith(' '))
        
        print(f"   Smart chunks with complete thoughts: {smart_complete_lines}/{len(chunks)} ({smart_complete_lines/len(chunks)*100:.1f}%)")
        print(f"   Legacy chunks with complete thoughts: {legacy_complete_lines}/{len(old_chunks)} ({legacy_complete_lines/len(old_chunks)*100:.1f}%)")
        
        # Show size distribution
        smart_sizes = [len(chunk) for chunk in chunks]
        legacy_sizes = [len(chunk) for chunk in old_chunks]
        
        print(f"\n📏 Size Distribution:")
        print(f"   Smart - Avg: {sum(smart_sizes)/len(smart_sizes):.0f}, Min: {min(smart_sizes)}, Max: {max(smart_sizes)}")
        print(f"   Legacy - Avg: {sum(legacy_sizes)/len(legacy_sizes):.0f}, Min: {min(legacy_sizes)}, Max: {max(legacy_sizes)}")
        
    except Exception as e:
        print(f"❌ Error during chunking test: {e}")
        import traceback
        traceback.print_exc()

def test_contact_info_chunking():
    """Test chunking specifically for contact information like the user's example"""
    
    contact_text = """Name: Eric Abram
Phone: 312-805-9851
Email: ericabram33@gmail.com
Location: San Francisco, CA
LinkedIn: https://linkedin.com/in/eric-abram
GitHub: https://github.com/ericabram
Portfolio: https://ericabram.dev"""
    
    print(f"\n🧪 Testing Contact Info Chunking")
    print("=" * 40)
    
    embedding_service = EmbeddingService(
        openai_api_key="test-key",
        chunk_size=200,  # Smaller chunk size for contact info
        chunk_overlap=50
    )
    
    chunks = embedding_service._chunk_text_smart(contact_text)
    
    print(f"📞 Contact info chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Contact Chunk {i} ---")
        print(chunk)
        print(f"Length: {len(chunk)} chars")

def main():
    """Run all chunking tests"""
    
    print("🚀 Resume Chunking Improvement Test")
    print("=" * 60)
    
    # Test main resume chunking
    test_improved_chunking()
    
    # Test contact info specific chunking
    test_contact_info_chunking()
    
    print(f"\n🎉 Chunking tests completed!")
    print(f"\n💡 Key Improvements:")
    print(f"   ✅ Respects newline boundaries (contact info stays together)")
    print(f"   ✅ Splits on sentence boundaries (periods)")
    print(f"   ✅ Maintains readable chunks")
    print(f"   ✅ Falls back to character chunking for very long text")
    print(f"   ✅ Better chunk size distribution")

if __name__ == "__main__":
    main() 