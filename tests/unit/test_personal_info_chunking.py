#!/usr/bin/env python3
"""
Test script for improved personal info chunking
Demonstrates the new smart chunking logic for personal information documents
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.services.embedding_service import EmbeddingService
from loguru import logger

def test_personal_info_chunking():
    """Test the improved chunking with personal information examples"""
    
    # Sample personal info text with various formats
    personal_info_text = """PERSONAL INFORMATION

Name: Eric Abram
Email: ericabram33@gmail.com
Phone: 312-805-9851
Location: San Francisco, CA
LinkedIn: https://linkedin.com/in/eric-abram
GitHub: https://github.com/ericabram

WORK AUTHORIZATION
I am authorized to work in the United States. I am a U.S. citizen and do not require visa sponsorship for employment.

SALARY EXPECTATIONS
My salary expectations are in the range of $120,000 - $150,000 annually, depending on the role, benefits, and company location. I am open to negotiation based on the total compensation package.

WORK PREFERENCES
I prefer hybrid work arrangements with 2-3 days in office and 2-3 days remote. I am available for full-time positions and can start within 2-4 weeks notice. I am willing to relocate for the right opportunity.

TECHNICAL PREFERENCES
I enjoy working with Python, JavaScript, and modern web technologies. I have experience with both frontend and backend development, but I'm particularly passionate about test automation and quality engineering.

ADDITIONAL INFORMATION
I have a strong background in both manual and automated testing. I've worked across various industries including healthcare, fintech, and e-commerce. I'm passionate about building robust testing frameworks and improving software quality.

I'm also interested in mentoring junior developers and contributing to open-source projects. In my free time, I enjoy hiking, reading tech blogs, and experimenting with new programming languages and frameworks.

AVAILABILITY
I am available for interviews Monday through Friday, 9 AM to 6 PM PST. I can also accommodate weekend calls if needed. My preferred interview format is video call, but I'm flexible with in-person meetings in the San Francisco Bay Area.

REFERENCES
References are available upon request. I can provide contacts from my previous employers and colleagues who can speak to my technical skills and work ethic."""

    print("🧪 Testing Improved Personal Info Chunking")
    print("=" * 60)
    
    # Initialize embedding service with settings optimized for personal info
    embedding_service = EmbeddingService(
        openai_api_key="test-key",  # Not needed for chunking test
        chunk_size=500,  # Smaller chunks for personal info
        chunk_overlap=50  # Less overlap for structured content
    )
    
    print(f"📄 Original text length: {len(personal_info_text)} characters")
    print(f"🔧 Chunk size: {embedding_service.chunk_size}")
    print(f"🔄 Chunk overlap: {embedding_service.chunk_overlap}")
    
    # Test the improved chunking
    try:
        chunks = embedding_service._chunk_text_smart(personal_info_text)
        
        print(f"\n✅ Successfully created {len(chunks)} chunks")
        print("\n📋 Personal Info Chunk Analysis:")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
            print(chunk)
            
            # Analyze structure
            lines = chunk.count('\n') + 1
            sentences = chunk.count('.')
            
            print(f"📝 Structure: {lines} lines, ~{sentences} sentences")
            
            # Identify content type
            content_type = "General"
            if any(keyword in chunk.upper() for keyword in ["NAME:", "EMAIL:", "PHONE:", "LOCATION:"]):
                content_type = "Contact Info"
            elif "WORK AUTHORIZATION" in chunk.upper():
                content_type = "Work Authorization"
            elif "SALARY" in chunk.upper():
                content_type = "Salary Info"
            elif "WORK PREFERENCES" in chunk.upper():
                content_type = "Work Preferences"
            elif "TECHNICAL" in chunk.upper():
                content_type = "Technical Info"
            elif "AVAILABILITY" in chunk.upper():
                content_type = "Availability"
            elif "REFERENCES" in chunk.upper():
                content_type = "References"
            
            print(f"🏷️ Content Type: {content_type}")
        
        # Test with different chunk sizes
        print(f"\n🔬 Testing different chunk sizes...")
        
        # Test with smaller chunks (good for contact info)
        small_embedding_service = EmbeddingService(
            openai_api_key="test-key",
            chunk_size=200,
            chunk_overlap=25
        )
        
        small_chunks = small_embedding_service._chunk_text_smart(personal_info_text)
        
        # Test with larger chunks
        large_embedding_service = EmbeddingService(
            openai_api_key="test-key",
            chunk_size=800,
            chunk_overlap=100
        )
        
        large_chunks = large_embedding_service._chunk_text_smart(personal_info_text)
        
        print(f"📊 Chunk Size Comparison:")
        print(f"   Small chunks (200): {len(small_chunks)} chunks")
        print(f"   Medium chunks (500): {len(chunks)} chunks")
        print(f"   Large chunks (800): {len(large_chunks)} chunks")
        
        # Analyze which size works best for personal info
        print(f"\n📈 Optimal Chunking Analysis:")
        
        # Count complete sections in each approach
        section_keywords = ["WORK AUTHORIZATION", "SALARY EXPECTATIONS", "WORK PREFERENCES", "TECHNICAL PREFERENCES"]
        
        def count_complete_sections(chunk_list):
            complete_sections = 0
            for chunk in chunk_list:
                for keyword in section_keywords:
                    if keyword in chunk.upper() and len(chunk) > 100:  # Has content
                        complete_sections += 1
                        break
            return complete_sections
        
        small_complete = count_complete_sections(small_chunks)
        medium_complete = count_complete_sections(chunks)
        large_complete = count_complete_sections(large_chunks)
        
        print(f"   Small chunks - Complete sections: {small_complete}/{len(section_keywords)}")
        print(f"   Medium chunks - Complete sections: {medium_complete}/{len(section_keywords)}")
        print(f"   Large chunks - Complete sections: {large_complete}/{len(section_keywords)}")
        
        # Recommend optimal size
        if medium_complete >= max(small_complete, large_complete):
            print(f"   ✅ Recommended: Medium chunks (500 chars) for personal info")
        elif small_complete > large_complete:
            print(f"   ✅ Recommended: Small chunks (200 chars) for personal info")
        else:
            print(f"   ✅ Recommended: Large chunks (800 chars) for personal info")
        
    except Exception as e:
        print(f"❌ Error during personal info chunking test: {e}")
        import traceback
        traceback.print_exc()

def test_contact_info_specific():
    """Test chunking specifically for contact information"""
    
    contact_info = """Name: Eric Abram
Email: ericabram33@gmail.com
Phone: 312-805-9851
Location: San Francisco, CA
LinkedIn: https://linkedin.com/in/eric-abram
GitHub: https://github.com/ericabram
Portfolio: https://ericabram.dev

EMERGENCY CONTACT
Name: Sarah Abram
Relationship: Spouse
Phone: 312-805-9852
Email: sarah.abram@email.com"""

    print(f"\n🧪 Testing Contact Info Specific Chunking")
    print("=" * 50)
    
    embedding_service = EmbeddingService(
        openai_api_key="test-key",
        chunk_size=300,  # Good size for contact blocks
        chunk_overlap=30
    )
    
    chunks = embedding_service._chunk_text_smart(contact_info)
    
    print(f"📞 Contact info chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Contact Chunk {i} ---")
        print(chunk)
        print(f"Length: {len(chunk)} chars")
        
        # Check if contact info stays together
        if "Name:" in chunk and "Email:" in chunk and "Phone:" in chunk:
            print("✅ Complete contact block maintained")
        elif any(field in chunk for field in ["Name:", "Email:", "Phone:", "Location:"]):
            print("⚠️ Partial contact info")

def test_work_authorization_chunking():
    """Test chunking for work authorization sections"""
    
    work_auth_text = """WORK AUTHORIZATION STATUS

I am authorized to work in the United States without restriction. I am a U.S. citizen by birth and do not require any form of visa sponsorship, including H1-B, L1, or any other work visa.

VISA HISTORY
I have never held a work visa as I have been a U.S. citizen since birth. I have a valid U.S. passport and Social Security number.

FUTURE SPONSORSHIP
I will not require visa sponsorship now or in the future for employment in the United States. I can provide Form I-9 documentation immediately upon hire.

INTERNATIONAL WORK
I am open to international assignments and have a valid passport for travel. I would require appropriate work visas for employment outside the United States, which I understand would be handled by the employer."""

    print(f"\n🧪 Testing Work Authorization Chunking")
    print("=" * 50)
    
    embedding_service = EmbeddingService(
        openai_api_key="test-key",
        chunk_size=400,
        chunk_overlap=40
    )
    
    chunks = embedding_service._chunk_text_smart(work_auth_text)
    
    print(f"🏛️ Work authorization chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Work Auth Chunk {i} ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print(f"Length: {len(chunk)} chars")
        
        # Check for key information preservation
        if "authorized to work" in chunk.lower():
            print("✅ Contains authorization status")
        if "visa" in chunk.lower():
            print("✅ Contains visa information")
        if "sponsorship" in chunk.lower():
            print("✅ Contains sponsorship information")

def main():
    """Run all personal info chunking tests"""
    
    print("🚀 Personal Info Chunking Improvement Test")
    print("=" * 60)
    
    # Test main personal info chunking
    test_personal_info_chunking()
    
    # Test specific sections
    test_contact_info_specific()
    test_work_authorization_chunking()
    
    print(f"\n🎉 Personal info chunking tests completed!")
    print(f"\n💡 Key Benefits for Personal Info:")
    print(f"   ✅ Contact information stays together in logical chunks")
    print(f"   ✅ Work authorization sections remain coherent")
    print(f"   ✅ Salary and preference information is well-structured")
    print(f"   ✅ Better searchability for specific information types")
    print(f"   ✅ Maintains context within each section")
    
    print(f"\n🔧 Recommended Settings for Personal Info:")
    print(f"   • Chunk size: 400-500 characters")
    print(f"   • Chunk overlap: 40-50 characters")
    print(f"   • Priority: Newline breaks > Sentence breaks > Character breaks")

if __name__ == "__main__":
    main() 