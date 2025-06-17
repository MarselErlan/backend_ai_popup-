"""
Text extraction utilities for different file types
"""
import io
from typing import Union
import PyPDF2
from docx import Document
import chardet

async def extract_text_from_file(content: bytes, file_ext: str) -> str:
    """
    Extract text from various file types
    Supports: PDF, DOCX, DOC, TXT
    """
    try:
        if file_ext == '.pdf':
            return extract_from_pdf(content)
        elif file_ext in ['.docx', '.doc']:
            return extract_from_docx(content)
        elif file_ext == '.txt':
            return extract_from_txt(content)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")

def extract_from_pdf(content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"PDF extraction error: {str(e)}")

def extract_from_docx(content: bytes) -> str:
    """Extract text from DOCX/DOC file"""
    try:
        doc_file = io.BytesIO(content)
        doc = Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"DOCX extraction error: {str(e)}")

def extract_from_txt(content: bytes) -> str:
    """Extract text from TXT file with encoding detection"""
    try:
        # Detect encoding
        encoding = chardet.detect(content)['encoding'] or 'utf-8'
        return content.decode(encoding).strip()
    except Exception as e:
        raise Exception(f"TXT extraction error: {str(e)}") 