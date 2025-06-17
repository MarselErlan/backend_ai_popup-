#!/usr/bin/env python3
"""
Optimized Resume Extractor - High-Performance Vector Database Operations
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from functools import lru_cache
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Try to import Hugging Face embeddings as fallback
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from app.services.document_service import DocumentService


class ResumeExtractorOptimized:
    """Optimized Resume extractor with performance enhancements"""
    
    def __init__(self, openai_api_key: str = None, database_url: str = None, user_id: str = None, use_hf_fallback: bool = True):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.database_url = database_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smart_form_filler")
        self.user_id = user_id
        self.use_hf_fallback = use_hf_fallback
        
        # Cached document service
        self._document_service = None
        
        # Initialize embedding provider with fallback
        self._init_embedding_provider()
        
        # Optimized text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Reduced chunk size for better performance
            chunk_overlap=100,  # Reduced overlap
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Vector database path
        self.vectordb_path = Path("resume/vectordb")
        self.vectordb_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.cache_hits = 0
        self.processed_documents = 0
        
        # Cache for processed embeddings
        self._embedding_cache = {}
        
        logger.info("✅ Optimized Resume extractor initialized")
    
    @property
    def document_service(self):
        """Lazy-loaded document service"""
        if self._document_service is None:
            self._document_service = DocumentService(self.database_url)
        return self._document_service
    
    def _init_embedding_provider(self):
        """Initialize embedding provider with optimized fallback"""
        self.embeddings = None
        self.client = None
        self.embedding_model = "none"
        
        # Try OpenAI first
        if self.openai_api_key:
            try:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=self.openai_api_key,
                    model="text-embedding-3-small",
                    chunk_size=100  # Optimized chunk size for API calls
                )
                # Quick test
                self.embeddings.embed_query("test")
                self.client = OpenAI(api_key=self.openai_api_key)
                self.embedding_model = "openai"
                logger.info("✅ Using OpenAI embeddings (optimized)")
                return
            except Exception as e:
                logger.warning(f"⚠️ OpenAI setup failed: {e}")
        
        # Fallback to Hugging Face
        if self.use_hf_fallback and HF_AVAILABLE:
            try:
                logger.info("🤗 Initializing Hugging Face embeddings...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.embedding_model = "huggingface"
                logger.info("✅ Using Hugging Face embeddings (optimized)")
            except Exception as e:
                logger.error(f"❌ Hugging Face embeddings setup failed: {e}")
    
    @lru_cache(maxsize=10)
    def _get_document_hash(self, user_id: str) -> Optional[str]:
        """Get cached document hash for change detection"""
        try:
            resume_docs = self.document_service.get_user_resume_documents(user_id)
            if not resume_docs:
                return None
            
            # Create hash from document content
            doc = resume_docs[0]  # Get the active document
            content_hash = hashlib.md5(doc.file_content).hexdigest()
            return f"{doc.id}_{content_hash}"
        except Exception as e:
            logger.error(f"Failed to get document hash: {e}")
            return None
    
    def load_resume_from_database_optimized(self) -> List[Any]:
        """Optimized resume loading from database"""
        try:
            # Check if document has changed
            doc_hash = self._get_document_hash(self.user_id)
            if not doc_hash:
                raise FileNotFoundError("No active resume document found")
            
            # Check cache
            if doc_hash in self._embedding_cache:
                self.cache_hits += 1
                logger.info(f"📄 Using cached resume for user: {self.user_id}")
                return self._embedding_cache[doc_hash]
            
            # Load resume documents directly from database
            resume_docs = self.document_service.get_user_resume_documents(self.user_id)
            if not resume_docs:
                raise FileNotFoundError("No active resume document found in database")
            
            # Process documents based on content type
            documents = []
            for doc in resume_docs:
                logger.info(f"📄 Processing resume: {doc.filename} (type: {doc.content_type})")
                
                try:
                    if doc.content_type == 'text/plain':
                        # Handle plain text files
                        page_content = doc.file_content.decode('utf-8')
                    elif doc.content_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
                        # Handle DOCX/DOC files - for now, try UTF-8 decoding
                        # TODO: Implement proper DOCX parsing for binary files
                        try:
                            page_content = doc.file_content.decode('utf-8')
                        except UnicodeDecodeError:
                            # If it's a real DOCX file, we'd need proper parsing
                            logger.warning(f"Cannot decode DOCX file {doc.filename} as UTF-8. Skipping.")
                            page_content = f"[DOCX file: {doc.filename} - content extraction not implemented]"
                    elif doc.content_type == 'application/pdf':
                        # Handle PDF files - for now, try UTF-8 decoding
                        # TODO: Implement proper PDF parsing
                        try:
                            page_content = doc.file_content.decode('utf-8')
                        except UnicodeDecodeError:
                            logger.warning(f"Cannot decode PDF file {doc.filename} as UTF-8. Skipping.")
                            page_content = f"[PDF file: {doc.filename} - content extraction not implemented]"
                    else:
                        # For other file types, try UTF-8 decoding
                        page_content = doc.file_content.decode('utf-8', errors='ignore')
                    
                    # Create document object
                    from langchain.schema import Document
                    document = Document(
                        page_content=page_content,
                        metadata={
                            'source': f"database:{doc.filename}",
                            'user_id': self.user_id,
                            'doc_hash': doc_hash,
                            'filename': doc.filename,
                            'content_type': doc.content_type,
                            'document_id': doc.id
                        }
                    )
                    documents.append(document)
                    
                except Exception as e:
                    logger.warning(f"Failed to process resume document {doc.id}: {e}")
                    # Fallback to raw bytes decoding
                    page_content = doc.file_content.decode('utf-8', errors='ignore')
                    from langchain.schema import Document
                    document = Document(
                        page_content=page_content,
                        metadata={
                            'source': f"database:{doc.filename}",
                            'user_id': self.user_id,
                            'doc_hash': doc_hash,
                            'filename': doc.filename,
                            'content_type': doc.content_type,
                            'document_id': doc.id
                        }
                    )
                    documents.append(document)
            
            # Cache the documents
            self._embedding_cache[doc_hash] = documents
            self.processed_documents += 1
            
            logger.info(f"✅ Loaded {len(documents)} resume document(s) from database (cached)")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error loading resume from database: {e}")
            raise
    
    def split_documents_optimized(self, documents: List[Any]) -> List[Any]:
        """Optimized document splitting"""
        try:
            logger.info("🔪 Splitting documents (optimized)...")
            
            # Create cache key from document content
            doc_content = "".join([doc.page_content for doc in documents])
            content_hash = hashlib.md5(doc_content.encode()).hexdigest()
            
            cache_key = f"chunks_{content_hash}"
            if cache_key in self._embedding_cache:
                self.cache_hits += 1
                logger.info("✅ Using cached document chunks")
                return self._embedding_cache[cache_key]
            
            # Split documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Cache the chunks
            self._embedding_cache[cache_key] = chunks
            
            logger.info(f"✅ Created {len(chunks)} chunks (cached)")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error splitting documents: {e}")
            raise
    
    def create_embeddings_optimized(self, chunks: List[Any]) -> Dict[str, Any]:
        """Optimized embedding creation with batching"""
        try:
            if not self.embeddings:
                logger.error("❌ Embeddings not initialized")
                return {"error": "No embedding provider available"}
            
            logger.info("🔢 Creating embeddings (optimized)...")
            
            # Extract texts efficiently
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Create cache key
            texts_hash = hashlib.md5("".join(texts).encode()).hexdigest()
            cache_key = f"embeddings_{texts_hash}"
            
            if cache_key in self._embedding_cache:
                self.cache_hits += 1
                logger.info("✅ Using cached embeddings")
                return self._embedding_cache[cache_key]
            
            logger.info(f"📊 Creating embeddings for {len(texts)} chunks...")
            
            # Batch processing for better performance
            batch_size = 20  # Optimized batch size
            embeddings_list = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                embeddings_list.extend(batch_embeddings)
                logger.info(f"   Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Create query embedding from sample text
            sample_text = " ".join(texts[:3])  # Use first 3 chunks
            query_embedding = self.embeddings.embed_query(sample_text)
            
            embedding_data = {
                "embeddings": embeddings_list,
                "query_embedding": query_embedding,
                "texts": texts,
                "metadatas": metadatas,
                "model": self.embedding_model,
                "total_chunks": len(texts),
                "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0,
                "creation_timestamp": datetime.now().isoformat(),
                "optimized": True
            }
            
            # Cache the result 
            self._embedding_cache[cache_key] = embedding_data
            
            logger.info(f"✅ Created {len(embeddings_list)} embeddings (cached)")
            return embedding_data
            
        except Exception as e:
            logger.error(f"❌ Optimized embedding creation failed: {e}")
            return {"error": str(e)}
    
    def create_faiss_vectorstore_optimized(self, chunks: List[Any]) -> Any:
        """Optimized FAISS vector store creation"""
        try:
            logger.info("🔍 Creating FAISS vector store (optimized)...")
            
            if not self.embeddings:
                raise ValueError("Embeddings not initialized")
            
            # Create vector store with optimized settings
            vectorstore = FAISS.from_documents(
                chunks, 
                self.embeddings,
                distance_strategy="COSINE"  # Optimized for semantic similarity
            )
            
            logger.info(f"✅ FAISS vector store created with {len(chunks)} documents")
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ FAISS vector store creation failed: {e}")
            raise
    
    def save_to_vectordb_optimized(self, embedding_data: Dict[str, Any], vectorstore: Any = None) -> str:
        """Optimized vector database saving"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user_suffix = f"_{self.user_id}" if self.user_id else ""
            
            # Save embeddings
            embeddings_file = self.vectordb_path / f"resume_embeddings{user_suffix}_{timestamp}.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embedding_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save vector store if provided
            if vectorstore:
                vectorstore_path = self.vectordb_path / f"resume_faiss{user_suffix}_{timestamp}"
                vectorstore.save_local(str(vectorstore_path))
                logger.info(f"✅ FAISS vector store saved: {vectorstore_path}")
            
            logger.info(f"✅ Optimized vector database saved: {embeddings_file}")
            return str(embeddings_file)
            
        except Exception as e:
            logger.error(f"❌ Optimized vector database save failed: {e}")
            raise
    
    def process_resume_optimized(self) -> Dict[str, Any]:
        """Optimized resume processing pipeline"""
        try:
            start_time = datetime.now()
            logger.info(f"🚀 Starting optimized resume processing for user: {self.user_id}")
            
            # Step 1: Load documents (with caching)
            documents = self.load_resume_from_database_optimized()
            
            # Step 2: Split documents (with caching)
            chunks = self.split_documents_optimized(documents)
            
            # Step 3: Create embeddings (with caching)
            embedding_data = self.create_embeddings_optimized(chunks)
            
            if "error" in embedding_data:
                return {"status": "error", "error": embedding_data["error"]}
            
            # Step 4: Create vector store
            vectorstore = self.create_faiss_vectorstore_optimized(chunks)
            
            # Step 5: Save to database
            saved_path = self.save_to_vectordb_optimized(embedding_data, vectorstore)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "status": "success",
                "message": f"Resume processed successfully in {processing_time:.2f}s",
                "chunks": len(chunks),
                "embeddings": len(embedding_data["embeddings"]),
                "dimension": embedding_data["embedding_dimension"],
                "processing_time": processing_time,
                "cache_hits": self.cache_hits,
                "optimized": True,
                "saved_path": saved_path
            }
            
            logger.info(f"✅ Optimized resume processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Optimized resume processing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def search_resume(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Optimized resume search with caching"""
        try:
            # Load the latest vector store - check multiple naming patterns
            vectorstore_files = list(self.vectordb_path.glob(f"resume_faiss_{self.user_id}_*"))
            if not vectorstore_files:
                # Fallback to any resume vector store
                vectorstore_files = list(self.vectordb_path.glob("resume_faiss_*"))
            if not vectorstore_files:
                # Fallback to legacy naming pattern
                vectorstore_files = list(self.vectordb_path.glob("faiss_store_*"))
            
            if not vectorstore_files:
                return {"status": "not_found", "message": "No vector store found"}
            
            # Get the latest vector store
            latest_vectorstore = max(vectorstore_files, key=lambda x: x.stat().st_mtime)
            
            # Load vector store with caching
            cache_key = f"vectorstore_{latest_vectorstore.name}"
            if cache_key not in self._embedding_cache:
                if not self.embeddings:
                    return {"status": "error", "error": "Embeddings not initialized"}
                
                vectorstore = FAISS.load_local(
                    str(latest_vectorstore), 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self._embedding_cache[cache_key] = vectorstore
            else:
                vectorstore = self._embedding_cache[cache_key]
                self.cache_hits += 1
            
            # Perform optimized similarity search
            docs = vectorstore.similarity_search(query, k=k)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance": "high"  # Simplified relevance scoring
                })
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_results": len(results),
                "optimized": True
            }
            
        except Exception as e:
            logger.error(f"❌ Optimized resume search failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "cache_hits": self.cache_hits,
            "processed_documents": self.processed_documents,
            "cache_size": len(self._embedding_cache),
            "embedding_model": self.embedding_model,
            "optimization_enabled": True
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self._embedding_cache.clear()
        # Clear LRU cache
        self._get_document_hash.cache_clear()
        logger.info("🧹 All caches cleared")


def main():
    """Test the optimized resume extractor"""
    extractor = ResumeExtractorOptimized(user_id="default")
    result = extractor.process_resume_optimized()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main() 