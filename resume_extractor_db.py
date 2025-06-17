#!/usr/bin/env python3
"""
Database-based Resume Extractor - Smart Form Fill Vector Database Creation
Reads resume from database instead of files
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
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


class ResumeExtractorDB:
    """Database-based Resume extractor for creating vector embeddings"""
    
    def __init__(self, openai_api_key: str = None, database_url: str = None, user_id: str = None, use_hf_fallback: bool = True):
        # API setup
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.database_url = database_url or os.getenv("POSTGRES_DB_URL", "postgresql://ai_popup:Erlan1824@localhost:5432/ai_popup")
        self.user_id = user_id
        self.use_hf_fallback = use_hf_fallback
        
        # Initialize document service
        self.document_service = DocumentService(self.database_url)
        
        # Try OpenAI first, then fallback to Hugging Face
        if self.openai_api_key:
            try:
                # Test OpenAI with a small embedding to verify quota
                test_embeddings = OpenAIEmbeddings(
                    openai_api_key=self.openai_api_key,
                    model="text-embedding-3-small"
                )
                # Quick test to verify API works
                test_embeddings.embed_query("test")
                
                self.embeddings = test_embeddings
                self.client = OpenAI(api_key=self.openai_api_key)
                self.embedding_model = "openai"
                logger.info("✅ Using OpenAI embeddings")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI setup failed: {e}")
                self.embeddings = None
                self.client = None
        else:
            self.embeddings = None
            self.client = None
        
        # Fallback to Hugging Face if OpenAI not available
        if self.embeddings is None and self.use_hf_fallback and HF_AVAILABLE:
            try:
                logger.info("🤗 Initializing Hugging Face embeddings as fallback...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.embedding_model = "huggingface"
                self.client = None  # No OpenAI client needed
                logger.info("✅ Using Hugging Face embeddings (all-MiniLM-L6-v2)")
            except Exception as e:
                logger.error(f"❌ Hugging Face embeddings setup failed: {e}")
                self.embeddings = None
        
        if self.embeddings is None:
            logger.error("❌ No embedding provider available. Install sentence-transformers for HF support.")
            self.embedding_model = "none"
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Vector database path (still used for FAISS storage)
        self.vectordb_path = Path("resume/vectordb")
        self.vectordb_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Resume extractor (DB) initialized with LangChain")
        logger.info(f"Database URL: {self.database_url}")
        logger.info(f"User ID: {self.user_id}")
        logger.info(f"Vector DB path: {self.vectordb_path}")
    
    def load_resume_from_database(self) -> List[Any]:
        """Load resume from database using document service"""
        try:
            # Get resume as temporary file
            temp_result = self.document_service.get_resume_as_temp_file(self.user_id)
            if not temp_result:
                raise FileNotFoundError("No active resume document found in database")
            
            temp_path, filename = temp_result
            
            logger.info(f"📄 Loading resume from database: {filename}")
            
            try:
                # Use LangChain's Docx2txtLoader with temporary file
                loader = Docx2txtLoader(temp_path)
                documents = loader.load()
                
                logger.info(f"✅ Loaded {len(documents)} document(s) from database")
                
                # Log document details
                for i, doc in enumerate(documents):
                    logger.info(f"   Document {i+1}: {len(doc.page_content)} characters")
                    # Update metadata to reflect database source
                    doc.metadata['source'] = f"database:{filename}"
                    doc.metadata['user_id'] = self.user_id
                    logger.info(f"   Metadata: {doc.metadata}")
                
                return documents
                
            finally:
                # Clean up temporary file
                self.document_service.cleanup_temp_file(temp_path)
            
        except Exception as e:
            logger.error(f"❌ Error loading resume from database: {e}")
            raise
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks using LangChain text splitter"""
        try:
            logger.info("🔪 Splitting documents into chunks...")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"✅ Created {len(chunks)} chunks from documents")
            
            # Log chunk details
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            avg_chunk_size = total_chars / len(chunks) if chunks else 0
            
            logger.info(f"   Total characters: {total_chars}")
            logger.info(f"   Average chunk size: {avg_chunk_size:.0f} characters")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error splitting documents: {e}")
            raise
    
    def create_embeddings_with_langchain(self, chunks: List[Any]) -> Dict[str, Any]:
        """Create embeddings using LangChain embeddings"""
        try:
            if not self.embeddings:
                logger.error("❌ Embeddings not initialized. Cannot create embeddings.")
                return {"error": "No embedding provider available"}
            
            logger.info("🔢 Creating embeddings with LangChain...")
            
            # Extract text content from chunks
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            logger.info(f"📊 Creating embeddings for {len(texts)} text chunks...")
            
            # Create embeddings using LangChain
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # Also create a query embedding for the full resume
            full_text = "\n\n".join(texts)
            query_embedding = self.embeddings.embed_query(full_text)
            
            embedding_data = {
                "embeddings": embeddings_list,
                "query_embedding": query_embedding,
                "texts": texts,
                "metadatas": metadatas,
                "model": getattr(self, 'embedding_model', 'unknown'),
                "total_chunks": len(texts),
                "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0,
                "creation_timestamp": datetime.now().isoformat(),
                "source_database": "database",
                "chunk_stats": {
                    "total_characters": sum(len(text) for text in texts),
                    "avg_chunk_size": sum(len(text) for text in texts) / len(texts) if texts else 0,
                    "min_chunk_size": min(len(text) for text in texts) if texts else 0,
                    "max_chunk_size": max(len(text) for text in texts) if texts else 0
                }
            }
            
            logger.info(f"✅ Created {len(embeddings_list)} embeddings with dimension {embedding_data['embedding_dimension']}")
            
            return embedding_data
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings with LangChain: {e}")
            raise
    
    def create_faiss_vectorstore(self, chunks: List[Any]) -> Any:
        """Create FAISS vector store using LangChain"""
        try:
            if not self.embeddings:
                logger.error("❌ Embeddings not initialized. Cannot create vector store.")
                return None
            
            logger.info("🗄️ Creating FAISS vector store...")
            
            # Extract texts and metadatas
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Create FAISS vector store
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"✅ Created FAISS vector store with {len(texts)} documents")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error creating FAISS vector store: {e}")
            raise
    
    def save_to_vectordb(self, embedding_data: Dict[str, Any], vectorstore: Any = None) -> str:
        """Save embeddings and vector store to database"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save embedding data as JSON
            embeddings_file = self.vectordb_path / f"embeddings_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_data = embedding_data.copy()
            if "embeddings" in json_data:
                json_data["embeddings"] = [emb if isinstance(emb, list) else emb.tolist() for emb in json_data["embeddings"]]
            if "query_embedding" in json_data:
                json_data["query_embedding"] = json_data["query_embedding"] if isinstance(json_data["query_embedding"], list) else json_data["query_embedding"].tolist()
            
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Save raw embeddings as pickle for faster loading
            pickle_file = self.vectordb_path / f"embeddings_{timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(embedding_data, f)
            
            # Save FAISS vector store if available
            faiss_path = None
            if vectorstore:
                faiss_path = self.vectordb_path / f"faiss_store_{timestamp}"
                vectorstore.save_local(str(faiss_path))
                logger.info(f"   💾 FAISS store: {faiss_path}")
            
            # Save metadata
            metadata = {
                "embeddings_json": str(embeddings_file),
                "embeddings_pickle": str(pickle_file),
                "faiss_store": str(faiss_path) if faiss_path else None,
                "timestamp": timestamp,
                "creation_date": datetime.now().isoformat(),
                "source_database": "database",
                "total_chunks": embedding_data.get("total_chunks", 0),
                "embedding_dimension": embedding_data.get("embedding_dimension", 0),
                "model": embedding_data.get("model", "unknown"),
                "chunk_stats": embedding_data.get("chunk_stats", {})
            }
            
            metadata_file = self.vectordb_path / f"metadata_{timestamp}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Update index
            index_file = self.vectordb_path / "index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
            else:
                index_data = {"entries": []}
            
            index_data["entries"].append(metadata)
            index_data["last_updated"] = datetime.now().isoformat()
            index_data["total_entries"] = len(index_data["entries"])
            
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            logger.info(f"✅ Saved to vector database:")
            logger.info(f"   📄 Embeddings JSON: {embeddings_file}")
            logger.info(f"   🥒 Embeddings Pickle: {pickle_file}")
            logger.info(f"   📋 Metadata: {metadata_file}")
            logger.info(f"   📚 Index updated: {index_file}")
            
            return timestamp
            
        except Exception as e:
            logger.error(f"❌ Error saving to vector database: {e}")
            raise
    
    def process_resume(self) -> Dict[str, Any]:
        """Complete pipeline: load from DB → split → embed → save"""
        try:
            start_time = datetime.now()
            
            logger.info("🚀 Starting LangChain resume processing pipeline (Database)...")
            
            # Get active resume document for logging
            resume_doc = self.document_service.get_active_resume_document(self.user_id)
            if not resume_doc:
                raise ValueError("No active resume document found in database")
            
            # Log processing start
            log_id = self.document_service.log_processing_start("resume", resume_doc.id, self.user_id)
            
            try:
                # Update resume status to processing
                self.document_service.update_resume_processing_status(resume_doc.id, "processing")
                
                # Step 1: Load resume from database
                logger.info("📄 Step 1: Loading resume from database...")
                documents = self.load_resume_from_database()
                
                # Step 2: Split documents
                logger.info("🔪 Step 2: Splitting documents into chunks...")
                chunks = self.split_documents(documents)
                
                # Step 3: Create embeddings
                logger.info("🔢 Step 3: Creating embeddings...")
                embedding_data = self.create_embeddings_with_langchain(chunks)
                
                if "error" in embedding_data:
                    raise ValueError(embedding_data["error"])
                
                # Step 4: Create vector store
                logger.info("🗄️ Step 4: Creating FAISS vector store...")
                vectorstore = self.create_faiss_vectorstore(chunks)
                
                # Step 5: Save to vector database
                logger.info("💾 Step 5: Saving to vector database...")
                timestamp = self.save_to_vectordb(embedding_data, vectorstore)
                
                end_time = datetime.now()
                processing_time = int((end_time - start_time).total_seconds())
                
                # Update processing status
                self.document_service.update_resume_processing_status(resume_doc.id, "completed", end_time)
                
                # Log processing completion
                self.document_service.log_processing_complete(
                    log_id, processing_time, 
                    embedding_data.get("total_chunks"),
                    embedding_data.get("embedding_dimension"),
                    embedding_data.get("model")
                )
                
                result = {
                    "status": "success",
                    "message": "Resume processing completed successfully from database",
                    "timestamp": timestamp,
                    "processing_time": processing_time,
                    "source": "database",
                    "user_id": self.user_id,
                    "document_id": resume_doc.id,
                    "database_info": {
                        "total_chunks": embedding_data.get("total_chunks", 0),
                        "embedding_dimension": embedding_data.get("embedding_dimension", 0),
                        "model": embedding_data.get("model", "unknown"),
                        "chunk_stats": embedding_data.get("chunk_stats", {}),
                        "vectordb_path": str(self.vectordb_path)
                    }
                }
                
                logger.info("🎉 LangChain resume processing completed successfully from database!")
                return result
                
            except Exception as e:
                # Update processing status to failed
                self.document_service.update_resume_processing_status(resume_doc.id, "failed")
                
                # Log processing error
                self.document_service.log_processing_error(log_id, str(e))
                
                raise e
        
        except Exception as e:
            logger.error(f"❌ Resume processing failed: {e}")
            return {
                "status": "error",
                "message": f"Resume processing failed: {str(e)}",
                "source": "database",
                "user_id": self.user_id
            }
    
    def search_resume(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search the resume vector database for relevant content"""
        try:
            # Load the latest FAISS store
            index_file = self.vectordb_path / "index.json"
            if not index_file.exists():
                return {"error": "No vector database found. Run process_resume() first."}
            
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            if not index_data["entries"]:
                return {"error": "No entries in vector database."}
            
            # Get the latest entry
            latest_entry = index_data["entries"][-1]
            faiss_path = latest_entry.get("faiss_store")
            
            if not faiss_path:
                return {"error": "No FAISS store available."}
            
            # Load FAISS store
            vectorstore = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            
            # Search
            results = vectorstore.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
            
            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Error searching resume: {e}")
            return {"error": str(e)}


def main():
    """Test the database-based resume extractor"""
    extractor = ResumeExtractorDB()
    result = extractor.process_resume()
    print(f"Processing result: {result}")


if __name__ == "__main__":
    main() 