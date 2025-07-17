#!/usr/bin/env python3
"""
RAG Service - Proper Vector Similarity Search Implementation
Based on the working RAG pattern from in-class-capstone
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from app.services.embedding_service import EmbeddingService
from loguru import logger

class RAGService:
    """Proper RAG implementation using vector similarity search."""
    
    def __init__(self, openai_api_key: str = None):
        self.embedding_service = EmbeddingService()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        # System prompt for field answering
        self.system_prompt = """You are a helpful AI assistant that extracts specific information from documents to fill form fields.

Rules:
1. ONLY use information from the provided context
2. Extract the EXACT answer to the question 
3. If the information is not in the context, return "Not found"
4. Be precise and concise - return only the requested information
5. Do not make up or infer information

Context from documents:
{context}

Question: {question}

Answer:"""

    def _retrieve_relevant_context(self, query: str, user_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant context using proper vector similarity search."""
        try:
            logger.info(f"🔍 RAG: Searching for '{query}' for user {user_id}")
            
            # Search resume documents
            resume_results = self.embedding_service.search_similar_by_document_type(
                query=query,
                user_id=user_id,
                document_type="resume",
                top_k=top_k,
                min_score=0.1  # Low threshold to get results
            )
            
            # Search personal info documents  
            personal_results = self.embedding_service.search_similar_by_document_type(
                query=query,
                user_id=user_id,
                document_type="personal_info",
                top_k=top_k,
                min_score=0.1  # Low threshold to get results
            )
            
            # Combine and sort by similarity score (higher is better)
            all_results = resume_results + personal_results
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Take top results
            top_results = all_results[:top_k]
            
            logger.info(f"📊 RAG: Found {len(resume_results)} resume + {len(personal_results)} personal = {len(all_results)} total, using top {len(top_results)}")
            
            # Log similarity scores for debugging
            for i, result in enumerate(top_results):
                score = result.get('score', 0)
                text_preview = result.get('text', '')[:100].replace('\n', ' ')
                logger.info(f"   Result {i+1}: score={score:.3f} | '{text_preview}...'")
            
            return top_results
            
        except Exception as e:
            logger.error(f"❌ RAG: Error retrieving context: {e}")
            return []

    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context string for the LLM."""
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            content = result.get('text', 'No content available')
            score = result.get('score', 0)
            doc_type = result.get('document_type', 'unknown')
            
            context_parts.append(f"Document {i} ({doc_type}, similarity: {score:.3f}):")
            context_parts.append(content)
            context_parts.append("---")
        
        return "\n".join(context_parts)

    def generate_field_answer(self, question: str, user_id: str, top_k: int = 3) -> Dict[str, Any]:
        """Generate answer for a form field using proper RAG approach."""
        try:
            logger.info(f"🧠 RAG: Processing question '{question}' for user {user_id}")
            
            # Step 1: Retrieve relevant context using vector similarity
            search_results = self._retrieve_relevant_context(question, user_id, top_k)
            
            if not search_results:
                logger.warning("⚠️ RAG: No search results found")
                return {
                    "answer": "Not found",
                    "confidence": 0,
                    "data_source": "no_results",
                    "reasoning": "No relevant information found in documents",
                    "similarity_scores": []
                }
            
            # Step 2: Check if we have high-similarity results
            best_score = search_results[0].get('score', 0) if search_results else 0
            
            if best_score < 0.2:  # Very low similarity
                logger.warning(f"⚠️ RAG: Best similarity score too low: {best_score:.3f}")
                return {
                    "answer": "Not found",
                    "confidence": 0,
                    "data_source": "low_similarity",
                    "reasoning": f"Best similarity score {best_score:.3f} below threshold",
                    "similarity_scores": [r.get('score', 0) for r in search_results]
                }
            
            # Step 3: Format context for LLM
            context = self._format_context(search_results)
            
            # Step 4: Generate answer using LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt)
            ])
            
            messages = prompt.format_messages(
                context=context,
                question=question
            )
            
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            # Step 5: Determine confidence based on similarity scores
            avg_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
            confidence = min(100, int(avg_score * 200))  # Convert to percentage
            
            # Step 6: Determine data source
            data_source = "vector_search"
            if answer.lower() in ["not found", "not available", "unknown"]:
                data_source = "not_found"
                confidence = 0
            elif best_score > 0.7:
                data_source = "high_confidence_match"
            elif best_score > 0.4:
                data_source = "medium_confidence_match"
            else:
                data_source = "low_confidence_match"
            
            logger.info(f"✅ RAG: Generated answer '{answer}' with confidence {confidence}% (source: {data_source})")
            
            return {
                "answer": answer,
                "confidence": confidence,
                "data_source": data_source,
                "reasoning": f"Vector search found {len(search_results)} relevant chunks, best similarity: {best_score:.3f}",
                "similarity_scores": [r.get('score', 0) for r in search_results],
                "context_used": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"❌ RAG: Error generating answer: {e}")
            return {
                "answer": "Error processing request",
                "confidence": 0,
                "data_source": "error",
                "reasoning": f"Error: {str(e)}",
                "similarity_scores": []
            } 