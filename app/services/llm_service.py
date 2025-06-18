"""
🧠 Redis-Enabled LLM Service for Smart Form Filling
Integrates with Redis Vector Store for fast, intelligent field answers
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from loguru import logger

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import RedisVectorStore

class RedisLLMService:
    """
    🧠 Redis-powered LLM service for intelligent form filling
    
    Features:
    - 3-tier data retrieval (Resume Redis → Personal Info Redis → LLM Generation)
    - Early exit optimization (skip tiers when sufficient data found)
    - Smart field analysis and contextual understanding
    - Performance tracking and analytics
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key
        
        # Initialize Redis-enabled services
        self.embedding_service = EmbeddingService()
        self.vector_store = RedisVectorStore()
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "tier_1_exits": 0,
            "tier_2_exits": 0,
            "tier_3_completions": 0,
            "avg_processing_time": 0.0
        }
        
        logger.info("🧠 Redis LLM Service initialized with vector store integration")
    
    async def generate_field_answer(
        self,
        field_label: str,
        user_id: str,
        field_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        🎯 Generate intelligent field answer using Redis-powered 3-tier system
        
        Args:
            field_label: The form field label/question
            user_id: User identifier for vector search
            field_context: Additional context about the field
            
        Returns:
            Dict with answer, data_source, reasoning, and performance metrics
        """
        start_time = time.time()
        self.performance_stats["total_requests"] += 1
        
        logger.info(f"🧠 Generating answer for: '{field_label}' (User: {user_id})")
        
        try:
            # TIER 1: Search Resume Vector Store
            logger.info("🎯 TIER 1: Resume Vector Search")
            resume_results = await self._search_resume_vectors(field_label, user_id)
            
            # Check if TIER 1 is sufficient
            tier1_confidence = self._calculate_confidence(resume_results, field_label)
            
            if tier1_confidence >= 0.8:
                # Early exit with resume data
                answer = self._extract_answer_from_results(resume_results, field_label)
                if answer:
                    processing_time = time.time() - start_time
                    self.performance_stats["tier_1_exits"] += 1
                    
                    logger.info(f"✅ TIER 1 EXIT: '{answer}' (confidence: {tier1_confidence:.2f})")
                    
                    return {
                        "answer": answer,
                        "data_source": "resume_vectordb",
                        "reasoning": f"Found high-confidence answer in resume data (confidence: {tier1_confidence:.2f})",
                        "status": "success",
                        "performance_metrics": {
                            "processing_time_seconds": processing_time,
                            "tier_exit": 1,
                            "confidence_score": tier1_confidence,
                            "early_exit": True
                        }
                    }
            
            # TIER 2: Search Personal Info Vector Store
            logger.info("🎯 TIER 2: Personal Info Vector Search")
            personal_results = await self._search_personal_info_vectors(field_label, user_id)
            
            # Check if TIER 1 + TIER 2 is sufficient
            combined_confidence = self._calculate_combined_confidence(
                resume_results, personal_results, field_label
            )
            
            if combined_confidence >= 0.8:
                # Early exit with combined data
                answer = self._extract_answer_from_combined_results(
                    resume_results, personal_results, field_label
                )
                if answer:
                    processing_time = time.time() - start_time
                    self.performance_stats["tier_2_exits"] += 1
                    
                    data_source = self._determine_primary_source(answer, resume_results, personal_results)
                    
                    logger.info(f"✅ TIER 2 EXIT: '{answer}' (confidence: {combined_confidence:.2f})")
                    
                    return {
                        "answer": answer,
                        "data_source": data_source,
                        "reasoning": f"Found high-confidence answer in combined data (confidence: {combined_confidence:.2f})",
                        "status": "success",
                        "performance_metrics": {
                            "processing_time_seconds": processing_time,
                            "tier_exit": 2,
                            "confidence_score": combined_confidence,
                            "early_exit": True
                        }
                    }
            
            # TIER 3: LLM Generation with Context
            logger.info("🎯 TIER 3: LLM Generation")
            llm_answer = await self._generate_llm_answer(
                field_label, resume_results, personal_results, field_context
            )
            
            processing_time = time.time() - start_time
            self.performance_stats["tier_3_completions"] += 1
            
            logger.info(f"✅ TIER 3 COMPLETE: '{llm_answer['answer']}' ({processing_time:.2f}s)")
            
            return {
                "answer": llm_answer["answer"],
                "data_source": llm_answer["data_source"],
                "reasoning": llm_answer["reasoning"],
                "status": "success",
                "performance_metrics": {
                    "processing_time_seconds": processing_time,
                    "tier_exit": 3,
                    "confidence_score": combined_confidence,
                    "early_exit": False
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ LLM answer generation failed: {e}")
            
            return {
                "answer": "Unable to generate answer",
                "data_source": "error",
                "reasoning": f"Error occurred: {str(e)}",
                "status": "error",
                "performance_metrics": {
                    "processing_time_seconds": processing_time,
                    "tier_exit": 0,
                    "confidence_score": 0.0,
                    "early_exit": False
                }
            }
    
    async def _search_resume_vectors(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Search resume vectors in Redis"""
        try:
            results = self.embedding_service.search_similar_by_document_type(
                query=query,
                user_id=user_id,
                document_type="resume",
                top_k=3
            )
            logger.info(f"📄 Resume search: {len(results)} results found")
            return results
        except Exception as e:
            logger.error(f"❌ Resume vector search failed: {e}")
            return []
    
    async def _search_personal_info_vectors(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Search personal info vectors in Redis"""
        try:
            results = self.embedding_service.search_similar_by_document_type(
                query=query,
                user_id=user_id,
                document_type="personal_info",
                top_k=3
            )
            logger.info(f"📝 Personal info search: {len(results)} results found")
            return results
        except Exception as e:
            logger.error(f"❌ Personal info vector search failed: {e}")
            return []
    
    def _calculate_confidence(self, results: List[Dict[str, Any]], field_label: str) -> float:
        """Calculate confidence score for search results"""
        if not results:
            return 0.0
        
        # Base confidence on similarity scores and relevance
        scores = [result.get("score", 0.0) for result in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Boost confidence for exact matches
        field_lower = field_label.lower()
        for result in results:
            text = result.get("text", "").lower()
            if any(keyword in text for keyword in ["email", "phone", "name", "address"]):
                if any(keyword in field_lower for keyword in ["email", "phone", "name", "address"]):
                    avg_score *= 1.2  # 20% boost for keyword matches
        
        return min(avg_score, 1.0)  # Cap at 1.0
    
    def _calculate_combined_confidence(
        self, 
        resume_results: List[Dict[str, Any]], 
        personal_results: List[Dict[str, Any]], 
        field_label: str
    ) -> float:
        """Calculate confidence for combined results"""
        resume_conf = self._calculate_confidence(resume_results, field_label)
        personal_conf = self._calculate_confidence(personal_results, field_label)
        
        # Weight based on field type
        field_lower = field_label.lower()
        
        if any(keyword in field_lower for keyword in ["contact", "email", "phone", "address", "salary", "authorization"]):
            # Personal info fields get higher weight for personal data
            return (resume_conf * 0.3) + (personal_conf * 0.7)
        else:
            # Professional fields get higher weight for resume data
            return (resume_conf * 0.7) + (personal_conf * 0.3)
    
    def _extract_answer_from_results(self, results: List[Dict[str, Any]], field_label: str) -> Optional[str]:
        """Extract direct answer from search results"""
        if not results:
            return None
        
        # Simple extraction logic - can be enhanced
        best_result = max(results, key=lambda x: x.get("score", 0.0))
        text = best_result.get("text", "")
        
        # Field-specific extraction
        field_lower = field_label.lower()
        
        if "email" in field_lower:
            # Extract email pattern
            import re
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                return emails[0]
        
        elif "phone" in field_lower:
            # Extract phone pattern
            import re
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phones = re.findall(phone_pattern, text)
            if phones:
                return phones[0]
        
        elif "name" in field_lower:
            # Extract name (first line or first few words)
            lines = text.strip().split('\n')
            if lines:
                first_line = lines[0].strip()
                # If it looks like a name (2-3 words, no special chars)
                words = first_line.split()
                if 2 <= len(words) <= 3 and all(word.isalpha() for word in words):
                    return first_line
        
        # For other fields, return most relevant sentence
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip()
        
        return text[:100] + "..." if len(text) > 100 else text
    
    def _extract_answer_from_combined_results(
        self, 
        resume_results: List[Dict[str, Any]], 
        personal_results: List[Dict[str, Any]], 
        field_label: str
    ) -> Optional[str]:
        """Extract answer from combined results"""
        # Try personal info first for contact fields
        field_lower = field_label.lower()
        
        if any(keyword in field_lower for keyword in ["contact", "email", "phone", "address", "salary"]):
            answer = self._extract_answer_from_results(personal_results, field_label)
            if answer:
                return answer
        
        # Fall back to resume
        return self._extract_answer_from_results(resume_results, field_label)
    
    def _determine_primary_source(
        self, 
        answer: str, 
        resume_results: List[Dict[str, Any]], 
        personal_results: List[Dict[str, Any]]
    ) -> str:
        """Determine which data source provided the answer"""
        # Check if answer appears in resume results
        for result in resume_results:
            if answer.lower() in result.get("text", "").lower():
                return "resume_vectordb"
        
        # Check if answer appears in personal results
        for result in personal_results:
            if answer.lower() in result.get("text", "").lower():
                return "personal_info_vectordb"
        
        return "combined_data"
    
    async def _generate_llm_answer(
        self,
        field_label: str,
        resume_results: List[Dict[str, Any]],
        personal_results: List[Dict[str, Any]],
        field_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate answer using LLM with context"""
        
        # Prepare context
        resume_context = "\n".join([result.get("text", "") for result in resume_results[:2]])
        personal_context = "\n".join([result.get("text", "") for result in personal_results[:2]])
        
        prompt = f"""
You are a professional form filling assistant. Generate a concise, accurate answer for this form field.

FIELD QUESTION: {field_label}

AVAILABLE DATA:

RESUME DATA:
{resume_context if resume_context else "No resume data available"}

PERSONAL INFO DATA:
{personal_context if personal_context else "No personal info data available"}

INSTRUCTIONS:
1. Use the available data to provide the most accurate answer
2. If data is insufficient, generate a professional, realistic response
3. Keep answers concise and appropriate for form fields
4. For contact info (email, phone), use exact data if available
5. For experience/skills, summarize from resume data
6. For preferences/authorization, use personal info data

Provide only the answer, no explanation.
"""

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional form filling assistant. Provide concise, accurate answers based on the available data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Determine data source
            data_source = "generated"
            if resume_context and any(word in answer.lower() for word in resume_context.lower().split()[:10]):
                data_source = "resume_vectordb"
            elif personal_context and any(word in answer.lower() for word in personal_context.lower().split()[:10]):
                data_source = "personal_info_vectordb"
            
            return {
                "answer": answer,
                "data_source": data_source,
                "reasoning": f"Generated using LLM with available context data (source: {data_source})"
            }
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return {
                "answer": "Unable to generate answer",
                "data_source": "error",
                "reasoning": f"LLM generation failed: {str(e)}"
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total = self.performance_stats["total_requests"]
        if total == 0:
            return self.performance_stats
        
        return {
            **self.performance_stats,
            "tier_1_exit_rate": self.performance_stats["tier_1_exits"] / total,
            "tier_2_exit_rate": self.performance_stats["tier_2_exits"] / total,
            "tier_3_completion_rate": self.performance_stats["tier_3_completions"] / total,
        } 