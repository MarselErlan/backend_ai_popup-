"""
🧠 Redis-Enabled LLM Service for Smart Form Filling
Integrates with Redis Vector Store for fast, intelligent field answers
Enhanced with Advanced Caching System for Maximum Performance
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from functools import lru_cache
import openai
from app.utils.logger import logger

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import RedisVectorStore

class RedisLLMService:
    """
    🧠 Redis-powered LLM service for intelligent form filling
    
    Features:
    - 3-tier data retrieval (Resume Redis → Personal Info Redis → LLM Generation)
    - Early exit optimization (skip tiers when sufficient data found)
    - Advanced multi-level caching system for maximum speed
    - Smart field analysis and contextual understanding
    - Comprehensive performance tracking and analytics
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key
        
        # Initialize Redis-enabled services
        self.embedding_service = EmbeddingService()
        self.vector_store = RedisVectorStore()
        
        # Advanced multi-level cache system
        self._search_cache = {}
        self._answer_cache = {}  # Cache for final answers
        self._cache_size = 200  # Increased cache size
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "tier_1_exits": 0,
            "tier_2_exits": 0,
            "tier_3_completions": 0,
            "avg_processing_time": 0.0
        }
        
        # Advanced cache analytics
        self.cache_stats = {
            'resume_cache_hits': 0,
            'resume_cache_misses': 0,
            'personal_cache_hits': 0,
            'personal_cache_misses': 0,
            'answer_cache_hits': 0,
            'answer_cache_misses': 0,
            'total_requests': 0,
            'vector_searches_performed': 0,
            'vector_searches_avoided': 0,
            'total_time_saved': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info("🧠 Redis LLM Service initialized with advanced caching and vector store integration")

    def log_cache_analysis(self, operation: str, cache_type: str, is_hit: bool, time_taken: float, details: str = ""):
        """
        🔍 DETAILED CACHE ANALYSIS LOGGING
        This helps understand exactly what's happening with caching performance
        """
        status_emoji = "🎯" if is_hit else "❌"
        speed_emoji = "⚡" if is_hit else "🐌"
        
        logger.info(f"")
        logger.info(f"{'='*100}")
        logger.info(f"🔍 CACHE ANALYSIS: {operation}")
        logger.info(f"{'='*100}")
        logger.info(f"   📊 Cache Type: {cache_type}")
        logger.info(f"   {status_emoji} Result: {'CACHE HIT' if is_hit else 'CACHE MISS'}")
        logger.info(f"   {speed_emoji} Speed: {'FAST (cached)' if is_hit else 'SLOW (vector search)'}")
        logger.info(f"   ⏱️  Time: {time_taken:.3f} seconds")
        logger.info(f"   📝 Details: {details}")
        
        if is_hit:
            logger.info(f"   💡 WHY FAST: Data was already in memory cache")
            logger.info(f"   🚀 BENEFIT: Avoided expensive vector database search + embedding computation")
            self.cache_stats['total_time_saved'] += 1.5  # Estimate 1.5s saved per cache hit
        else:
            logger.info(f"   💡 WHY SLOW: Had to search Redis vector database + compute embeddings")
            logger.info(f"   📈 FUTURE: This result is now cached for next time")
        
        # Update cache hit rate
        total_cache_requests = (self.cache_stats['resume_cache_hits'] + self.cache_stats['resume_cache_misses'] + 
                               self.cache_stats['personal_cache_hits'] + self.cache_stats['personal_cache_misses'] +
                               self.cache_stats['answer_cache_hits'] + self.cache_stats['answer_cache_misses'])
        
        if total_cache_requests > 0:
            total_hits = (self.cache_stats['resume_cache_hits'] + self.cache_stats['personal_cache_hits'] + 
                         self.cache_stats['answer_cache_hits'])
            self.cache_stats['cache_hit_rate'] = total_hits / total_cache_requests
        
        # Show current cache statistics
        logger.info(f"   📊 CURRENT CACHE STATS:")
        logger.info(f"      • {cache_type} Hits: {self.cache_stats.get(f'{cache_type}_hits', 0)}")
        logger.info(f"      • {cache_type} Misses: {self.cache_stats.get(f'{cache_type}_misses', 0)}")
        logger.info(f"      • Overall Hit Rate: {self.cache_stats['cache_hit_rate']:.1%}")
        logger.info(f"      • Total Time Saved: {self.cache_stats['total_time_saved']:.1f}s")
        logger.info(f"      • Vector Searches Avoided: {self.cache_stats['vector_searches_avoided']}")
        logger.info(f"{'='*100}")

    def _generate_cache_key(self, query: str, user_id: str, cache_type: str) -> str:
        """Generate optimized cache key"""
        # Normalize query for better cache hits
        normalized_query = query.lower().strip()
        return f"{cache_type}_{user_id}_{hash(normalized_query)}"

    def _generate_search_queries(self, field_label: str) -> List[str]:
        """
        🔍 Generate optimized search queries based on field label
        Similar to form_filler_cache_analytics approach
        """
        field_lower = field_label.lower()
        queries = [field_label]  # Always include original
        
        # Field-specific query expansion
        if "email" in field_lower:
            queries.extend(["email address", "contact information", "email"])
        elif "phone" in field_lower:
            queries.extend(["phone number", "contact information", "mobile"])
        elif "name" in field_lower:
            queries.extend(["full name", "contact information", "name"])
        elif "address" in field_lower:
            queries.extend(["address", "location", "residence"])
        elif "company" in field_lower or "employer" in field_lower:
            queries.extend(["work experience", "employer", "company"])
        elif "position" in field_lower or "title" in field_lower:
            queries.extend(["job title", "position", "role"])
        elif "experience" in field_lower:
            queries.extend(["work experience", "professional background"])
        elif "skill" in field_lower:
            queries.extend(["skills", "technical skills", "expertise"])
        elif "education" in field_lower:
            queries.extend(["education", "degree", "university"])
        elif "salary" in field_lower:
            queries.extend(["salary expectations", "compensation"])
        elif "authorization" in field_lower or "visa" in field_lower:
            queries.extend(["work authorization", "visa status"])
        else:
            # Generic professional queries
            queries.extend(["professional background", "work experience"])
        
        return list(set(queries))[:3]  # Remove duplicates, limit to 3

    async def generate_field_answer(
        self,
        field_label: str,
        user_id: str,
        field_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        🎯 Generate intelligent field answer using Redis-powered 3-tier system with advanced caching
        
        Args:
            field_label: The form field label/question
            user_id: User identifier for vector search
            field_context: Additional context about the field
            
        Returns:
            Dict with answer, data_source, reasoning, and performance metrics
        """
        start_time = time.time()
        self.performance_stats["total_requests"] += 1
        self.cache_stats["total_requests"] += 1
        
        # Log detailed request start
        self._log_request_start(field_label, user_id, self.cache_stats["total_requests"])
        
        # Check answer cache first (fastest possible response)
        answer_cache_key = self._generate_cache_key(field_label, user_id, "answer")
        if answer_cache_key in self._answer_cache:
            self.cache_stats['answer_cache_hits'] += 1
            cached_answer = self._answer_cache[answer_cache_key]
            processing_time = time.time() - start_time
            
            self.log_cache_analysis(
                "Field Answer Generation",
                "answer_cache",
                True,
                processing_time,
                f"Found cached answer for '{field_label}' - instant response!"
            )
            
            # Update performance metrics
            cached_answer["performance_metrics"]["processing_time_seconds"] = processing_time
            cached_answer["performance_metrics"]["cache_hit"] = True
            
            logger.info(f"🎯 CACHE HIT - Returning instant answer: '{cached_answer['answer']}'")
            
            return cached_answer
        
        self.cache_stats['answer_cache_misses'] += 1
        
        try:
            # Generate optimized search queries
            search_queries = self._generate_search_queries(field_label)
            self._log_query_generation_analysis(field_label, search_queries)
            
            # TIER 1: Search Resume Vector Store with caching
            logger.info("🎯 TIER 1: Resume Vector Search with Advanced Caching")
            tier1_start = time.time()
            resume_results = await self._search_resume_vectors_cached(search_queries, user_id)
            tier1_time = time.time() - tier1_start
            
            # Check if TIER 1 is sufficient
            tier1_confidence = self._calculate_confidence(resume_results, field_label)
            tier1_exit = tier1_confidence >= 0.8
            
            self._log_tier_analysis(1, tier1_confidence, len(resume_results), tier1_time, tier1_exit)
            
            if tier1_exit:
                # Early exit with resume data
                answer = self._extract_answer_from_results(resume_results, field_label)
                self._log_answer_extraction_analysis(resume_results, field_label, answer)
                
                if answer:
                    processing_time = time.time() - start_time
                    self.performance_stats["tier_1_exits"] += 1
                    
                    logger.info(f"✅ TIER 1 EARLY EXIT SUCCESS")
                    
                    result = {
                        "answer": answer,
                        "data_source": "resume_vectordb",
                        "reasoning": f"Found high-confidence answer in resume data (confidence: {tier1_confidence:.2f})",
                        "status": "success",
                        "performance_metrics": {
                            "processing_time_seconds": processing_time,
                            "tier_exit": 1,
                            "confidence_score": tier1_confidence,
                            "early_exit": True,
                            "cache_hit": False,
                            "cache_stats": self._get_cache_analytics()
                        }
                    }
                    
                    # Cache the result for future requests
                    self._cache_answer(answer_cache_key, result)
                    
                    # Log comprehensive performance report
                    self._log_comprehensive_performance_report(processing_time, 1, field_label)
                    
                    return result
            
            # TIER 2: Search Personal Info Vector Store with caching
            logger.info("🎯 TIER 2: Personal Info Vector Search with Advanced Caching")
            tier2_start = time.time()
            personal_results = await self._search_personal_info_vectors_cached(search_queries, user_id)
            tier2_time = time.time() - tier2_start
            
            # Check if TIER 1 + TIER 2 is sufficient
            combined_confidence = self._calculate_combined_confidence(
                resume_results, personal_results, field_label
            )
            tier2_exit = combined_confidence >= 0.8
            
            self._log_tier_analysis(2, combined_confidence, len(resume_results) + len(personal_results), tier2_time, tier2_exit)
            
            if tier2_exit:
                # Early exit with combined data
                answer = self._extract_answer_from_combined_results(
                    resume_results, personal_results, field_label
                )
                self._log_answer_extraction_analysis(resume_results + personal_results, field_label, answer)
                
                if answer:
                    processing_time = time.time() - start_time
                    self.performance_stats["tier_2_exits"] += 1
                    
                    data_source = self._determine_primary_source(answer, resume_results, personal_results)
                    
                    logger.info(f"✅ TIER 2 EARLY EXIT SUCCESS")
                    
                    result = {
                        "answer": answer,
                        "data_source": data_source,
                        "reasoning": f"Found high-confidence answer in combined data (confidence: {combined_confidence:.2f})",
                        "status": "success",
                        "performance_metrics": {
                            "processing_time_seconds": processing_time,
                            "tier_exit": 2,
                            "confidence_score": combined_confidence,
                            "early_exit": True,
                            "cache_hit": False,
                            "cache_stats": self._get_cache_analytics()
                        }
                    }
                    
                    # Cache the result
                    self._cache_answer(answer_cache_key, result)
                    
                    # Log comprehensive performance report
                    self._log_comprehensive_performance_report(processing_time, 2, field_label)
                    
                    return result
            
            # TIER 3: Enhanced LLM Generation
            logger.info("🎯 TIER 3: Enhanced LLM Generation with Premium Quality")
            tier3_start = time.time()
            llm_answer = await self._generate_llm_answer_enhanced(
                field_label, resume_results, personal_results, field_context
            )
            tier3_time = time.time() - tier3_start
            
            processing_time = time.time() - start_time
            self.performance_stats["tier_3_completions"] += 1
            
            # Log LLM generation analysis
            resume_context = "\n".join([result.get("text", "") for result in resume_results[:3]])
            personal_context = "\n".join([result.get("text", "") for result in personal_results[:3]])
            self._log_llm_generation_analysis(field_label, resume_context, personal_context, llm_answer["answer"], llm_answer["data_source"])
            
            logger.info(f"✅ TIER 3 FULL PROCESSING COMPLETE")
            
            result = {
                "answer": llm_answer["answer"],
                "data_source": llm_answer["data_source"],
                "reasoning": llm_answer["reasoning"],
                "status": "success",
                "performance_metrics": {
                    "processing_time_seconds": processing_time,
                    "tier_exit": 3,
                    "confidence_score": combined_confidence,
                    "early_exit": False,
                    "cache_hit": False,
                    "cache_stats": self._get_cache_analytics(),
                    "tier_breakdown": {
                        "tier_1_time": tier1_time,
                        "tier_2_time": tier2_time,
                        "tier_3_time": tier3_time
                    }
                }
            }
            
            # Cache the final result
            self._cache_answer(answer_cache_key, result)
            
            # Log comprehensive performance report
            self._log_comprehensive_performance_report(processing_time, 3, field_label)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ LLM answer generation failed: {e}")
            
            # Log error analysis
            logger.info(f"")
            logger.info(f"💥 ERROR ANALYSIS")
            logger.info(f"{'='*50}")
            logger.info(f"   ❌ Error Type: {type(e).__name__}")
            logger.info(f"   📝 Error Message: {str(e)}")
            logger.info(f"   ⏱️  Time Before Error: {processing_time:.3f}s")
            logger.info(f"{'='*50}")
            
            return {
                "answer": "Unable to generate answer",
                "data_source": "error",
                "reasoning": f"Error occurred: {str(e)}",
                "status": "error",
                "performance_metrics": {
                    "processing_time_seconds": processing_time,
                    "tier_exit": 0,
                    "confidence_score": 0.0,
                    "early_exit": False,
                    "cache_hit": False,
                    "cache_stats": self._get_cache_analytics()
                }
            }

    def _cache_answer(self, cache_key: str, result: Dict[str, Any]):
        """Cache the final answer result"""
        # Manage cache size
        if len(self._answer_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._answer_cache))
            del self._answer_cache[oldest_key]
        
        self._answer_cache[cache_key] = result
        logger.debug(f"📦 Cached answer for future use (cache size: {len(self._answer_cache)})")

    async def _search_resume_vectors_cached(self, queries: List[str], user_id: str) -> List[Dict[str, Any]]:
        """Search resume vectors with advanced caching"""
        cache_key = self._generate_cache_key("_".join(queries), user_id, "resume")
        
        # Check cache first
        if cache_key in self._search_cache:
            self.cache_stats['resume_cache_hits'] += 1
            self.cache_stats['vector_searches_avoided'] += len(queries)
            
            self.log_cache_analysis(
                "Resume Vector Search",
                "resume_cache",
                True,
                0.001,
                f"Found cached resume data for user {user_id}. Avoided {len(queries)} vector searches!"
            )
            
            return self._search_cache[cache_key]
        
        # Cache miss - perform vector search
        self.cache_stats['resume_cache_misses'] += 1
        self.cache_stats['vector_searches_performed'] += len(queries)
        search_start = time.time()
        
        try:
            all_results = []
            for query in queries:
                results = self.embedding_service.search_similar_by_document_type(
                    query=query,
                    user_id=user_id,
                    document_type="resume",
                    top_k=2
                )
                all_results.extend(results)
            
            search_time = time.time() - search_start
            
            self.log_cache_analysis(
                "Resume Vector Search",
                "resume_cache",
                False,
                search_time,
                f"Performed {len(queries)} vector searches, found {len(all_results)} results"
            )
            
            # Cache the results
            self._cache_search_results(cache_key, all_results)
            
            logger.info(f"📄 Resume search: {len(all_results)} results found")
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Resume vector search failed: {e}")
            return []

    async def _search_personal_info_vectors_cached(self, queries: List[str], user_id: str) -> List[Dict[str, Any]]:
        """Search personal info vectors with advanced caching"""
        cache_key = self._generate_cache_key("_".join(queries), user_id, "personal")
        
        # Check cache first
        if cache_key in self._search_cache:
            self.cache_stats['personal_cache_hits'] += 1
            self.cache_stats['vector_searches_avoided'] += len(queries)
            
            self.log_cache_analysis(
                "Personal Info Vector Search",
                "personal_cache",
                True,
                0.001,
                f"Found cached personal info for user {user_id}. Avoided {len(queries)} vector searches!"
            )
            
            return self._search_cache[cache_key]
        
        # Cache miss - perform vector search
        self.cache_stats['personal_cache_misses'] += 1
        self.cache_stats['vector_searches_performed'] += len(queries)
        search_start = time.time()
        
        try:
            all_results = []
            for query in queries:
                results = self.embedding_service.search_similar_by_document_type(
                    query=query,
                    user_id=user_id,
                    document_type="personal_info",
                    top_k=2
                )
                all_results.extend(results)
            
            search_time = time.time() - search_start
            
            self.log_cache_analysis(
                "Personal Info Vector Search",
                "personal_cache",
                False,
                search_time,
                f"Performed {len(queries)} vector searches, found {len(all_results)} results"
            )
            
            # Cache the results
            self._cache_search_results(cache_key, all_results)
            
            logger.info(f"📝 Personal info search: {len(all_results)} results found")
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Personal info vector search failed: {e}")
            return []

    def _cache_search_results(self, cache_key: str, results: List[Dict[str, Any]]):
        """Cache search results with size management"""
        # Manage cache size
        if len(self._search_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]
        
        self._search_cache[cache_key] = results
        logger.debug(f"📦 Cached search results (cache size: {len(self._search_cache)})")

    # Keep existing methods for backward compatibility
    async def _search_resume_vectors(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Legacy method - redirects to cached version"""
        return await self._search_resume_vectors_cached([query], user_id)

    async def _search_personal_info_vectors(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Legacy method - redirects to cached version"""
        return await self._search_personal_info_vectors_cached([query], user_id)

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

    async def _generate_llm_answer_enhanced(
        self,
        field_label: str,
        resume_results: List[Dict[str, Any]],
        personal_results: List[Dict[str, Any]],
        field_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhanced LLM generation with better prompting (inspired by form_filler quality)"""
        
        # Prepare context
        resume_context = "\n".join([result.get("text", "") for result in resume_results[:3]])
        personal_context = "\n".join([result.get("text", "") for result in personal_results[:3]])
        
        # Enhanced prompt with form_filler-style sophistication
        prompt = f"""
You are a professional form filling expert with advanced contextual understanding.
You excel at analyzing form fields and generating intelligent, professional responses.

🎯 FIELD TO FILL: {field_label}

📊 AVAILABLE DATA SOURCES (prioritized):

1. RESUME/PROFESSIONAL DATA (HIGHEST PRIORITY):
{resume_context if resume_context else "No resume data available"}

2. PERSONAL INFO DATA (SECOND PRIORITY):
{personal_context if personal_context else "No personal info data available"}

🧠 INTELLIGENT FILLING INSTRUCTIONS:
- **ALWAYS prioritize real data from vector databases over generated content**
- **Use resume data for**: work experience, skills, education, professional background, job titles
- **Use personal info data for**: contact details, work authorization, salary expectations, preferences
- **Only generate content when**: real data is missing or insufficient for the specific field
- **Maintain consistency**: Ensure all responses align with available real data

🔧 FIELD ANALYSIS & RESPONSE GUIDELINES:
- **Contact Fields** (email, phone, address): Extract exact values from personal data
- **Professional Fields** (experience, skills, title): Use resume data, summarize professionally
- **Authorization Fields**: Use personal data, provide realistic work authorization responses
- **Salary Fields**: Use personal data or generate market-appropriate ranges
- **Name Fields**: Extract exact names from contact information
- **Company/Position**: Use most recent or relevant from resume data

⚡ RESPONSE REQUIREMENTS:
1. **Provide ONLY the raw field value** - no explanations or prefixes
2. **Be concise and form-appropriate** - single values for single fields
3. **Use exact data when available** - don't paraphrase contact info
4. **Professional formatting** - proper capitalization, phone format, etc.
5. **Realistic generation** - if generating, make it believable and professional

🚫 CRITICAL: Do NOT include:
- "FIELD ANSWER:" prefix
- "Answer:" prefix  
- Any labels or explanations
- Any formatting markers
- Quotes around the entire response

EXAMPLES:
- Field "First Name" → "John" (NOT "Answer: John")
- Field "Email Address" → "john.doe@email.com" (NOT "Email: john.doe@email.com")
- Field "Years of Experience" → "5" (NOT "5 years")
- Field "Current Company" → "Tech Solutions Inc" (NOT "Company: Tech Solutions Inc")

Generate the precise value that should go directly into the form field:
"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4-0125-preview",  # Use GPT-4 for better quality like form_filler
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional form filling expert with advanced contextual understanding. You analyze form fields and provide precise, professional responses based on available data. Always respond with just the raw field value."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Balanced for consistency and creativity
                max_tokens=200
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Enhanced cleanup with more comprehensive prefix removal
            prefixes_to_remove = [
                "FIELD ANSWER:", "Answer:", "Response:", "Field Answer:", "Value:", "Result:",
                field_label + ":", field_label.replace("?", "") + ":",
                "The answer is:", "The field value is:", "Field value:",
                "Response:", "Output:", "Generated answer:"
            ]
            
            for prefix in prefixes_to_remove:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
                    break
            
            # Remove quotes if the entire answer is wrapped in quotes
            if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
                answer = answer[1:-1].strip()
            
            # Determine data source with enhanced logic
            data_source = "generated"
            if resume_context:
                resume_words = set(resume_context.lower().split())
                answer_words = set(answer.lower().split())
                if resume_words & answer_words:  # If there's overlap
                    data_source = "resume_vectordb"
            
            if personal_context:
                personal_words = set(personal_context.lower().split())
                answer_words = set(answer.lower().split())
                if personal_words & answer_words:
                    data_source = "personal_info_vectordb"
            
            return {
                "answer": answer,
                "data_source": data_source,
                "reasoning": f"Enhanced LLM generation using available context data (source: {data_source})"
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced LLM generation failed: {e}")
            return {
                "answer": "Unable to generate answer",
                "data_source": "error",
                "reasoning": f"Enhanced LLM generation failed: {str(e)}"
            }

    def _get_cache_analytics(self) -> Dict[str, Any]:
        """Get comprehensive cache analytics"""
        total_cache_requests = (self.cache_stats['resume_cache_hits'] + self.cache_stats['resume_cache_misses'] + 
                               self.cache_stats['personal_cache_hits'] + self.cache_stats['personal_cache_misses'] +
                               self.cache_stats['answer_cache_hits'] + self.cache_stats['answer_cache_misses'])
        
        if total_cache_requests > 0:
            total_hits = (self.cache_stats['resume_cache_hits'] + self.cache_stats['personal_cache_hits'] + 
                         self.cache_stats['answer_cache_hits'])
            hit_rate = total_hits / total_cache_requests
        else:
            hit_rate = 0.0
        
        return {
            **self.cache_stats,
            "cache_hit_rate": hit_rate,
            "cache_sizes": {
                "search_cache": len(self._search_cache),
                "answer_cache": len(self._answer_cache)
            },
            "performance_improvement": f"{self.cache_stats['total_time_saved']:.1f}s saved"
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics including cache analytics"""
        total = self.performance_stats["total_requests"]
        if total == 0:
            return {**self.performance_stats, "cache_analytics": self._get_cache_analytics()}
        
        return {
            **self.performance_stats,
            "tier_1_exit_rate": self.performance_stats["tier_1_exits"] / total,
            "tier_2_exit_rate": self.performance_stats["tier_2_exits"] / total,
            "tier_3_completion_rate": self.performance_stats["tier_3_completions"] / total,
            "cache_analytics": self._get_cache_analytics()
        }

    def clear_cache(self):
        """Clear all caches (useful for testing or memory management)"""
        self._search_cache.clear()
        self._answer_cache.clear()
        logger.info("🧹 All caches cleared")

    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status and sizes"""
        return {
            "search_cache_size": len(self._search_cache),
            "answer_cache_size": len(self._answer_cache),
            "max_cache_size": self._cache_size,
            "cache_analytics": self._get_cache_analytics()
        }

    def _log_request_start(self, field_label: str, user_id: str, request_number: int):
        """Log detailed request start information"""
        logger.info(f"")
        logger.info(f"🚀 STARTING REQUEST #{request_number} WITH COMPREHENSIVE ANALYTICS")
        logger.info(f"{'='*100}")
        logger.info(f"📊 REQUEST DETAILS:")
        logger.info(f"   • Field: '{field_label}'")
        logger.info(f"   • User ID: {user_id}")
        logger.info(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   • Cache Status: {len(self._search_cache)} search entries, {len(self._answer_cache)} answer entries")
        logger.info(f"{'='*100}")

    def _log_tier_analysis(self, tier: int, confidence: float, results_count: int, time_taken: float, exit_decision: bool):
        """Log detailed tier analysis"""
        tier_emoji = "🎯" if tier == 1 else "🔍" if tier == 2 else "🧠"
        tier_name = "RESUME VECTOR SEARCH" if tier == 1 else "PERSONAL INFO SEARCH" if tier == 2 else "LLM GENERATION"
        
        logger.info(f"")
        logger.info(f"{tier_emoji} TIER {tier} ANALYSIS: {tier_name}")
        logger.info(f"{'='*80}")
        logger.info(f"   📊 Results Found: {results_count}")
        logger.info(f"   🎯 Confidence Score: {confidence:.2f} ({confidence*100:.1f}%)")
        logger.info(f"   ⏱️  Processing Time: {time_taken:.3f}s")
        logger.info(f"   🚪 Exit Decision: {'YES - Early Exit' if exit_decision else 'NO - Continue to next tier'}")
        
        if exit_decision:
            logger.info(f"   ✅ EARLY EXIT REASON: Confidence {confidence:.2f} >= 0.8 threshold")
            logger.info(f"   🚀 PERFORMANCE BENEFIT: Avoided processing in remaining tiers")
        else:
            logger.info(f"   ⏭️  CONTINUE REASON: Confidence {confidence:.2f} < 0.8 threshold")
            logger.info(f"   📈 STRATEGY: Gathering more data from next tier")
        
        logger.info(f"{'='*80}")

    def _log_comprehensive_performance_report(self, total_time: float, tier_exit: int, field_label: str):
        """Log comprehensive performance report similar to form_filler_cache_analytics"""
        logger.info(f"")
        logger.info(f"📊 COMPREHENSIVE PERFORMANCE REPORT")
        logger.info(f"{'='*100}")
        logger.info(f"🎯 REQUEST SUMMARY:")
        logger.info(f"   • Field: '{field_label}'")
        logger.info(f"   • Total Processing Time: {total_time:.3f}s")
        logger.info(f"   • Tier Exit Level: {tier_exit}")
        logger.info(f"   • Exit Strategy: {'Early Exit' if tier_exit < 3 else 'Full Processing'}")
        
        # Timing breakdown analysis
        logger.info(f"")
        logger.info(f"⏱️  PERFORMANCE ANALYSIS:")
        if tier_exit == 1:
            logger.info(f"   • Resume Search: {total_time:.3f}s")
            logger.info(f"   • Personal Search: SKIPPED (early exit)")
            logger.info(f"   • LLM Processing: SKIPPED (early exit)")
            logger.info(f"   • Efficiency: EXCELLENT - Found answer in Tier 1")
        elif tier_exit == 2:
            logger.info(f"   • Resume + Personal Search: {total_time:.3f}s")
            logger.info(f"   • LLM Processing: SKIPPED (early exit)")
            logger.info(f"   • Efficiency: GOOD - Found answer in Tier 2")
        else:
            logger.info(f"   • Full 3-Tier Processing: {total_time:.3f}s")
            logger.info(f"   • Efficiency: STANDARD - Required LLM generation")
        
        # Cache efficiency analysis
        total_hits = self.cache_stats['resume_cache_hits'] + self.cache_stats['personal_cache_hits'] + self.cache_stats['answer_cache_hits']
        total_misses = self.cache_stats['resume_cache_misses'] + self.cache_stats['personal_cache_misses'] + self.cache_stats['answer_cache_misses']
        hit_rate = (total_hits / (total_hits + total_misses)) * 100 if (total_hits + total_misses) > 0 else 0
        
        logger.info(f"")
        logger.info(f"🎯 CACHE EFFICIENCY REPORT:")
        logger.info(f"   • Overall Hit Rate: {hit_rate:.1f}%")
        logger.info(f"   • Resume Cache: {self.cache_stats['resume_cache_hits']} hits, {self.cache_stats['resume_cache_misses']} misses")
        logger.info(f"   • Personal Cache: {self.cache_stats['personal_cache_hits']} hits, {self.cache_stats['personal_cache_misses']} misses")
        logger.info(f"   • Answer Cache: {self.cache_stats['answer_cache_hits']} hits, {self.cache_stats['answer_cache_misses']} misses")
        
        logger.info(f"")
        logger.info(f"⚡ OPTIMIZATION IMPACT:")
        logger.info(f"   • Vector Searches Performed: {self.cache_stats['vector_searches_performed']}")
        logger.info(f"   • Vector Searches Avoided: {self.cache_stats['vector_searches_avoided']}")
        logger.info(f"   • Estimated Time Saved: {self.cache_stats['total_time_saved']:.1f}s")
        
        # Performance trend analysis
        total_requests = self.cache_stats['total_requests']
        if total_requests == 1:
            trend = "First request - establishing baseline"
            trend_emoji = "🏁"
        elif hit_rate > 70:
            trend = "EXCELLENT - High cache efficiency, optimal performance"
            trend_emoji = "🏆"
        elif hit_rate > 40:
            trend = "GOOD - Cache warming up, performance improving"
            trend_emoji = "📈"
        elif hit_rate > 15:
            trend = "BUILDING - Cache building, performance will improve"
            trend_emoji = "🔨"
        else:
            trend = "INITIAL - Performance baseline being established"
            trend_emoji = "🌱"
        
        logger.info(f"   • Performance Trend: {trend_emoji} {trend}")
        
        # Session statistics
        logger.info(f"")
        logger.info(f"📈 SESSION STATISTICS:")
        logger.info(f"   • Total Requests: {total_requests}")
        logger.info(f"   • Tier 1 Exits: {self.performance_stats['tier_1_exits']} ({(self.performance_stats['tier_1_exits']/total_requests)*100:.1f}%)")
        logger.info(f"   • Tier 2 Exits: {self.performance_stats['tier_2_exits']} ({(self.performance_stats['tier_2_exits']/total_requests)*100:.1f}%)")
        logger.info(f"   • Tier 3 Completions: {self.performance_stats['tier_3_completions']} ({(self.performance_stats['tier_3_completions']/total_requests)*100:.1f}%)")
        
        # Memory usage
        logger.info(f"")
        logger.info(f"💾 MEMORY USAGE:")
        logger.info(f"   • Search Cache Size: {len(self._search_cache)}/{self._cache_size}")
        logger.info(f"   • Answer Cache Size: {len(self._answer_cache)}/{self._cache_size}")
        logger.info(f"   • Memory Efficiency: {((len(self._search_cache) + len(self._answer_cache))/(self._cache_size*2))*100:.1f}% utilized")
        
        logger.info(f"{'='*100}")

    def _log_query_generation_analysis(self, field_label: str, generated_queries: List[str]):
        """Log detailed query generation analysis"""
        logger.info(f"")
        logger.info(f"🔍 SMART QUERY GENERATION ANALYSIS")
        logger.info(f"{'='*80}")
        logger.info(f"   📝 Original Field: '{field_label}'")
        logger.info(f"   🧠 Generated Queries: {len(generated_queries)} optimized queries")
        
        for i, query in enumerate(generated_queries, 1):
            logger.info(f"      {i}. '{query}'")
        
        # Query strategy analysis
        field_lower = field_label.lower()
        if any(keyword in field_lower for keyword in ["email", "phone", "name", "address"]):
            strategy = "CONTACT INFO STRATEGY - Prioritizing personal data extraction"
        elif any(keyword in field_lower for keyword in ["company", "position", "title", "experience", "skill"]):
            strategy = "PROFESSIONAL STRATEGY - Prioritizing resume data extraction"
        elif any(keyword in field_lower for keyword in ["salary", "authorization", "visa"]):
            strategy = "PREFERENCE STRATEGY - Prioritizing personal preferences"
        else:
            strategy = "GENERAL STRATEGY - Balanced professional and personal search"
        
        logger.info(f"   🎯 Query Strategy: {strategy}")
        logger.info(f"   📊 Expected Benefits: Multiple targeted searches for higher accuracy")
        logger.info(f"{'='*80}")

    def _log_answer_extraction_analysis(self, results: List[Dict[str, Any]], field_label: str, extracted_answer: Optional[str]):
        """Log detailed answer extraction analysis"""
        logger.info(f"")
        logger.info(f"🎯 ANSWER EXTRACTION ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"   📊 Input Results: {len(results)} vector search results")
        logger.info(f"   🎯 Target Field: '{field_label}'")
        
        if extracted_answer:
            logger.info(f"   ✅ Extracted Answer: '{extracted_answer}'")
            logger.info(f"   🧠 Extraction Method: Pattern matching + field-specific logic")
            
            # Show extraction reasoning
            field_lower = field_label.lower()
            if "email" in field_lower:
                logger.info(f"   📧 Strategy: Email regex pattern extraction")
            elif "phone" in field_lower:
                logger.info(f"   📞 Strategy: Phone number pattern extraction")
            elif "name" in field_lower:
                logger.info(f"   👤 Strategy: Name pattern recognition")
            else:
                logger.info(f"   📝 Strategy: Best result + sentence extraction")
        else:
            logger.info(f"   ❌ No Answer Extracted: Insufficient data in results")
            logger.info(f"   📈 Next Step: Will proceed to next tier or LLM generation")
        
        # Show top results for transparency
        if results:
            logger.info(f"   📊 Top Results Analysis:")
            for i, result in enumerate(results[:2], 1):
                score = result.get("score", 0.0)
                text_preview = result.get("text", "")[:50] + "..." if len(result.get("text", "")) > 50 else result.get("text", "")
                logger.info(f"      {i}. Score: {score:.3f} | Preview: '{text_preview}'")
        
        logger.info(f"{'='*60}")

    def _log_llm_generation_analysis(self, field_label: str, resume_context: str, personal_context: str, generated_answer: str, data_source: str):
        """Log detailed LLM generation analysis"""
        logger.info(f"")
        logger.info(f"🧠 LLM GENERATION ANALYSIS")
        logger.info(f"{'='*80}")
        logger.info(f"   🎯 Target Field: '{field_label}'")
        logger.info(f"   🤖 Model: GPT-4-0125-preview (Premium quality)")
        logger.info(f"   📊 Context Analysis:")
        
        resume_words = len(resume_context.split()) if resume_context else 0
        personal_words = len(personal_context.split()) if personal_context else 0
        
        logger.info(f"      • Resume Context: {resume_words} words {'✅' if resume_words > 0 else '❌'}")
        logger.info(f"      • Personal Context: {personal_words} words {'✅' if personal_words > 0 else '❌'}")
        
        context_quality = "EXCELLENT" if (resume_words + personal_words) > 100 else "GOOD" if (resume_words + personal_words) > 50 else "LIMITED"
        logger.info(f"      • Context Quality: {context_quality}")
        
        logger.info(f"   ✅ Generated Answer: '{generated_answer}'")
        logger.info(f"   📍 Data Source: {data_source.upper()}")
        
        # Answer quality analysis
        answer_length = len(generated_answer)
        if answer_length < 3:
            quality = "MINIMAL - Very short answer"
        elif answer_length < 20:
            quality = "CONCISE - Appropriate for form field"
        elif answer_length < 50:
            quality = "DETAILED - Good information content"
        else:
            quality = "COMPREHENSIVE - Rich information"
        
        logger.info(f"   📏 Answer Quality: {quality} ({answer_length} characters)")
        
        # Professional formatting check
        has_email = "@" in generated_answer and "." in generated_answer
        has_phone = any(char.isdigit() for char in generated_answer) and len([c for c in generated_answer if c.isdigit()]) >= 10
        is_capitalized = generated_answer and generated_answer[0].isupper()
        
        formatting_score = sum([has_email and "email" in field_label.lower(), 
                               has_phone and "phone" in field_label.lower(), 
                               is_capitalized])
        
        logger.info(f"   ✨ Formatting Quality: {'EXCELLENT' if formatting_score > 0 or is_capitalized else 'STANDARD'}")
        logger.info(f"{'='*80}") 