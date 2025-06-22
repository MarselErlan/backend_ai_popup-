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
from loguru import logger

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
        logger.info(f"{'='*80}")
        logger.info(f"🔍 CACHE ANALYSIS: {operation}")
        logger.info(f"{'='*80}")
        logger.info(f"   📊 Cache Type: {cache_type}")
        logger.info(f"   {status_emoji} Result: {'CACHE HIT' if is_hit else 'CACHE MISS'}")
        logger.info(f"   {speed_emoji} Speed: {'FAST (cached)' if is_hit else 'SLOW (vector search)'}")
        logger.info(f"   ⏱️  Time: {time_taken:.3f} seconds")
        logger.info(f"   📝 Details: {details}")
        
        if is_hit:
            logger.info(f"   💡 BENEFIT: Avoided expensive vector database search")
            self.cache_stats['total_time_saved'] += 1.5  # Estimate 1.5s saved per cache hit
        else:
            logger.info(f"   📈 FUTURE: This result is now cached for next time")
        
        # Update cache hit rate
        total_cache_requests = (self.cache_stats['resume_cache_hits'] + self.cache_stats['resume_cache_misses'] + 
                               self.cache_stats['personal_cache_hits'] + self.cache_stats['personal_cache_misses'] +
                               self.cache_stats['answer_cache_hits'] + self.cache_stats['answer_cache_misses'])
        
        if total_cache_requests > 0:
            total_hits = (self.cache_stats['resume_cache_hits'] + self.cache_stats['personal_cache_hits'] + 
                         self.cache_stats['answer_cache_hits'])
            self.cache_stats['cache_hit_rate'] = total_hits / total_cache_requests
        
        logger.info(f"   📊 CACHE STATS: Hit Rate: {self.cache_stats['cache_hit_rate']:.1%}")
        logger.info(f"{'='*80}")

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
        
        logger.info(f"🧠 Generating answer for: '{field_label}' (User: {user_id})")
        
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
            
            return cached_answer
        
        self.cache_stats['answer_cache_misses'] += 1
        
        try:
            # Generate optimized search queries
            search_queries = self._generate_search_queries(field_label)
            logger.info(f"🔍 Generated search queries: {search_queries}")
            
            # TIER 1: Search Resume Vector Store with caching
            logger.info("🎯 TIER 1: Resume Vector Search with Advanced Caching")
            resume_results = await self._search_resume_vectors_cached(search_queries, user_id)
            
            # Check if TIER 1 is sufficient
            tier1_confidence = self._calculate_confidence(resume_results, field_label)
            
            if tier1_confidence >= 0.8:
                # Early exit with resume data
                answer = self._extract_answer_from_results(resume_results, field_label)
                if answer:
                    processing_time = time.time() - start_time
                    self.performance_stats["tier_1_exits"] += 1
                    
                    logger.info(f"✅ TIER 1 EXIT: '{answer}' (confidence: {tier1_confidence:.2f})")
                    
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
                    
                    return result
            
            # TIER 2: Search Personal Info Vector Store with caching
            logger.info("🎯 TIER 2: Personal Info Vector Search with Advanced Caching")
            personal_results = await self._search_personal_info_vectors_cached(search_queries, user_id)
            
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
                    
                    return result
            
            # TIER 3: LLM Generation with Context
            logger.info("🎯 TIER 3: Enhanced LLM Generation")
            llm_answer = await self._generate_llm_answer_enhanced(
                field_label, resume_results, personal_results, field_context
            )
            
            processing_time = time.time() - start_time
            self.performance_stats["tier_3_completions"] += 1
            
            logger.info(f"✅ TIER 3 COMPLETE: '{llm_answer['answer']}' ({processing_time:.2f}s)")
            
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
                    "cache_stats": self._get_cache_analytics()
                }
            }
            
            # Cache the final result
            self._cache_answer(answer_cache_key, result)
            
            return result
            
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