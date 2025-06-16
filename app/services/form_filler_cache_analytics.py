import os
import time
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import json
from datetime import datetime
from openai import AsyncOpenAI
from loguru import logger

class FormFillerWithCacheAnalytics:
    """
    🔍 FORM FILLER WITH DETAILED CACHE ANALYTICS FOR LEARNING
    
    This version provides detailed logging to understand:
    - Which cache is being used
    - Why requests are fast or slow
    - Performance impact of each cache level
    - Vector search optimization
    """
    
    def __init__(self, openai_api_key: str, resume_extractor=None, personal_info_extractor=None):
        self.openai_api_key = openai_api_key
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # Service instances
        self.resume_extractor = resume_extractor
        self.personal_info_extractor = personal_info_extractor
        
        # Multi-level cache system
        self._search_cache = {}
        self._cache_size = 100
        
        # Detailed cache statistics for learning
        self.cache_stats = {
            'resume_cache_hits': 0,
            'resume_cache_misses': 0,
            'personal_cache_hits': 0,
            'personal_cache_misses': 0,
            'total_requests': 0,
            'vector_searches_performed': 0,
            'vector_searches_avoided': 0,
            'total_time_saved': 0.0
        }
        
        logger.info("🔍 Form Filler with DETAILED CACHE ANALYTICS initialized")

    def log_cache_analysis(self, operation: str, cache_type: str, is_hit: bool, time_taken: float, details: str = ""):
        """
        🔍 DETAILED CACHE ANALYSIS LOGGING
        This helps you understand exactly what's happening with caching
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
        logger.info(f"   📝 What happened: {details}")
        
        if is_hit:
            logger.info(f"   💡 WHY FAST: Data was already in memory cache")
            logger.info(f"   🚀 BENEFIT: Avoided expensive vector database search")
            self.cache_stats['total_time_saved'] += 2.0  # Estimate 2s saved per cache hit
        else:
            logger.info(f"   💡 WHY SLOW: Had to search vector database + compute embeddings")
            logger.info(f"   📈 FUTURE: This result is now cached for next time")
        
        # Show current cache statistics
        logger.info(f"   📊 CACHE STATS:")
        logger.info(f"      • {cache_type} Hits: {self.cache_stats.get(f'{cache_type}_hits', 0)}")
        logger.info(f"      • {cache_type} Misses: {self.cache_stats.get(f'{cache_type}_misses', 0)}")
        logger.info(f"      • Total Time Saved: {self.cache_stats['total_time_saved']:.1f}s")
        logger.info(f"{'='*100}")

    async def generate_field_values_with_analytics(self, fields: List[Dict], user_data: Dict, user_id: str = "default") -> Dict[str, Any]:
        """
        🚀 Generate field values with detailed cache performance analytics
        """
        start_time = time.time()
        self.cache_stats['total_requests'] += 1
        
        logger.info(f"")
        logger.info(f"🚀 STARTING REQUEST #{self.cache_stats['total_requests']} WITH CACHE ANALYTICS")
        logger.info(f"📊 User: {user_id} | Fields: {len(fields)}")
        
        # Extract field purposes
        field_purposes = []
        for field in fields:
            purpose = field.get("field_purpose", field.get("name", field.get("selector", "unknown")))
            field_purposes.append(purpose)
        
        # Generate search queries
        search_queries = self._generate_search_queries(field_purposes)
        logger.info(f"🔍 Generated search queries: {search_queries}")
        
        # TIER 1: Resume Vector Database Search
        logger.info(f"")
        logger.info(f"🎯 TIER 1: Resume Vector Database Search")
        resume_start = time.time()
        resume_data = await self._search_resume_with_analytics(search_queries, user_id)
        resume_time = time.time() - resume_start
        
        # TIER 2: Personal Info Vector Database Search
        logger.info(f"")
        logger.info(f"🎯 TIER 2: Personal Info Vector Database Search")
        personal_start = time.time()
        personal_data = await self._search_personal_info_with_analytics(search_queries, user_id)
        personal_time = time.time() - personal_start
        
        # Combine data
        combined_data = {
            "provided_user_data": user_data,
            "resume_vectordb_data": resume_data,
            "personal_info_vectordb_data": personal_data
        }
        
        # TIER 3: LLM Processing
        logger.info(f"")
        logger.info(f"🎯 TIER 3: LLM Field Mapping")
        llm_start = time.time()
        field_mappings = await self._generate_field_mappings(fields, combined_data)
        llm_time = time.time() - llm_start
        
        total_time = time.time() - start_time
        
        # COMPREHENSIVE PERFORMANCE REPORT
        self._log_performance_report(resume_time, personal_time, llm_time, total_time)
        
        return {
            "status": "success",
            "values": field_mappings,
            "processing_time": total_time,
            "cache_analytics": self._get_cache_analytics()
        }

    async def _search_resume_with_analytics(self, search_queries: List[str], user_id: str) -> Dict[str, Any]:
        """Resume search with detailed cache analytics"""
        cache_key = f"resume_{user_id}_{hash(tuple(str(q) for q in search_queries))}"
        
        # Check cache first
        if cache_key in self._search_cache:
            self.cache_stats['resume_cache_hits'] += 1
            self.cache_stats['vector_searches_avoided'] += len(search_queries)
            
            self.log_cache_analysis(
                "Resume Vector Search",
                "resume_cache",
                True,  # Cache hit
                0.001,  # Nearly instant
                f"Found cached resume data for user {user_id}. Avoided {len(search_queries)} vector searches!"
            )
            
            return self._search_cache[cache_key]
        
        # Cache miss - perform vector search
        self.cache_stats['resume_cache_misses'] += 1
        self.cache_stats['vector_searches_performed'] += len(search_queries)
        search_start = time.time()
        
        logger.info(f"🔍 PERFORMING VECTOR SEARCHES (Cache Miss)")
        logger.info(f"   • Queries: {search_queries}")
        logger.info(f"   • Vector searches needed: {len(search_queries)}")
        
        try:
            if not self.resume_extractor:
                search_time = time.time() - search_start
                self.log_cache_analysis(
                    "Resume Vector Search",
                    "resume_cache",
                    False,
                    search_time,
                    "Resume extractor not available"
                )
                return {"status": "not_found", "message": "Resume extractor not available"}
            
            self.resume_extractor.user_id = user_id
            
            # Perform each vector search
            all_results = []
            for i, query in enumerate(search_queries[:3]):
                vector_start = time.time()
                try:
                    result = self.resume_extractor.search_resume(query, k=2)
                    vector_time = time.time() - vector_start
                    
                    logger.info(f"   Vector Search {i+1}: '{query}' → {vector_time:.3f}s")
                    
                    if result and "results" in result:
                        all_results.extend(result["results"])
                        logger.info(f"      Found {len(result['results'])} results")
                    else:
                        logger.info(f"      No results found")
                        
                except Exception as e:
                    logger.warning(f"   Vector Search {i+1} failed: {e}")
            
            # Process results
            if all_results:
                unique_content = []
                seen_content = set()
                
                for result in all_results[:8]:
                    content = result.get("content", "").strip() if isinstance(result, dict) else str(result).strip()
                    if content and content not in seen_content and len(content) > 10:
                        unique_content.append(content)
                        seen_content.add(content)
                
                resume_data = {
                    "status": "found",
                    "unique_content_pieces": len(unique_content),
                    "content": " ".join(unique_content[:6]),
                    "search_queries_used": search_queries,
                    "data_source": "resume_vectordb"
                }
                
                # Cache the result for future use
                if len(self._search_cache) >= self._cache_size:
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]
                
                self._search_cache[cache_key] = resume_data
                search_time = time.time() - search_start
                
                self.log_cache_analysis(
                    "Resume Vector Search",
                    "resume_cache",
                    False,  # Cache miss
                    search_time,
                    f"Performed {len(search_queries)} vector searches, found {len(unique_content)} unique pieces. Result cached for future requests."
                )
                
                return resume_data
            else:
                search_time = time.time() - search_start
                self.log_cache_analysis(
                    "Resume Vector Search",
                    "resume_cache",
                    False,
                    search_time,
                    f"Performed {len(search_queries)} vector searches but found no resume data"
                )
                return {"status": "not_found", "message": "No resume data available"}
                
        except Exception as e:
            search_time = time.time() - search_start
            self.log_cache_analysis(
                "Resume Vector Search",
                "resume_cache",
                False,
                search_time,
                f"Error during vector search: {str(e)}"
            )
            return {"status": "error", "error": str(e)}

    async def _search_personal_info_with_analytics(self, search_queries: List[str], user_id: str) -> Dict[str, Any]:
        """Personal info search with detailed cache analytics"""
        personal_queries = ["contact information", "work authorization", "salary expectations"]
        cache_key = f"personal_{user_id}_{hash(tuple(personal_queries))}"
        
        # Check cache first
        if cache_key in self._search_cache:
            self.cache_stats['personal_cache_hits'] += 1
            self.cache_stats['vector_searches_avoided'] += len(personal_queries)
            
            self.log_cache_analysis(
                "Personal Info Vector Search",
                "personal_cache",
                True,  # Cache hit
                0.001,  # Nearly instant
                f"Found cached personal info for user {user_id}. Avoided {len(personal_queries)} vector searches!"
            )
            
            return self._search_cache[cache_key]
        
        # Cache miss - perform vector search
        self.cache_stats['personal_cache_misses'] += 1
        self.cache_stats['vector_searches_performed'] += len(personal_queries)
        search_start = time.time()
        
        logger.info(f"🔍 PERFORMING VECTOR SEARCHES (Cache Miss)")
        logger.info(f"   • Queries: {personal_queries}")
        logger.info(f"   • Vector searches needed: {len(personal_queries)}")
        
        try:
            if not self.personal_info_extractor:
                search_time = time.time() - search_start
                self.log_cache_analysis(
                    "Personal Info Vector Search",
                    "personal_cache",
                    False,
                    search_time,
                    "Personal info extractor not available"
                )
                return {"status": "not_found", "message": "Personal info extractor not available"}
            
            self.personal_info_extractor.user_id = user_id
            
            # Perform each vector search
            all_results = []
            for i, query in enumerate(personal_queries):
                vector_start = time.time()
                try:
                    result = self.personal_info_extractor.search_personal_info(query, k=2)
                    vector_time = time.time() - vector_start
                    
                    logger.info(f"   Vector Search {i+1}: '{query}' → {vector_time:.3f}s")
                    
                    if result and "results" in result:
                        all_results.extend(result["results"])
                        logger.info(f"      Found {len(result['results'])} results")
                    else:
                        logger.info(f"      No results found")
                        
                except Exception as e:
                    logger.warning(f"   Vector Search {i+1} failed: {e}")
            
            # Process results
            if all_results:
                unique_content = []
                seen_content = set()
                
                for result in all_results[:6]:
                    content = result.get("content", "").strip() if isinstance(result, dict) else str(result).strip()
                    if content and content not in seen_content and len(content) > 5:
                        unique_content.append(content)
                        seen_content.add(content)
                
                personal_data = {
                    "status": "found",
                    "unique_content_pieces": len(unique_content),
                    "content": " ".join(unique_content[:5]),
                    "search_queries_used": personal_queries,
                    "data_source": "personal_info_vectordb"
                }
                
                # Cache the result for future use
                if len(self._search_cache) >= self._cache_size:
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]
                
                self._search_cache[cache_key] = personal_data
                search_time = time.time() - search_start
                
                self.log_cache_analysis(
                    "Personal Info Vector Search",
                    "personal_cache",
                    False,  # Cache miss
                    search_time,
                    f"Performed {len(personal_queries)} vector searches, found {len(unique_content)} unique pieces. Result cached for future requests."
                )
                
                return personal_data
            else:
                search_time = time.time() - search_start
                self.log_cache_analysis(
                    "Personal Info Vector Search",
                    "personal_cache",
                    False,
                    search_time,
                    f"Performed {len(personal_queries)} vector searches but found no personal info"
                )
                return {"status": "not_found", "message": "No personal info data available"}
                
        except Exception as e:
            search_time = time.time() - search_start
            self.log_cache_analysis(
                "Personal Info Vector Search",
                "personal_cache",
                False,
                search_time,
                f"Error during vector search: {str(e)}"
            )
            return {"status": "error", "error": str(e)}

    def _log_performance_report(self, resume_time: float, personal_time: float, llm_time: float, total_time: float):
        """Log comprehensive performance report"""
        logger.info(f"")
        logger.info(f"📊 COMPREHENSIVE PERFORMANCE REPORT")
        logger.info(f"{'='*80}")
        logger.info(f"⏱️  TIMING BREAKDOWN:")
        logger.info(f"   • Resume Search: {resume_time:.3f}s")
        logger.info(f"   • Personal Search: {personal_time:.3f}s")
        logger.info(f"   • LLM Processing: {llm_time:.3f}s")
        logger.info(f"   • Total Time: {total_time:.3f}s")
        
        # Cache efficiency
        total_hits = self.cache_stats['resume_cache_hits'] + self.cache_stats['personal_cache_hits']
        total_misses = self.cache_stats['resume_cache_misses'] + self.cache_stats['personal_cache_misses']
        hit_rate = (total_hits / (total_hits + total_misses)) * 100 if (total_hits + total_misses) > 0 else 0
        
        logger.info(f"")
        logger.info(f"🎯 CACHE EFFICIENCY:")
        logger.info(f"   • Overall Hit Rate: {hit_rate:.1f}%")
        logger.info(f"   • Resume Cache: {self.cache_stats['resume_cache_hits']} hits, {self.cache_stats['resume_cache_misses']} misses")
        logger.info(f"   • Personal Cache: {self.cache_stats['personal_cache_hits']} hits, {self.cache_stats['personal_cache_misses']} misses")
        
        logger.info(f"")
        logger.info(f"⚡ OPTIMIZATION IMPACT:")
        logger.info(f"   • Vector Searches Performed: {self.cache_stats['vector_searches_performed']}")
        logger.info(f"   • Vector Searches Avoided: {self.cache_stats['vector_searches_avoided']}")
        logger.info(f"   • Estimated Time Saved: {self.cache_stats['total_time_saved']:.1f}s")
        
        # Performance trend
        if self.cache_stats['total_requests'] == 1:
            trend = "First request - establishing baseline"
        elif hit_rate > 50:
            trend = "EXCELLENT - High cache efficiency"
        elif hit_rate > 25:
            trend = "GOOD - Cache warming up"
        else:
            trend = "BUILDING - Performance will improve"
        
        logger.info(f"   • Performance Trend: {trend}")
        logger.info(f"{'='*80}")

    def _generate_search_queries(self, field_purposes: List[str]) -> List[str]:
        """Generate optimized search queries"""
        queries = []
        
        # Name-related fields
        if any(keyword in ' '.join(field_purposes).lower() for keyword in ['name', 'full name']):
            queries.extend(['name', 'full name'])
        
        # Contact-related fields
        if any(keyword in ' '.join(field_purposes).lower() for keyword in ['email', 'phone', 'contact']):
            queries.extend(['email', 'phone', 'contact'])
        
        # Default queries if none specific
        if not queries:
            queries = ['professional experience', 'contact information']
        
        return list(set(queries))[:4]  # Remove duplicates, limit to 4

    async def _generate_field_mappings(self, fields: List[Dict], combined_data: Dict) -> List[Dict]:
        """Generate field mappings using LLM"""
        try:
            # Create prompt
            prompt = self._create_prompt(fields, combined_data)
            
            # LLM call
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a form filling assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            field_mappings = self._parse_response(response_text, fields)
            
            # Log data usage
            resume_used = sum(1 for m in field_mappings if m.get('data_source') == 'resume_vectordb')
            personal_used = sum(1 for m in field_mappings if m.get('data_source') == 'personal_info_vectordb')
            generated = sum(1 for m in field_mappings if m.get('data_source') == 'generated')
            
            logger.info(f"🧠 LLM Processing Complete:")
            logger.info(f"   • Resume Data Used: {resume_used} fields")
            logger.info(f"   • Personal Data Used: {personal_used} fields")
            logger.info(f"   • Generated: {generated} fields")
            
            return field_mappings
            
        except Exception as e:
            logger.error(f"❌ LLM processing failed: {e}")
            return [{"field": f.get("name", "unknown"), "value": "", "data_source": "error"} for f in fields]

    def _create_prompt(self, fields: List[Dict], combined_data: Dict) -> str:
        """Create prompt for LLM"""
        parts = ["Fill these form fields using the provided data:"]
        
        # Add data sources
        resume_data = combined_data.get("resume_vectordb_data", {})
        if resume_data.get("status") == "found":
            parts.append(f"\nResume Data: {resume_data.get('content', '')}")
        
        personal_data = combined_data.get("personal_info_vectordb_data", {})
        if personal_data.get("status") == "found":
            parts.append(f"\nPersonal Info: {personal_data.get('content', '')}")
        
        # Add fields
        parts.append("\nFields:")
        for i, field in enumerate(fields):
            name = field.get("name", field.get("selector", f"field_{i}"))
            parts.append(f"{i+1}. {name}")
        
        parts.append("\nFormat: 1. [answer] 2. [answer] etc.")
        return "\n".join(parts)

    def _parse_response(self, response_text: str, fields: List[Dict]) -> List[Dict]:
        """Parse LLM response"""
        mappings = []
        lines = response_text.strip().split('\n')
        
        for i, field in enumerate(fields):
            name = field.get("name", field.get("selector", f"field_{i}"))
            value = ""
            data_source = "generated"
            
            # Extract value
            for line in lines:
                if line.strip().startswith(f"{i+1}."):
                    value = line.split(".", 1)[1].strip()
                    break
            
            # Determine source
            if any(known in value for known in ["Eric Abram", "ericabram33@gmail.com", "312-805-9851"]):
                data_source = "resume_vectordb"
            
            mappings.append({
                "field": name,
                "value": value,
                "data_source": data_source,
                "reasoning": f"Data from {data_source}"
            })
        
        return mappings

    def _get_cache_analytics(self) -> Dict[str, Any]:
        """Get comprehensive cache analytics"""
        total_hits = self.cache_stats['resume_cache_hits'] + self.cache_stats['personal_cache_hits']
        total_misses = self.cache_stats['resume_cache_misses'] + self.cache_stats['personal_cache_misses']
        hit_rate = (total_hits / (total_hits + total_misses)) * 100 if (total_hits + total_misses) > 0 else 0
        
        return {
            "cache_hit_rate": hit_rate,
            "total_requests": self.cache_stats['total_requests'],
            "vector_searches_performed": self.cache_stats['vector_searches_performed'],
            "vector_searches_avoided": self.cache_stats['vector_searches_avoided'],
            "time_saved_estimate": self.cache_stats['total_time_saved'],
            "cache_breakdown": {
                "resume_cache": {
                    "hits": self.cache_stats['resume_cache_hits'],
                    "misses": self.cache_stats['resume_cache_misses']
                },
                "personal_cache": {
                    "hits": self.cache_stats['personal_cache_hits'],
                    "misses": self.cache_stats['personal_cache_misses']
                }
            }
        } 