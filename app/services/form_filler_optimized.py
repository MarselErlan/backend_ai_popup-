#!/usr/bin/env python3
"""
Optimized Form Filler - High-Performance AI-powered intelligent form filling service
Uses cached extractors, connection pooling, and optimized search operations
"""

import os
import json
import asyncio
from functools import lru_cache
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from openai import AsyncOpenAI
from loguru import logger


class FormFillerOptimized:
    """Optimized Form Filler with performance enhancements"""
    
    def __init__(self, openai_api_key: str, resume_extractor=None, personal_info_extractor=None, headless: bool = True):
        self.openai_api_key = openai_api_key
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.headless = headless
        
        # Use provided cached extractors
        self.resume_extractor = resume_extractor
        self.personal_info_extractor = personal_info_extractor
        
        # Performance tracking
        self.cache_hits = 0
        self.database_queries = 0
        
        # LRU cache for search results
        self._search_cache = {}
        self._cache_size = 100
        
        logger.info("✅ Optimized Form Filler initialized")
    
    @lru_cache(maxsize=50)
    def _get_search_queries(self, field_purposes_tuple: Tuple[str, ...]) -> List[str]:
        """Cached search query generation"""
        field_purposes = list(field_purposes_tuple)
        
        # Optimized purpose mapping with priority
        purpose_mapping = {
            "name": ["name", "full name"],
            "email": ["email", "contact"],
            "phone": ["phone", "contact"],
            "location": ["location", "address"],
            "company": ["company", "employer"],
            "position": ["position", "title"],
            "experience": ["experience", "work history"],
            "skills": ["skills", "expertise"],
            "education": ["education", "degree"],
            "linkedin": ["linkedin", "profile"],
            "github": ["github", "portfolio"],
            "salary": ["salary", "compensation"],
            "work_authorization": ["work authorization", "visa"],
            "cover_letter": ["summary", "objective"],
        }
        
        search_queries = []
        for purpose in field_purposes:
            purpose_lower = purpose.lower()
            for key, terms in purpose_mapping.items():
                if any(term in purpose_lower for term in [key] + terms):
                    search_queries.extend(terms[:2])
                    break
        
        # Remove duplicates and limit queries
        unique_queries = list(dict.fromkeys(search_queries))[:4]  # Reduced from 5 to 4
        
        return unique_queries or ["professional experience", "skills"]
    
    async def generate_field_values_optimized(self, fields: List[Dict], user_data: Dict, user_id: str = "default") -> Dict[str, Any]:
        """
        🚀 OPTIMIZED 3-TIER DATA RETRIEVAL SYSTEM
        
        Performance improvements:
        1. Cached search queries
        2. Parallel vector searches
        3. Reduced database queries
        4. Optimized content processing
        """
        start_time = datetime.now()
        
        # Step 1: Extract field purposes and cache queries
        field_purposes = []
        for field in fields:
            purpose = field.get("field_purpose", field.get("name", field.get("selector", "unknown")))
            field_purposes.append(purpose)
        
        logger.info(f"🔍 Starting OPTIMIZED 3-tier data retrieval for {len(fields)} fields")
        
        # Step 2: Get cached search queries
        field_purposes_tuple = tuple(field_purposes)
        search_queries = self._get_search_queries(field_purposes_tuple)
        
        # Step 3: Parallel vector database searches
        resume_data, personal_data = await asyncio.gather(
            self._search_resume_vectordb_optimized(search_queries, user_id),
            self._search_personal_info_vectordb_optimized(search_queries, user_id),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(resume_data, Exception):
            logger.error(f"Resume search failed: {resume_data}")
            resume_data = {"status": "error", "error": str(resume_data)}
        
        if isinstance(personal_data, Exception):
            logger.error(f"Personal info search failed: {personal_data}")
            personal_data = {"status": "error", "error": str(personal_data)}
        
        # Step 4: Combine all available data
        combined_data = {
            "provided_user_data": user_data,
            "resume_vectordb_data": resume_data,
            "personal_info_vectordb_data": personal_data
        }
        
        # Step 5: Quick data assessment
        data_assessment = self._assess_data_completeness_optimized(fields, combined_data)
        
        logger.info(f"📊 Data Assessment: {data_assessment['summary']}")
        
        # Step 6: Generate field values using optimized LLM call
        field_mappings = await self._generate_llm_field_mappings_optimized(fields, combined_data, data_assessment)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"⚡ Optimized field generation completed in {processing_time:.2f}s")
        
        return {
            "status": "success",
            "values": field_mappings,
            "processing_time": processing_time,
            "cache_hits": self.cache_hits,
            "database_queries": self.database_queries,
            "optimization_metrics": {
                "search_queries_generated": len(search_queries),
                "parallel_searches": 2,
                "data_sources_used": data_assessment.get("sources_count", 0)
            }
        }
    
    async def _search_resume_vectordb_optimized(self, search_queries: List[str], user_id: str) -> Dict[str, Any]:
        """🎯 Optimized resume vector database search"""
        try:
            # Check cache first
            cache_key = f"resume_{user_id}_{hash(tuple(search_queries))}"
            if cache_key in self._search_cache:
                self.cache_hits += 1
                logger.info(f"✅ Resume cache hit for user: {user_id}")
                return self._search_cache[cache_key]
            
            if not self.resume_extractor:
                return {"status": "not_found", "message": "Resume extractor not available"}
            
            # Set user context
            self.resume_extractor.user_id = user_id
            
            logger.info(f"🔍 Optimized resume search queries: {search_queries}")
            
            # Parallel search with reduced k value for better performance
            search_tasks = []
            for query in search_queries[:3]:  # Limit to 3 queries max
                search_tasks.append(self.resume_extractor.search_resume(query, k=2))  # Reduced from 3 to 2
            
            # Execute searches in parallel
            all_results = []
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for result in search_results:
                if isinstance(result, Exception):
                    logger.warning(f"Resume search failed: {result}")
                    continue
                
                if result and "results" in result:
                    all_results.extend(result["results"])
            
            self.database_queries += len(search_tasks)
            
            # Process results efficiently
            if all_results:
                unique_content = []
                seen_content = set()
                
                for result in all_results[:8]:  # Limit processing to 8 results
                    content = result.get("content", "").strip() if isinstance(result, dict) else str(result).strip()
                    
                    if content and content not in seen_content and len(content) > 10:
                        unique_content.append(content)
                        seen_content.add(content)
                
                resume_data = {
                    "status": "found",
                    "unique_content_pieces": len(unique_content),
                    "content": " ".join(unique_content[:6]),  # Limit content
                    "search_queries_used": search_queries,
                    "data_source": "resume_vectordb"
                }
                
                # Cache the result
                if len(self._search_cache) >= self._cache_size:
                    # Remove oldest item
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]
                
                self._search_cache[cache_key] = resume_data
                
                logger.info(f"✅ Resume data found: {len(unique_content)} unique pieces (cached)")
                return resume_data
            else:
                logger.warning("⚠️ No resume data found in vector database")
                return {"status": "not_found", "message": "No resume data available"}
                
        except Exception as e:
            logger.error(f"❌ Optimized resume vector search failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _search_personal_info_vectordb_optimized(self, search_queries: List[str], user_id: str) -> Dict[str, Any]:
        """🎯 Optimized personal info vector database search"""
        try:
            # Check cache first
            cache_key = f"personal_{user_id}_{hash(tuple(search_queries))}"
            if cache_key in self._search_cache:
                self.cache_hits += 1
                logger.info(f"✅ Personal info cache hit for user: {user_id}")
                return self._search_cache[cache_key]
            
            if not self.personal_info_extractor:
                return {"status": "not_found", "message": "Personal info extractor not available"}
            
            # Set user context
            self.personal_info_extractor.user_id = user_id
            
            # Targeted personal info queries
            personal_queries = ["contact information", "work authorization", "salary expectations"]
            
            logger.info(f"🔍 Optimized personal info search queries: {personal_queries}")
            
            # Parallel search
            search_tasks = []
            for query in personal_queries:
                search_tasks.append(self.personal_info_extractor.search_personal_info(query, k=2))
            
            all_results = []
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for result in search_results:
                if isinstance(result, Exception):
                    logger.warning(f"Personal info search failed: {result}")
                    continue
                
                if result and "results" in result:
                    all_results.extend(result["results"])
            
            self.database_queries += len(search_tasks)
            
            # Process results efficiently
            if all_results:
                unique_content = []
                seen_content = set()
                
                for result in all_results[:6]:  # Limit processing
                    content = result.get("content", "").strip() if isinstance(result, dict) else str(result).strip()
                    
                    if content and content not in seen_content and len(content) > 5:
                        unique_content.append(content)
                        seen_content.add(content)
                
                personal_data = {
                    "status": "found",
                    "unique_content_pieces": len(unique_content),
                    "content": " ".join(unique_content[:5]),  # Limit content
                    "search_queries_used": personal_queries,
                    "data_source": "personal_info_vectordb"
                }
                
                # Cache the result
                if len(self._search_cache) >= self._cache_size:
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]
                
                self._search_cache[cache_key] = personal_data
                
                logger.info(f"✅ Personal info data found: {len(unique_content)} unique pieces (cached)")
                return personal_data
            else:
                logger.warning("⚠️ No personal info data found in vector database")
                return {"status": "not_found", "message": "No personal info data available"}
                
        except Exception as e:
            logger.error(f"❌ Optimized personal info vector search failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _assess_data_completeness_optimized(self, fields: List[Dict], combined_data: Dict) -> Dict[str, Any]:
        """📊 Optimized data completeness assessment"""
        resume_data = combined_data.get("resume_vectordb_data", {})
        personal_data = combined_data.get("personal_info_vectordb_data", {})
        user_data = combined_data.get("provided_user_data", {})
        
        sources_count = 0
        quality_parts = []
        
        if resume_data.get("status") == "found":
            sources_count += 1
            quality_parts.append(f"Resume: {resume_data.get('unique_content_pieces', 0)} pieces")
        
        if personal_data.get("status") == "found":
            sources_count += 1
            quality_parts.append(f"Personal: {personal_data.get('unique_content_pieces', 0)} pieces")
        
        if user_data:
            sources_count += 1
            quality_parts.append(f"User: {len(user_data)} fields")
        
        # Quick quality assessment
        if sources_count >= 2:
            quality = "EXCELLENT"
            generation_need = "MINIMAL"
        elif sources_count == 1:
            quality = "GOOD"
            generation_need = "MODERATE"
        else:
            quality = "LIMITED"
            generation_need = "SIGNIFICANT"
        
        summary = f"{quality} - {generation_need} generation needed | Sources: {', '.join(quality_parts)}"
        
        return {
            "summary": summary,
            "sources_count": sources_count,
            "quality": quality,
            "generation_need": generation_need
        }
    
    async def _generate_llm_field_mappings_optimized(self, fields: List[Dict], combined_data: Dict, data_assessment: Dict) -> List[Dict]:
        """🧠 Optimized LLM field mapping generation"""
        try:
            resume_data = combined_data.get("resume_vectordb_data", {})
            personal_data = combined_data.get("personal_info_vectordb_data", {})
            user_data = combined_data.get("provided_user_data", {})
            
            # Optimized prompt - shorter and more focused
            prompt = f"""
            You are a professional form filler. Generate field mappings based on available data.
            
            FIELDS TO FILL:
            {json.dumps(fields, indent=2)}
            
            AVAILABLE DATA:
            Resume Data: {resume_data.get('content', 'Not available')[:500]}...
            Personal Data: {personal_data.get('content', 'Not available')[:300]}...
            User Data: {json.dumps(user_data, indent=2) if user_data else 'Not available'}
            
            DATA ASSESSMENT: {data_assessment['summary']}
            
            INSTRUCTIONS:
            - Prioritize real data from vector databases
            - Generate professional content only when needed
            - For file uploads: use action "skip" and value null
            - Be concise and accurate
            
            RESPONSE FORMAT (JSON):
            {{
                "field_mappings": [
                    {{
                        "selector": "css_selector",
                        "field_type": "input_type",
                        "field_purpose": "purpose",
                        "value": "value_or_null",
                        "action": "fill|skip",
                        "data_source": "resume_vectordb|personal_info_vectordb|generated",
                        "reasoning": "brief_explanation"
                    }}
                ]
            }}
            """
            
            # Optimized OpenAI call with shorter max_tokens
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use faster model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,  # Reduced from default
                temperature=0.3,  # Lower temperature for consistency
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_content = response.choices[0].message.content
            parsed_response = json.loads(response_content)
            
            field_mappings = parsed_response.get("field_mappings", [])
            
            # Log generation results
            data_sources = [mapping.get("data_source", "unknown") for mapping in field_mappings]
            source_counts = {
                "resume": data_sources.count("resume_vectordb"),
                "personal": data_sources.count("personal_info_vectordb"),
                "generated": data_sources.count("generated")
            }
            
            logger.info(f"🧠 Optimized LLM field mapping completed:")
            logger.info(f"   📊 Data Usage: Resume={source_counts['resume']}, Personal={source_counts['personal']}, Generated={source_counts['generated']}")
            
            return field_mappings
            
        except Exception as e:
            logger.error(f"❌ Optimized LLM field mapping failed: {e}")
            return [{
                "selector": fields[0].get("selector", "#unknown") if fields else "#unknown",
                "field_type": "text",
                "field_purpose": "error",
                "value": "",
                "action": "skip",
                "data_source": "error",
                "reasoning": f"LLM generation failed: {str(e)}"
            }]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "cache_hits": self.cache_hits,
            "database_queries": self.database_queries,
            "cache_size": len(self._search_cache),
            "cache_hit_ratio": self.cache_hits / max(1, self.cache_hits + self.database_queries)
        }
    
    def clear_cache(self):
        """Clear the search cache"""
        self._search_cache.clear()
        logger.info("🧹 Search cache cleared") 