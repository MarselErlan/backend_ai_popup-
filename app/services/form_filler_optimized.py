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
import time
import re

from app.services.resume_extractor_optimized import ResumeExtractorOptimized
from app.services.personal_info_extractor_optimized import PersonalInfoExtractorOptimized
from app.services.document_service import DocumentService


class OptimizedFormFiller:
    """
    🚀 HIGH-PERFORMANCE FORM FILLER WITH EARLY EXIT OPTIMIZATION
    
    Features:
    - Multi-level LRU caching system
    - Early exit optimization (skip TIER 2/3 if TIER 1 is sufficient)
    - Satisfaction scoring for intelligent tier skipping
    - Real-time performance metrics
    - Progressive performance improvement
    """
    
    def __init__(self, openai_api_key: str, resume_extractor=None, personal_info_extractor=None, headless: bool = True):
        self.openai_api_key = openai_api_key
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # Service instances (will be cached)
        self.resume_extractor = resume_extractor
        self.personal_info_extractor = personal_info_extractor
        
        # Performance tracking
        self.cache_hits = 0
        self.database_queries = 0
        
        # Multi-level cache system
        self._search_cache = {}
        self._resume_cache = {}
        self._personal_cache = {}
        self._cache_size = 100
        
        # Detailed cache statistics for learning
        self.cache_stats = {
            'resume_cache_hits': 0,
            'resume_cache_misses': 0,
            'personal_cache_hits': 0,
            'personal_cache_misses': 0,
            'search_cache_hits': 0,
            'search_cache_misses': 0,
            'total_requests': 0,
            'vector_searches_performed': 0,
            'vector_searches_cached': 0,
            'early_exits': 0,  # New: Track early exits
            'tiers_skipped': 0  # New: Track how many tiers we skipped
        }
        
        logger.info("✅ Optimized Form Filler initialized with EARLY EXIT OPTIMIZATION")

    def _log_detailed_cache_performance(self, operation: str, cache_type: str, hit: bool, time_taken: float, details: str = ""):
        """🔍 DETAILED CACHE PERFORMANCE LOGGING FOR LEARNING"""
        status = "🎯 CACHE HIT" if hit else "❌ CACHE MISS"
        performance_impact = "⚡ FAST (0.001s)" if hit else f"🐌 SLOW ({time_taken:.3f}s)"
        
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"📊 CACHE ANALYSIS - {operation}")
        logger.info(f"{'='*80}")
        logger.info(f"   🏷️  Cache Type: {cache_type}")
        logger.info(f"   📈 Status: {status}")
        logger.info(f"   ⚡ Performance Impact: {performance_impact}")
        logger.info(f"   ⏱️  Time Taken: {time_taken:.3f}s")
        logger.info(f"   📝 Details: {details}")
        logger.info(f"   📊 Cache Stats:")
        logger.info(f"      • {cache_type} Hits: {self.cache_stats.get(f'{cache_type}_hits', 0)}")
        logger.info(f"      • {cache_type} Misses: {self.cache_stats.get(f'{cache_type}_misses', 0)}")
        
        # Show why this matters for performance
        if hit:
            logger.info(f"   💡 WHY FAST: Data retrieved from memory cache (no vector search needed)")
        else:
            logger.info(f"   💡 WHY SLOW: Had to perform vector database search + embedding computation")
        
        logger.info(f"{'='*80}")

    def calculate_satisfaction_score(self, field_label: str, resume_data: Dict[str, Any]) -> float:
        """
        🎯 ANSWER-FOCUSED SATISFACTION SCORING
        
        Returns satisfaction percentage (0-100):
        - 80%+ = Can extract actual answer, skip TIER 2 & 3
        - <80% = Cannot extract answer, need more tiers
        """
        if not resume_data or resume_data.get("status") != "found":
            return 0.0
        
        field_lower = field_label.lower().strip()
        content = resume_data.get("content", "")
        
        # 🎯 TRY TO EXTRACT ACTUAL ANSWER - Only give high score if we can extract it
        extracted_answer = self.extract_answer_from_tier1(field_label, resume_data)
        
        if extracted_answer:
            # We found an actual answer - high satisfaction!
            satisfaction_score = 95.0
            logger.info(f"🎯 ACTUAL ANSWER FOUND: '{extracted_answer}' - Satisfaction: {satisfaction_score}%")
            return satisfaction_score
        
        # 🎯 NO ANSWER FOUND - Check if we have relevant context for complex questions
        
        # For simple factual fields, if no answer found = 0% satisfaction
        simple_fields = ['name', 'email', 'phone', 'address', 'city', 'state', 'zip']
        if any(keyword in field_lower for keyword in simple_fields):
            logger.info(f"❌ NO ANSWER for simple field '{field_label}' - Satisfaction: 0%")
            return 0.0
        
        # For complex questions, check if we have relevant context
        content_lower = content.lower()
        
        # Experience/Work questions - check for substantial work content
        if any(keyword in field_lower for keyword in ['experience', 'work', 'job', 'position', 'startup', 'company']):
            work_indicators = ['experience', 'work', 'job', 'position', 'company', 'role', 'responsibilities']
            work_content_count = sum(1 for indicator in work_indicators if indicator in content_lower)
            if work_content_count >= 3:  # Substantial work content
                satisfaction_score = 60.0  # Some context, but not enough for early exit
                logger.info(f"📊 WORK CONTEXT found but no direct answer - Satisfaction: {satisfaction_score}%")
                return satisfaction_score
        
        # Technology/Skills questions
        elif any(keyword in field_lower for keyword in ['technology', 'skill', 'programming', 'software', 'computer']):
            tech_indicators = ['technology', 'programming', 'software', 'technical', 'skill', 'java', 'python']
            tech_content_count = sum(1 for indicator in tech_indicators if indicator in content_lower)
            if tech_content_count >= 2:  # Some tech content
                satisfaction_score = 50.0  # Minimal context
                logger.info(f"📊 TECH CONTEXT found but no direct answer - Satisfaction: {satisfaction_score}%")
                return satisfaction_score
        
        # Education questions
        elif any(keyword in field_lower for keyword in ['education', 'degree', 'school', 'university', 'college']):
            edu_indicators = ['education', 'degree', 'university', 'college', 'school', 'bachelor', 'master']
            edu_content_count = sum(1 for indicator in edu_indicators if indicator in content_lower)
            if edu_content_count >= 2:  # Some education content
                satisfaction_score = 60.0  # Some context
                logger.info(f"📊 EDUCATION CONTEXT found but no direct answer - Satisfaction: {satisfaction_score}%")
                return satisfaction_score
        
        # No relevant content found
        logger.info(f"❌ NO RELEVANT CONTENT for '{field_label}' - Satisfaction: 0%")
        return 0.0
    
    def _log_satisfaction_decision(self, field_label: str, satisfaction_score: float):
        """Log the satisfaction decision with clear reasoning"""
        if satisfaction_score >= 80.0:
            logger.info(f"📊 '{field_label}' satisfaction: {satisfaction_score}% → 🚀 EARLY EXIT")
        else:
            logger.info(f"📊 '{field_label}' satisfaction: {satisfaction_score}% → ➡️ CONTINUE")

    def extract_answer_from_tier1(self, field_label: str, resume_data: Dict[str, Any]) -> str:
        """
        🎯 EXTRACT ACTUAL ANSWER from TIER 1 data
        Returns the answer if found, empty string if not found
        """
        if not resume_data or resume_data.get("status") != "found":
            return ""
        
        field_lower = field_label.lower().strip()
        content = resume_data.get("content", "")
        
        # Name fields - extract actual names
        if any(keyword in field_lower for keyword in ['name', 'full name', 'first name', 'last name']):
            name_patterns = [
                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # First Last
                r'\b([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)\b',  # First M. Last
                r'name[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)',  # "Name: First Last"
                r'(Eric Abram)',  # Known name
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # Email fields - extract actual email addresses
        elif any(keyword in field_lower for keyword in ['email', 'e-mail', 'mail']):
            email_patterns = [
                r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',  # Standard email
                r'email[:\s]+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',  # "Email: address"
                r'(ericabram33@gmail\.com)',  # Known email
            ]
            
            for pattern in email_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # Phone fields - extract actual phone numbers
        elif any(keyword in field_lower for keyword in ['phone', 'telephone', 'mobile', 'cell']):
            phone_patterns = [
                r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b',  # 123-456-7890 or 123.456.7890 or 1234567890
                r'(\(\d{3}\)\s?\d{3}[-.]?\d{4})',   # (123) 456-7890
                r'phone[:\s]+(\d{3}[-.]?\d{3}[-.]?\d{4})',  # "Phone: 123-456-7890"
                r'(312-805-9851)',  # Known phone
            ]
            
            for pattern in phone_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # Address fields - extract address information
        elif any(keyword in field_lower for keyword in ['address', 'location', 'city', 'state', 'zip']):
            address_patterns = [
                r'\b(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln))\b',  # Street address
                r'\b([A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5})\b',  # City, ST 12345
                r'address[:\s]+(.+?)(?:\n|$)',  # "Address: ..."
                r'(San Francisco)',  # Known location
            ]
            
            for pattern in address_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # For complex questions, we can't extract simple answers from TIER 1
        # Return empty string to indicate no direct answer found
        return ""

    def calculate_combined_satisfaction_score(self, field_label: str, resume_data: Dict[str, Any], personal_data: Dict[str, Any]) -> float:
        """
        🎯 TIER 2 ANSWER-FOCUSED SATISFACTION SCORING
        """
        # Try to extract actual answer from combined data
        combined_answer = self.extract_answer_from_combined_data(field_label, resume_data, personal_data)
        
        if combined_answer:
            # We found an actual answer - high satisfaction!
            satisfaction_score = 95.0
            logger.info(f"🎯 ACTUAL ANSWER FOUND in TIER 2: '{combined_answer}' - Satisfaction: {satisfaction_score}%")
            return satisfaction_score
        
        # No answer found - check if we have relevant context
        tier1_score = self.calculate_satisfaction_score(field_label, resume_data)
        
        # If TIER 1 already found context, check if TIER 2 adds more context
        if tier1_score > 0 and personal_data and personal_data.get("status") == "found":
            personal_content = personal_data.get("content", "").lower()
            field_lower = field_label.lower().strip()
            
            # Check if TIER 2 has additional relevant context
            if any(keyword in field_lower for keyword in ['authorization', 'visa', 'work', 'salary', 'compensation']):
                relevant_terms = ['authorization', 'visa', 'work', 'salary', 'compensation', 'eligible', 'citizen']
                if any(term in personal_content for term in relevant_terms):
                    # Some additional context, but not enough for early exit
                    combined_score = min(tier1_score + 10.0, 75.0)  # Cap below 80% since no direct answer
                    logger.info(f"📊 TIER 2 CONTEXT: Additional context found - Combined: {combined_score}%")
                    return combined_score
        
        # No improvement from TIER 2
        logger.info(f"❌ NO ANSWER or additional context from TIER 2 - Satisfaction: {tier1_score}%")
        return tier1_score

    def extract_answer_from_combined_data(self, field_label: str, resume_data: Dict[str, Any], personal_data: Dict[str, Any]) -> str:
        """
        🎯 TIER 2 EARLY EXIT: Extract answer from combined TIER 1 + TIER 2 data
        """
        # First try to extract from TIER 1 (resume)
        tier1_answer = self.extract_answer_from_tier1(field_label, resume_data)
        if tier1_answer:
            return tier1_answer
        
        # If no answer from TIER 1, try TIER 2 (personal info)
        if not personal_data or personal_data.get("status") != "found":
            return ""
        
        content = personal_data.get("content", "")
        field_lower = field_label.lower().strip()
        
        # Work authorization extraction
        if any(keyword in field_lower for keyword in ['authorization', 'visa', 'work status', 'eligible']):
            if 'authorized' in content.lower():
                return "Yes, authorized to work"
            elif 'citizen' in content.lower():
                return "US Citizen"
            elif 'permanent resident' in content.lower():
                return "Permanent Resident"
        
        # Salary extraction
        elif any(keyword in field_lower for keyword in ['salary', 'compensation']):
            salary_match = re.search(r'\$[\d,]+', content)
            if salary_match:
                return salary_match.group(0)
        
        # Additional contact info extraction
        elif any(keyword in field_lower for keyword in ['phone', 'email']):
            if 'phone' in field_lower:
                phone_match = re.search(r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b', content)
                if phone_match:
                    return phone_match.group(1)
            elif 'email' in field_lower:
                email_match = re.search(r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b', content)
                if email_match:
                    return email_match.group(1)
        
        return ""

    def determine_data_source(self, answer: str, resume_data: Dict[str, Any], personal_data: Dict[str, Any]) -> str:
        """
        🎯 Determine which data source provided the answer
        """
        # Check if answer comes from resume data
        if resume_data and resume_data.get("status") == "found":
            resume_content = resume_data.get("content", "").lower()
            if answer.lower() in resume_content:
                return "resume_vectordb"
        
        # Check if answer comes from personal data
        if personal_data and personal_data.get("status") == "found":
            personal_content = personal_data.get("content", "").lower()
            if answer.lower() in personal_content:
                return "personal_info_vectordb"
        
        # Default to combined if unclear
        return "combined_data"

    async def generate_field_values_optimized(self, fields: List[Dict], user_data: Dict, user_id: str = "default") -> Dict[str, Any]:
        """
        🚀 OPTIMIZED 3-TIER DATA RETRIEVAL WITH EARLY EXIT OPTIMIZATION
        """
        start_time = time.time()
        self.cache_stats['total_requests'] += 1
        
        logger.info(f"")
        logger.info(f"🚀 STARTING OPTIMIZED REQUEST #{self.cache_stats['total_requests']} WITH EARLY EXIT")
        logger.info(f"📊 CACHE PERFORMANCE TRACKING: ENABLED")
        logger.info(f"🔍 Processing {len(fields)} fields for user: {user_id}")
        
        # Extract field purposes
        field_purposes = []
        for field in fields:
            purpose = field.get("field_purpose", field.get("name", field.get("selector", "unknown")))
            field_purposes.append(purpose)
        
        # Get cached search queries
        field_purposes_tuple = tuple(field_purposes)
        search_queries = self._get_search_queries(field_purposes_tuple)
        
        # TIER 1: Resume Vector Database Search (with detailed logging)
        logger.info(f"🎯 TIER 1: Resume Vector Database Search")
        resume_start = time.time()
        resume_data = await self._search_resume_vectordb_with_cache_analytics(search_queries, user_id)
        resume_time = time.time() - resume_start
        
        # 🚀 EARLY EXIT OPTIMIZATION: Check if TIER 1 is sufficient
        field_label = field_purposes[0] if field_purposes else "unknown"
        satisfaction_score = self.calculate_satisfaction_score(field_label, resume_data)
        self._log_satisfaction_decision(field_label, satisfaction_score)
        
        if satisfaction_score >= 80.0:
            # EARLY EXIT: TIER 1 is sufficient!
            self.cache_stats['early_exits'] += 1
            self.cache_stats['tiers_skipped'] += 2  # Skipped TIER 2 and TIER 3
            
            logger.info(f"🚀 TIER 1 EARLY EXIT - {satisfaction_score}% satisfaction")
            
            # Extract direct answer from TIER 1
            direct_answer = self.extract_answer_from_tier1(field_label, resume_data)
            
            if direct_answer:
                total_time = time.time() - start_time
                
                # Create field mapping for early exit
                field_mappings = [{
                    "field": field_label,
                    "value": direct_answer,
                    "data_source": "resume_vectordb",
                    "reasoning": f"Early exit optimization - {satisfaction_score}% satisfaction from resume data",
                    "action": "fill",
                    "early_exit": True
                }]
                
                logger.info(f"⏱️ TIER 1 EXIT: {total_time:.3f}s - Answer: '{direct_answer}'")
                
                # Cache efficiency report
                self._log_cache_efficiency_report()
                
                # Removed redundant log
                
                return {
                    "status": "success",
                    "values": field_mappings,
                    "processing_time": total_time,
                    "early_exit": True,
                    "tier_exit": 1,
                    "satisfaction_score": satisfaction_score,
                    "tiers_used": 1,
                    "tiers_skipped": 2,
                    "cache_analytics": {
                        "total_requests": self.cache_stats['total_requests'],
                        "cache_hit_rate": self._calculate_cache_hit_rate(),
                        "early_exits": self.cache_stats['early_exits'],
                        "tiers_skipped": self.cache_stats['tiers_skipped'],
                        "cache_breakdown": self.cache_stats
                    }
                }
        
        # Continue with normal 3-tier processing if satisfaction < 80%
        logger.info(f"➡️ CONTINUING TO TIER 2 (Satisfaction: {satisfaction_score}% < 80%)")
        
        # TIER 2: Personal Info Vector Database Search (with detailed logging)
        logger.info(f"🎯 TIER 2: Personal Info Vector Database Search")
        personal_start = time.time()
        personal_data = await self._search_personal_info_vectordb_with_cache_analytics(search_queries, user_id)
        personal_time = time.time() - personal_start
        
        # 🚀 SECOND EARLY EXIT CHECK: Check if TIER 1 + TIER 2 combined is sufficient
        combined_data_tier2 = {
            "provided_user_data": user_data,
            "resume_vectordb_data": resume_data,
            "personal_info_vectordb_data": personal_data
        }
        
        # Calculate satisfaction score for combined TIER 1 + TIER 2 data
        tier2_satisfaction_score = self.calculate_combined_satisfaction_score(field_label, resume_data, personal_data)
        
        if tier2_satisfaction_score >= 80.0:
            # SECOND EARLY EXIT: TIER 1 + TIER 2 is sufficient, skip TIER 3!
            self.cache_stats['early_exits'] += 1
            self.cache_stats['tiers_skipped'] += 1  # Skipped TIER 3 only
            
            logger.info(f"🚀 TIER 2 EARLY EXIT - {tier2_satisfaction_score}% combined satisfaction")
            
            # Extract answer from TIER 1 + TIER 2 combined data
            direct_answer = self.extract_answer_from_combined_data(field_label, resume_data, personal_data)
            
            if direct_answer:
                total_time = time.time() - start_time
                
                # Create field mapping for TIER 2 early exit
                field_mappings = [{
                    "field": field_label,
                    "value": direct_answer,
                    "data_source": self.determine_data_source(direct_answer, resume_data, personal_data),
                    "reasoning": f"TIER 2 early exit optimization - {tier2_satisfaction_score}% satisfaction from combined data",
                    "action": "fill",
                    "early_exit": True,
                    "tier_exit": 2
                }]
                
                logger.info(f"⏱️ TIER 2 EXIT: {total_time:.3f}s - Answer: '{direct_answer}'")
                
                # Cache efficiency report
                self._log_cache_efficiency_report()
                
                # Removed redundant log
                
                return {
                    "status": "success",
                    "values": field_mappings,
                    "processing_time": total_time,
                    "early_exit": True,
                    "tier_exit": 2,
                    "satisfaction_score": tier2_satisfaction_score,
                    "tiers_used": 2,
                    "tiers_skipped": 1,
                    "cache_analytics": {
                        "total_requests": self.cache_stats['total_requests'],
                        "cache_hit_rate": self._calculate_cache_hit_rate(),
                        "early_exits": self.cache_stats['early_exits'],
                        "tiers_skipped": self.cache_stats['tiers_skipped'],
                        "cache_breakdown": self.cache_stats
                    }
                }
        
        # Continue to TIER 3 if combined satisfaction < 80%
        logger.info(f"➡️ CONTINUING TO TIER 3 (Combined Satisfaction: {tier2_satisfaction_score}% < 80%)")
        
        # Combine data for TIER 3
        combined_data = {
            "provided_user_data": user_data,
            "resume_vectordb_data": resume_data,
            "personal_info_vectordb_data": personal_data
        }
        
        # Data assessment
        data_assessment = self._assess_data_completeness_optimized(fields, combined_data)
        logger.info(f"📊 Data Assessment: {data_assessment['summary']}")
        
        # TIER 3: LLM Field Mapping
        logger.info(f"🎯 TIER 3: LLM Field Mapping")
        llm_start = time.time()
        field_mappings = await self._generate_llm_field_mappings_optimized(fields, combined_data, data_assessment)
        llm_time = time.time() - llm_start
        
        total_time = time.time() - start_time
        
        logger.info(f"⏱️ FULL 3-TIER: {total_time:.3f}s (Resume: {resume_time:.3f}s, Personal: {personal_time:.3f}s, LLM: {llm_time:.3f}s)")
        
        # Cache efficiency report
        self._log_cache_efficiency_report()
        
        # Removed redundant log
        
        return {
            "status": "success",
            "values": field_mappings,
            "processing_time": total_time,
            "early_exit": False,
            "tier_exit": 3,
            "satisfaction_score": tier2_satisfaction_score,
            "tiers_used": 3,
            "tiers_skipped": 0,
            "cache_analytics": {
                "total_requests": self.cache_stats['total_requests'],
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "early_exits": self.cache_stats['early_exits'],
                "tiers_skipped": self.cache_stats['tiers_skipped'],
                "cache_breakdown": self.cache_stats
            }
        }

    async def _search_resume_vectordb_with_cache_analytics(self, search_queries: List[str], user_id: str) -> Dict[str, Any]:
        """🎯 Resume search with detailed cache analytics"""
        cache_key = f"resume_{user_id}_{hash(tuple(str(q) for q in search_queries))}"
        
        # Check cache first
        if cache_key in self._search_cache:
            self.cache_stats['resume_cache_hits'] += 1
            self.cache_stats['vector_searches_cached'] += 1
            
            self._log_detailed_cache_performance(
                "Resume Vector Search",
                "resume_cache",
                True,
                0.001,
                f"Retrieved cached resume data for user: {user_id}. Avoided {len(search_queries)} vector searches!"
            )
            
            return self._search_cache[cache_key]
        
        # Cache miss - perform actual search
        self.cache_stats['resume_cache_misses'] += 1
        self.cache_stats['vector_searches_performed'] += len(search_queries)
        search_start = time.time()
        
        try:
            if not self.resume_extractor:
                search_time = time.time() - search_start
                self._log_detailed_cache_performance(
                    "Resume Vector Search",
                    "resume_cache",
                    False,
                    search_time,
                    "Resume extractor not available"
                )
                return {"status": "not_found", "message": "Resume extractor not available"}
            
            self.resume_extractor.user_id = user_id
            logger.info(f"🔍 Performing {len(search_queries)} vector searches: {search_queries}")
            
            # Perform vector searches
            all_results = []
            for i, query in enumerate(search_queries[:3]):
                vector_start = time.time()
                try:
                    result = self.resume_extractor.search_resume(query, k=2)
                    vector_time = time.time() - vector_start
                    logger.info(f"   Vector Search {i+1}: '{query}' took {vector_time:.3f}s")
                    
                    if result and "results" in result:
                        all_results.extend(result["results"])
                except Exception as e:
                    logger.warning(f"   Vector Search {i+1} failed: {e}")
            
            self.database_queries += len(search_queries[:3])
            
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
                
                # Cache the result
                if len(self._search_cache) >= self._cache_size:
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]
                
                self._search_cache[cache_key] = resume_data
                search_time = time.time() - search_start
                
                self._log_detailed_cache_performance(
                    "Resume Vector Search",
                    "resume_cache",
                    False,
                    search_time,
                    f"Found {len(unique_content)} unique pieces. Performed {len(search_queries)} vector searches. Result cached for future requests."
                )
                
                return resume_data
            else:
                search_time = time.time() - search_start
                self._log_detailed_cache_performance(
                    "Resume Vector Search",
                    "resume_cache",
                    False,
                    search_time,
                    f"No resume data found after {len(search_queries)} vector searches"
                )
                return {"status": "not_found", "message": "No resume data available"}
                
        except Exception as e:
            search_time = time.time() - search_start
            self._log_detailed_cache_performance(
                "Resume Vector Search",
                "resume_cache",
                False,
                search_time,
                f"Error during vector search: {str(e)}"
            )
            return {"status": "error", "error": str(e)}

    async def _search_personal_info_vectordb_with_cache_analytics(self, search_queries: List[str], user_id: str) -> Dict[str, Any]:
        """🎯 Personal info search with detailed cache analytics"""
        personal_queries = ["contact information", "work authorization", "salary expectations"]
        cache_key = f"personal_{user_id}_{hash(tuple(personal_queries))}"
        
        # Check cache first
        if cache_key in self._search_cache:
            self.cache_stats['personal_cache_hits'] += 1
            self.cache_stats['vector_searches_cached'] += len(personal_queries)
            
            self._log_detailed_cache_performance(
                "Personal Info Vector Search",
                "personal_cache",
                True,
                0.001,
                f"Retrieved cached personal info for user: {user_id}. Avoided {len(personal_queries)} vector searches!"
            )
            
            return self._search_cache[cache_key]
        
        # Cache miss - perform actual search
        self.cache_stats['personal_cache_misses'] += 1
        self.cache_stats['vector_searches_performed'] += len(personal_queries)
        search_start = time.time()
        
        try:
            if not self.personal_info_extractor:
                search_time = time.time() - search_start
                self._log_detailed_cache_performance(
                    "Personal Info Vector Search",
                    "personal_cache",
                    False,
                    search_time,
                    "Personal info extractor not available"
                )
                return {"status": "not_found", "message": "Personal info extractor not available"}
            
            self.personal_info_extractor.user_id = user_id
            logger.info(f"🔍 Performing {len(personal_queries)} vector searches: {personal_queries}")
            
            # Perform vector searches
            all_results = []
            for i, query in enumerate(personal_queries):
                vector_start = time.time()
                try:
                    result = self.personal_info_extractor.search_personal_info(query, k=2)
                    vector_time = time.time() - vector_start
                    logger.info(f"   Vector Search {i+1}: '{query}' took {vector_time:.3f}s")
                    
                    if result and "results" in result:
                        all_results.extend(result["results"])
                except Exception as e:
                    logger.warning(f"   Vector Search {i+1} failed: {e}")
            
            self.database_queries += len(personal_queries)
            
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
                
                # Cache the result
                if len(self._search_cache) >= self._cache_size:
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]
                
                self._search_cache[cache_key] = personal_data
                search_time = time.time() - search_start
                
                self._log_detailed_cache_performance(
                    "Personal Info Vector Search",
                    "personal_cache",
                    False,
                    search_time,
                    f"Found {len(unique_content)} unique pieces. Performed {len(personal_queries)} vector searches. Result cached for future requests."
                )
                
                return personal_data
            else:
                search_time = time.time() - search_start
                self._log_detailed_cache_performance(
                    "Personal Info Vector Search",
                    "personal_cache",
                    False,
                    search_time,
                    f"No personal info found after {len(personal_queries)} vector searches"
                )
                return {"status": "not_found", "message": "No personal info data available"}
                
        except Exception as e:
            search_time = time.time() - search_start
            self._log_detailed_cache_performance(
                "Personal Info Vector Search",
                "personal_cache",
                False,
                search_time,
                f"Error during vector search: {str(e)}"
            )
            return {"status": "error", "error": str(e)}

    def _log_cache_efficiency_report(self):
        """📊 Comprehensive cache efficiency report with early exit metrics"""
        total_hits = sum([self.cache_stats[k] for k in self.cache_stats if 'hits' in k])
        total_misses = sum([self.cache_stats[k] for k in self.cache_stats if 'misses' in k])
        cache_hit_rate = (total_hits / (total_hits + total_misses)) * 100 if (total_hits + total_misses) > 0 else 0
        
        vector_searches_saved = self.cache_stats['vector_searches_cached']
        vector_searches_performed = self.cache_stats['vector_searches_performed']
        total_vector_operations = vector_searches_saved + vector_searches_performed
        vector_cache_efficiency = (vector_searches_saved / total_vector_operations) * 100 if total_vector_operations > 0 else 0
        
        # Early exit efficiency
        early_exit_rate = (self.cache_stats['early_exits'] / self.cache_stats['total_requests'] * 100) if self.cache_stats['total_requests'] > 0 else 0.0
        
        logger.info(f"")
        logger.info(f"📈 COMPREHENSIVE PERFORMANCE REPORT:")
        logger.info(f"   🎯 Overall Cache Hit Rate: {cache_hit_rate:.1f}%")
        logger.info(f"   📊 Cache Breakdown:")
        logger.info(f"      • Resume Cache: {self.cache_stats['resume_cache_hits']} hits, {self.cache_stats['resume_cache_misses']} misses")
        logger.info(f"      • Personal Cache: {self.cache_stats['personal_cache_hits']} hits, {self.cache_stats['personal_cache_misses']} misses")
        logger.info(f"   ⚡ Vector Search Optimization:")
        logger.info(f"      • Vector Searches Avoided (Cached): {vector_searches_saved}")
        logger.info(f"      • Vector Searches Performed: {vector_searches_performed}")
        logger.info(f"      • Vector Cache Efficiency: {vector_cache_efficiency:.1f}%")
        logger.info(f"   🚀 EARLY EXIT OPTIMIZATION:")
        logger.info(f"      • Early Exits: {self.cache_stats['early_exits']} / {self.cache_stats['total_requests']} requests")
        logger.info(f"      • Early Exit Rate: {early_exit_rate:.1f}%")
        logger.info(f"      • Tiers Skipped: {self.cache_stats['tiers_skipped']} (TIER 2 & 3)")
        logger.info(f"      • Time Saved: ~{self.cache_stats['tiers_skipped'] * 1.5:.1f}s (estimated)")
        logger.info(f"   💡 Performance Impact:")
        if early_exit_rate >= 50:
            logger.info(f"      • 🚀 BLAZING FAST: High early exit rate = Ultra-fast responses!")
        elif cache_hit_rate > 50:
            logger.info(f"      • EXCELLENT: High cache hit rate = Fast responses!")
        elif cache_hit_rate > 25:
            logger.info(f"      • GOOD: Moderate cache hit rate = Improving performance")
        else:
            logger.info(f"      • BUILDING: Low cache hit rate = Performance will improve with usage")

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        total_hits = sum([self.cache_stats[k] for k in self.cache_stats if 'hits' in k])
        total_misses = sum([self.cache_stats[k] for k in self.cache_stats if 'misses' in k])
        return (total_hits / (total_hits + total_misses)) * 100 if (total_hits + total_misses) > 0 else 0

    def _calculate_performance_improvement(self) -> str:
        """Calculate performance improvement trend"""
        if self.cache_stats['total_requests'] == 1:
            return "First request - establishing baseline"
        elif self.cache_stats['total_requests'] <= 3:
            return "Building cache - performance improving"
        else:
            hit_rate = self._calculate_cache_hit_rate()
            if hit_rate > 50:
                return "High performance - cache optimized"
            elif hit_rate > 25:
                return "Good performance - cache warming up"
            else:
                return "Performance building - more requests needed"

    @lru_cache(maxsize=50)
    def _get_search_queries(self, field_purposes_tuple: Tuple[str, ...]) -> List[str]:
        """🎯 DIRECT QUESTION-BASED SEARCH QUERY GENERATOR"""
        field_purposes = list(field_purposes_tuple)
        
        # 🚀 NEW APPROACH: Direct question-based search (no generic grouping)
        queries = []
        
        for field_purpose in field_purposes:
            field_lower = field_purpose.lower().strip()
            
            # 🎯 DIRECT MAPPING: Question → Specific Search Terms
            
            # Name questions → Name-specific search
            if any(keyword in field_lower for keyword in ['name', 'full name', 'first name', 'last name']):
                queries.extend(['name', 'full name'])
            
            # Email questions → Email-specific search ONLY
            elif any(keyword in field_lower for keyword in ['email', 'e-mail', 'mail address']):
                queries.extend(['email', 'e-mail'])  # Only email-related terms!
            
            # Phone questions → Phone-specific search ONLY
            elif any(keyword in field_lower for keyword in ['phone', 'telephone', 'mobile', 'cell']):
                queries.extend(['phone', 'telephone', 'mobile'])  # Only phone-related terms!
            
            # Address questions → Address-specific search ONLY
            elif any(keyword in field_lower for keyword in ['address', 'location', 'city', 'state', 'zip']):
                queries.extend(['address', 'location'])  # Only address-related terms!
            
            # Work authorization questions → Authorization-specific search
            elif any(keyword in field_lower for keyword in ['authorization', 'visa', 'work status', 'eligible']):
                queries.extend(['work authorization', 'visa status', 'eligible to work'])
            
            # Salary questions → Salary-specific search
            elif any(keyword in field_lower for keyword in ['salary', 'compensation', 'pay', 'wage']):
                queries.extend(['salary', 'compensation', 'expected salary'])
            
            # Experience questions → Experience-specific search
            elif any(keyword in field_lower for keyword in ['experience', 'work', 'job', 'position', 'title', 'role']):
                queries.extend(['work experience', 'professional experience', 'job title'])
            
            # Education questions → Education-specific search
            elif any(keyword in field_lower for keyword in ['education', 'degree', 'school', 'university', 'college']):
                queries.extend(['education', 'degree', 'university'])
            
            # Skills questions → Skills-specific search
            elif any(keyword in field_lower for keyword in ['skill', 'technology', 'programming', 'technical']):
                queries.extend(['skills', 'technical skills', 'programming'])
            
            # Generic/complex questions → Use the question itself as search term
            else:
                # For complex questions, use key words from the question
                question_words = [word for word in field_lower.split() if len(word) > 3]
                if question_words:
                    queries.extend(question_words[:2])  # Use first 2 meaningful words
                else:
                    queries.append(field_lower)  # Use the whole question
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in queries:
            if query not in seen:
                unique_queries.append(query)
                seen.add(query)
        
        # 🎯 OPTIMIZATION: Limit to 3 most relevant queries (not 5)
        final_queries = unique_queries[:3]
        
        logger.info(f"🎯 DIRECT SEARCH OPTIMIZATION:")
        logger.info(f"   • Question: {field_purposes}")
        logger.info(f"   • Generated Queries: {final_queries}")
        logger.info(f"   • Efficiency: {len(final_queries)} targeted searches (vs old generic 4-5)")
        
        return final_queries

    async def _generate_llm_field_mappings_optimized(self, fields: List[Dict], combined_data: Dict, data_assessment: Dict) -> List[Dict]:
        """Generate field mappings using optimized LLM call"""
        try:
            # Create optimized prompt
            prompt = self._create_optimized_prompt(fields, combined_data, data_assessment)
            
            # Single LLM call for all fields
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional career coach and job application expert. Your goal is to help candidates write compelling, enthusiastic, and professional answers that will help them get hired. Write answers that showcase personality, qualifications, and genuine interest in the role. Never give short, generic answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Slightly higher for more creative/human-like responses
                max_tokens=1500   # More tokens for detailed answers
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Extract field mappings
            field_mappings = self._parse_llm_response(response_text, fields)
            
            # Log data usage statistics
            resume_used = sum(1 for mapping in field_mappings if mapping.get('data_source') == 'resume_vectordb')
            personal_used = sum(1 for mapping in field_mappings if mapping.get('data_source') == 'personal_info_vectordb')
            generated = sum(1 for mapping in field_mappings if mapping.get('data_source') == 'generated')
            
            logger.info(f"🧠 Optimized LLM field mapping completed:")
            logger.info(f"   📊 Data Usage: Resume={resume_used}, Personal={personal_used}, Generated={generated}")
            
            return field_mappings
            
        except Exception as e:
            logger.error(f"❌ LLM field mapping failed: {e}")
            return [{"field": field.get("name", "unknown"), "value": "", "data_source": "error"} for field in fields]

    def _create_optimized_prompt(self, fields: List[Dict], combined_data: Dict, data_assessment: Dict) -> str:
        """Create job-focused prompt for compelling answers"""
        prompt_parts = []
        
        # 🎯 JOB APPLICATION CONTEXT
        prompt_parts.append("You are helping someone fill out a JOB APPLICATION form. Your goal is to help them GET HIRED by writing compelling, professional, and enthusiastic answers that showcase their qualifications and personality.")
        
        prompt_parts.append("\n🎯 WRITING GUIDELINES:")
        prompt_parts.append("- Write answers that sound HUMAN and ENTHUSIASTIC")
        prompt_parts.append("- Show PASSION and GENUINE INTEREST in the role/company")
        prompt_parts.append("- Use SPECIFIC EXAMPLES when possible")
        prompt_parts.append("- Keep answers PROFESSIONAL but PERSONABLE")
        prompt_parts.append("- NEVER give one-word answers like 'Yes' or 'No'")
        prompt_parts.append("- For yes/no questions, explain WHY and give context")
        prompt_parts.append("- Show CONFIDENCE without being arrogant")
        
        # Add available data with context
        resume_data = combined_data.get("resume_vectordb_data", {})
        if resume_data.get("status") == "found":
            prompt_parts.append(f"\n📄 CANDIDATE'S RESUME DATA:")
            prompt_parts.append(f"{resume_data.get('content', '')}")
        
        personal_data = combined_data.get("personal_info_vectordb_data", {})
        if personal_data.get("status") == "found":
            prompt_parts.append(f"\n👤 PERSONAL INFORMATION:")
            prompt_parts.append(f"{personal_data.get('content', '')}")
        
        user_data = combined_data.get("provided_user_data", {})
        if user_data:
            prompt_parts.append(f"\n💼 ADDITIONAL INFO: {json.dumps(user_data)}")
        
        # Add fields with job context
        prompt_parts.append("\n📝 FORM FIELDS TO FILL:")
        for i, field in enumerate(fields):
            field_name = field.get("name", field.get("selector", f"field_{i}"))
            
            # Add context hints for common question types
            context_hint = ""
            field_lower = field_name.lower()
            
            if any(keyword in field_lower for keyword in ['startup', 'fast paced', 'hard working']):
                context_hint = " (Show enthusiasm for startup culture and fast-paced environment)"
            elif any(keyword in field_lower for keyword in ['why', 'motivation', 'interest']):
                context_hint = " (Show genuine interest and specific reasons)"
            elif any(keyword in field_lower for keyword in ['experience', 'background']):
                context_hint = " (Highlight relevant experience with specific examples)"
            elif any(keyword in field_lower for keyword in ['strength', 'skill', 'good at']):
                context_hint = " (Be confident and provide concrete examples)"
            elif any(keyword in field_lower for keyword in ['challenge', 'difficult']):
                context_hint = " (Show problem-solving ability and growth mindset)"
            
            prompt_parts.append(f"{i+1}. {field_name}{context_hint}")
        
        # 🎯 EXAMPLE ANSWERS for reference
        prompt_parts.append("\n💡 EXAMPLE OF GOOD vs BAD ANSWERS:")
        prompt_parts.append("❌ BAD: 'Yes' (too short, no personality)")
        prompt_parts.append("✅ GOOD: 'Absolutely! I thrive in fast-paced environments where I can make a direct impact. In my previous role at [company], I successfully managed multiple projects under tight deadlines while maintaining high quality standards. I'm excited about the opportunity to contribute to a growing startup where my skills in [relevant skill] can help drive the company's success.'")
        
        prompt_parts.append("\n📋 PROVIDE ANSWERS IN FORMAT:")
        prompt_parts.append("1. [Your compelling answer here]")
        prompt_parts.append("2. [Your compelling answer here]")
        prompt_parts.append("etc.")
        
        return "\n".join(prompt_parts)

    def _parse_llm_response(self, response_text: str, fields: List[Dict]) -> List[Dict]:
        """Parse LLM response into field mappings"""
        field_mappings = []
        lines = response_text.strip().split('\n')
        
        for i, field in enumerate(fields):
            field_name = field.get("name", field.get("selector", f"field_{i}"))
            value = ""
            data_source = "generated"
            
            # Try to extract value from response
            for line in lines:
                if line.strip().startswith(f"{i+1}."):
                    value = line.split(".", 1)[1].strip()
                    break
            
            # Determine data source
            if "Eric Abram" in value or "ericabram33@gmail.com" in value or "312-805-9851" in value:
                data_source = "resume_vectordb"
            elif any(keyword in value.lower() for keyword in ["contact", "authorization", "salary"]):
                data_source = "personal_info_vectordb"
            
            field_mappings.append({
                "field": field_name,
                "value": value,
                "data_source": data_source,
                "reasoning": f"Extracted from {data_source}" if data_source != "generated" else "Generated based on context"
            })
        
        return field_mappings

    def _assess_data_completeness_optimized(self, fields: List[Dict], combined_data: Dict) -> Dict[str, Any]:
        """Quick assessment of data completeness"""
        resume_available = combined_data.get("resume_vectordb_data", {}).get("status") == "found"
        personal_available = combined_data.get("personal_info_vectordb_data", {}).get("status") == "found"
        user_data_available = bool(combined_data.get("provided_user_data"))
        
        sources_count = sum([resume_available, personal_available, user_data_available])
        
        if sources_count >= 2:
            summary = "EXCELLENT - Multiple data sources available"
        elif sources_count == 1:
            summary = "GOOD - One data source available"
        else:
            summary = "LIMITED - Minimal data available, will generate"
        
        return {
            "summary": summary,
            "sources_count": sources_count,
            "resume_available": resume_available,
            "personal_available": personal_available,
            "user_data_available": user_data_available
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "cache_stats": self.cache_stats,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "performance_trend": self._calculate_performance_improvement(),
            "database_queries": self.database_queries,
            "cache_hits": self.cache_hits
        }

    def clear_cache(self):
        """Clear all caches"""
        self._search_cache.clear()
        self._resume_cache.clear()
        self._personal_cache.clear()
        logger.info("🧹 All caches cleared")

    def _get_resume_extractor(self) -> ResumeExtractorOptimized:
        """Resume extractor instance (cache disabled for debugging)"""
        if self.resume_extractor:
            return self.resume_extractor
        return ResumeExtractorOptimized(
            openai_api_key=self.openai_api_key,
            use_hf_fallback=True
        )
    
    @lru_cache(maxsize=32)
    def _get_personal_info_extractor(self) -> PersonalInfoExtractorOptimized:
        """Cached personal info extractor instance"""
        return PersonalInfoExtractorOptimized()
    
    @lru_cache(maxsize=32)
    def _get_document_service(self) -> DocumentService:
        """Cached document service instance"""
        return DocumentService()

    def _generate_resume_search_queries(self, field_labels: List[str]) -> List[str]:
        # Implementation of _generate_resume_search_queries method
        pass

    def _generate_personal_info_search_queries(self, field_labels: List[str]) -> List[str]:
        # Implementation of _generate_personal_info_search_queries method
        pass 