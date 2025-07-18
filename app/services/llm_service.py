"""
🧠 SMART LLM SERVICE WITH LANGGRAPH TOOL CALLING
Intelligent form filling service that uses LLM with tools to answer ANY question
No hardcoding - pure AI intelligence with resume and personal info tools

OPTIMIZATIONS:
- Async/await optimization for better performance
- Intelligent caching system
- Enhanced error handling and retry logic
- Better prompt engineering
- Performance monitoring and metrics
- Connection pooling and resource management
"""

import os
import time
import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Annotated, Union
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from dataclasses import dataclass, field
from enum import Enum
import logging

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator
from typing_extensions import TypedDict

from app.utils.logger import logger
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import RedisVectorStore
from app.services.document_service import DocumentService
# Integrated usage analyzer import removed
from app.services.rag_service import RAGService


class FieldType(Enum):
    """Enum for different field types to optimize processing"""
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    ADDRESS = "address"
    EXPERIENCE = "experience"
    SKILL = "skill"
    EDUCATION = "education"
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    GENERIC = "generic"


@dataclass
class CacheEntry:
    """Cache entry for LLM responses"""
    answer: str
    confidence: float
    data_source: str
    timestamp: datetime
    field_type: FieldType
    ttl_seconds: int = 3600  # 1 hour default TTL


@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_processing_time: float = 0.0
    total_llm_time: float = 0.0
    total_vector_search_time: float = 0.0
    resume_tool_calls: int = 0
    personal_info_tool_calls: int = 0
    retry_attempts: int = 0
    
    @property
    def success_rate(self) -> float:
        return (self.successful_requests / max(self.total_requests, 1)) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        total_cache_attempts = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(total_cache_attempts, 1)) * 100
    
    @property
    def avg_processing_time(self) -> float:
        return self.total_processing_time / max(self.total_requests, 1)


class AgentState(TypedDict):
    """Enhanced state for the LangGraph agent"""
    question: str
    user_id: str
    field_type: FieldType
    messages: List[Any]
    resume_data: Optional[str]
    personal_data: Optional[str]
    final_answer: Optional[str]
    processing_time: float
    confidence_score: float
    retry_count: int
    context: Dict[str, Any]


class ResumeSearchInput(BaseModel):
    """Input for resume search tool with validation"""
    query: str = Field(description="Search query for resume data", min_length=1, max_length=500)
    user_id: str = Field(description="User ID to search for", min_length=1)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class PersonalInfoSearchInput(BaseModel):
    """Input for personal info search tool with validation"""
    query: str = Field(description="Search query for personal information", min_length=1, max_length=500)
    user_id: str = Field(description="User ID to search for", min_length=1)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Async retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


class SmartLLMService:
    """
    🧠 OPTIMIZED Smart LLM Service with LangGraph Tool Calling
    
    IMPROVEMENTS:
    - Intelligent caching system with TTL
    - Enhanced error handling with retry logic
    - Better prompt engineering for accuracy
    - Performance monitoring and metrics
    - Async optimization for better throughput
    - Field type detection for optimized processing
    - Connection pooling and resource management
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, cache_size: int = 1000):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Enhanced performance tracking
        self.metrics = PerformanceMetrics()
        
        # Intelligent caching system
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_size = cache_size
        
        # Initialize services with connection pooling
        self._init_services()
        
        # Initialize LLM with optimized settings
        self._init_llm()
        
        # Setup tools and graph
        self._setup_tools()
        self._setup_graph()
        
        # Field type mapping for optimization
        self._setup_field_type_mapping()
        
        logger.info("🧠 OPTIMIZED SMART LLM SERVICE initialized")
        logger.info(f"   ✅ Cache size: {cache_size}")
        logger.info("   ✅ Enhanced error handling enabled")
        logger.info("   ✅ Performance monitoring active")
        logger.info("   ✅ Async optimization enabled")

    def _init_services(self):
        """Initialize services with better error handling"""
        try:
            self.embedding_service = EmbeddingService()
            self.vector_store = RedisVectorStore()
            
            # Initialize document service with fallback URL
            postgres_url = os.getenv("DATABASE_URL", "postgresql://postgres:OZNHVfQlRwGhcUBFmkVluOzTonqTpIKa@interchange.proxy.rlwy.net:30153/railway")
            self.document_service = DocumentService(postgres_url)
            
            logger.info("✅ All services initialized successfully")
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise

    def _init_llm(self):
        """Initialize LLM with optimized settings"""
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Faster and cheaper than gpt-4
                temperature=0.1,
                max_tokens=500,  # Limit response length
                timeout=30,  # Add timeout
                max_retries=2,  # Built-in retry
                openai_api_key=self.openai_api_key
            )
            logger.info("✅ LLM initialized with optimized settings")
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            raise

    def _setup_field_type_mapping(self):
        """Setup field type detection for optimization"""
        self.field_patterns = {
            FieldType.EMAIL: ["email", "e-mail", "mail", "contact"],
            FieldType.PHONE: ["phone", "telephone", "mobile", "cell", "number"],
            FieldType.NAME: ["name", "first", "last", "full name"],
            FieldType.ADDRESS: ["address", "location", "city", "state", "zip"],
            FieldType.EXPERIENCE: ["experience", "work", "job", "position", "role"],
            FieldType.SKILL: ["skill", "ability", "proficiency", "expertise"],
            FieldType.EDUCATION: ["education", "degree", "school", "university", "college"],
            FieldType.PERSONAL: ["personal", "hobby", "interest", "preference"],
            FieldType.PROFESSIONAL: ["professional", "career", "industry", "company"]
        }

    def _detect_field_type(self, field_label: str) -> FieldType:
        """Detect field type for optimization"""
        field_lower = field_label.lower()
        
        for field_type, patterns in self.field_patterns.items():
            if any(pattern in field_lower for pattern in patterns):
                return field_type
        
        return FieldType.GENERIC

    def _generate_cache_key(self, field_label: str, user_id: str, context: Optional[Dict] = None) -> str:
        """Generate cache key for responses"""
        content = f"{field_label}:{user_id}"
        if context:
            content += f":{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[CacheEntry]:
        """Get cached response if valid"""
        if cache_key not in self.cache:
            self.metrics.cache_misses += 1
            return None
        
        entry = self.cache[cache_key]
        
        # Check TTL
        if datetime.now() - entry.timestamp > timedelta(seconds=entry.ttl_seconds):
            del self.cache[cache_key]
            self.metrics.cache_misses += 1
            return None
        
        self.metrics.cache_hits += 1
        return entry

    def _cache_response(self, cache_key: str, answer: str, confidence: float, 
                       data_source: str, field_type: FieldType, ttl_seconds: int = 3600):
        """Cache response with TTL"""
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        
        self.cache[cache_key] = CacheEntry(
            answer=answer,
            confidence=confidence,
            data_source=data_source,
            timestamp=datetime.now(),
            field_type=field_type,
            ttl_seconds=ttl_seconds
        )

    def _setup_tools(self):
        """Setup enhanced LangChain tools"""
        
        @tool("search_resume_data", args_schema=ResumeSearchInput)
        async def search_resume_data(query: str, user_id: str) -> str:
            """
            Search through user's resume data to find relevant information.
            Use this tool when you need professional information like job titles, 
            companies, work experience, skills, education, etc.
            """
            try:
                start_time = time.time()
                results = await self._search_resume_vectors(query, user_id)
                search_time = time.time() - start_time
                
                self.metrics.resume_tool_calls += 1
                self.metrics.total_vector_search_time += search_time
                
                if results:
                    # Combine and limit text to avoid token limits
                    combined_text = "\n".join([
                        result.get('content', result.get('text', ''))[:200]  # Limit each result
                        for result in results[:3]  # Top 3 results only
                    ])
                    logger.info(f"📄 Resume tool found {len(results)} results in {search_time:.3f}s")
                    return f"Resume data found: {combined_text}"
                else:
                    logger.info(f"📄 Resume tool found no results for: {query}")
                    return "No relevant resume data found for this query."
                    
            except Exception as e:
                logger.error(f"Resume search error: {e}")
                return "Error searching resume data."

        @tool("search_personal_info", args_schema=PersonalInfoSearchInput)
        async def search_personal_info(query: str, user_id: str) -> str:
            """
            Search through user's personal information to find relevant details.
            Use this tool when you need personal information like contact details,
            address, phone number, email, personal preferences, etc.
            """
            try:
                start_time = time.time()
                results = await self._search_personal_vectors(query, user_id)
                search_time = time.time() - start_time
                
                self.metrics.personal_info_tool_calls += 1
                self.metrics.total_vector_search_time += search_time
                
                if results:
                    # Combine and limit text to avoid token limits
                    combined_text = "\n".join([
                        result.get('content', result.get('text', ''))[:200]  # Limit each result
                        for result in results[:3]  # Top 3 results only
                    ])
                    logger.info(f"👤 Personal info tool found {len(results)} results in {search_time:.3f}s")
                    return f"Personal information found: {combined_text}"
                else:
                    logger.info(f"👤 Personal info tool found no results for: {query}")
                    return "No relevant personal information found for this query."
                    
            except Exception as e:
                logger.error(f"Personal info search error: {e}")
                return "Error searching personal information."

        # Store tools
        self.tools = [search_resume_data, search_personal_info]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _setup_graph(self):
        """Setup optimized LangGraph workflow"""
        
        def agent_node(state: AgentState) -> AgentState:
            """Enhanced agent node with better error handling"""
            messages = state["messages"]
            
            try:
                # Get LLM response with tool calling capability
                response = self.llm_with_tools.invoke(messages)
                messages.append(response)
                
                return {**state, "messages": messages}
            except Exception as e:
                logger.error(f"Agent node error: {e}")
                # Return error state
                return {**state, "messages": messages, "error": str(e)}
        
        # Create tool node using LangGraph's ToolNode
        tool_node = ToolNode(self.tools)
        
        def should_continue(state: AgentState) -> str:
            """Enhanced decision logic"""
            messages = state["messages"]
            
            # Check for errors
            if "error" in state:
                return "end"
            
            if not messages:
                return "end"
            
            last_message = messages[-1]
            
            # If the last message has tool calls, continue to tool execution
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            else:
                return "end"
        
        def extract_final_answer(state: AgentState) -> AgentState:
            """Enhanced answer extraction with better confidence scoring"""
            messages = state["messages"]
            
            # Get the last AI message as the final answer
            final_answer = "Unable to generate answer"
            confidence = 30.0  # Default low confidence
            
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    # Skip messages with tool calls, get the actual response
                    if not (hasattr(message, 'tool_calls') and message.tool_calls):
                        final_answer = message.content
                        break
            
            # Enhanced confidence calculation
            has_resume_data = any("Resume data found:" in str(msg.content) for msg in messages if hasattr(msg, 'content'))
            has_personal_data = any("Personal information found:" in str(msg.content) for msg in messages if hasattr(msg, 'content'))
            
            if has_resume_data and has_personal_data:
                confidence = 90.0
            elif has_resume_data or has_personal_data:
                confidence = 75.0
            elif any("found:" in str(msg.content) for msg in messages if hasattr(msg, 'content')):
                confidence = 60.0
            else:
                confidence = 45.0
            
            return {
                **state, 
                "final_answer": final_answer,
                "confidence_score": confidence
            }
        
        # Build the enhanced graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("extract_answer", extract_final_answer)
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": "extract_answer"
            }
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("extract_answer", END)
        
        # Compile the graph
        self.app = workflow.compile()

    @async_retry(max_retries=3, delay=1.0, backoff=2.0)
    async def generate_field_answer(
        self,
        field_label: str,
        user_id: str,
        field_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate intelligent field answer using the chunk confidence extraction pipeline.
        """
        start_time = time.time()
        self.metrics.total_requests += 1
        # Check cache first
        cache_key = self._generate_cache_key(field_label, user_id, field_context)
        cached_entry = self._get_cached_response(cache_key)
        if cached_entry:
            processing_time = time.time() - start_time
            logger.info(f"🎯 Cache hit for field: {field_label}")
            return {
                "status": "success",
                "answer": cached_entry.answer,
                "confidence": cached_entry.confidence,
                "data_source": f"{cached_entry.data_source} (cached)",
                "processing_time": processing_time,
                "reasoning": "Retrieved from cache",
                "performance_metrics": {
                    "processing_time_seconds": processing_time,
                    "cache_hit": True
                }
            }
        logger.info(f"🧠 RAG Pipeline - Processing: '{field_label}' for user {user_id}")
        try:
            # Use the new RAG service with proper vector similarity search
            rag_service = RAGService(openai_api_key=self.openai_api_key)
            result = await asyncio.to_thread(rag_service.generate_field_answer, field_label, user_id)
            answer = result["answer"]
            confidence = result["confidence"]
            source = result["data_source"]
            reasoning = result["reasoning"]
            processing_time = time.time() - start_time

            # If no answer found, use LLM to generate a professional answer
            if (not answer or answer.lower() in ["not found", "not found."] or confidence == 0):
                llm_prompt = (
                    f"You are a professional assistant. Generate a helpful, context-aware answer to the following question "
                    f"for a job application form. If the question is open-ended, provide a positive, relevant response. "
                    f"Question: {field_label}\n\n"
                    f"IMPORTANT:\n- Do not say 'I am an AI language model'.\n- Be concise and professional.\n\n"
                    f"Answer:"
                )
                llm_response = await asyncio.to_thread(self.llm.invoke, llm_prompt)
                answer = llm_response.content.strip()
                confidence = 70
                source = "llm_fallback"
                reasoning = f"LLM fallback: {reasoning}"

            # Cache the response
            self._cache_response(cache_key, answer, confidence, source, FieldType.GENERIC, 3600)
            self.metrics.successful_requests += 1
            self.metrics.total_processing_time += processing_time
            logger.info(f"✅ RAG Pipeline answer: '{answer}' (confidence: {confidence}%)")
            logger.info(f"   Source: {source}")
            logger.info(f"   Reasoning: {reasoning}")
            return {
                "status": "success",
                "answer": answer,
                "confidence": confidence,
                "data_source": source,
                "processing_time": processing_time,
                "reasoning": f"RAG pipeline: {reasoning}",
                "performance_metrics": {
                    "processing_time_seconds": processing_time,
                    "cache_hit": False
                }
            }
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.failed_requests += 1
            logger.error(f"❌ RAG Pipeline error: {str(e)}")
            return {
                "status": "error",
                "answer": "Unable to generate answer due to system error",
                "confidence": 0,
                "data_source": "error",
                "processing_time": processing_time,
                "reasoning": f"Error occurred: {str(e)}",
                "performance_metrics": {
                    "processing_time_seconds": processing_time,
                    "cache_hit": False
                }
            }

    async def _search_resume_vectors(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Optimized resume vector search with better error handling"""
        try:
            # Use the embedding service's search method for resume data
            results = self.embedding_service.search_similar_by_document_type(
                query=query,
                user_id=user_id,
                document_type="resume",
                top_k=3,  # Reduced for performance
                min_score=0.1  # Very low threshold to find any results
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Resume vector search error: {e}")
            return []

    async def _search_personal_vectors(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Optimized personal info vector search with better error handling"""
        try:
            # Use the embedding service's search method for personal info data
            results = self.embedding_service.search_similar_by_document_type(
                query=query,
                user_id=user_id,
                document_type="personal_info",
                top_k=3,  # Reduced for performance
                min_score=0.1  # Very low threshold to find any results
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Personal vector search error: {e}")
            return []

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": self.metrics.success_rate,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "cache_size": len(self.cache),
            "total_processing_time": self.metrics.total_processing_time,
            "total_llm_time": self.metrics.total_llm_time,
            "total_vector_search_time": self.metrics.total_vector_search_time,
            "avg_processing_time": self.metrics.avg_processing_time,
            "resume_tool_calls": self.metrics.resume_tool_calls,
            "personal_info_tool_calls": self.metrics.personal_info_tool_calls,
            "retry_attempts": self.metrics.retry_attempts
        }

    def _clean_answer(self, raw_answer: str, field_label: str, field_type: FieldType, fallback_text: str = "") -> str:
        """Enhanced answer cleaning with field-type-specific logic and fallback extraction"""
        import re
        
        if not raw_answer:
            cleaned = ""
        else:
            cleaned = raw_answer.strip()
        
        # Field-type-specific cleaning
        if field_type == FieldType.EMAIL:
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', cleaned)
            if email_match:
                return email_match.group()
        
        elif field_type == FieldType.PHONE:
            phone_match = re.search(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', cleaned)
            if phone_match:
                return phone_match.group()
        
        elif field_type == FieldType.NAME:
            # Enhanced name extraction: First Last, First M. Last, First Middle Last, etc.
            name_patterns = [
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # First Middle Last
                r'\b([A-Z][a-z]+\s+[A-Z]\.[ ]?[A-Z][a-z]+)\b',      # First M. Last
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',                  # First Last
                r'Name[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',             # Name: First Last
                r'Full Name[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',        # Full Name: First Last
                r'([A-Z][a-z]+)',                                      # Single capitalized word (fallback)
            ]
            for pattern in name_patterns:
                match = re.search(pattern, cleaned)
                if match:
                    candidate = match.group(1).strip()
                    # Avoid returning the field label itself
                    if candidate.lower() != field_label.lower() and len(candidate.split()) >= 2:
                        return candidate
        
        # Generic cleaning
        prefixes_to_remove = [
            "The answer is ",
            "Answer: ",
            f"The {field_label.lower()} is ",
            "Based on the data, ",
            "According to the information, "
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Remove trailing punctuation
        if cleaned.endswith('.') and not cleaned.endswith('...'):
            cleaned = cleaned[:-1].strip()
        
        # Remove quotes
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        
        # Fallback: If answer is empty or just the field label, try extracting from fallback_text
        if (not cleaned or cleaned.lower() == field_label.lower()) and field_type == FieldType.NAME and fallback_text:
            for pattern in name_patterns:
                match = re.search(pattern, fallback_text)
                if match:
                    candidate = match.group(1).strip()
                    if candidate.lower() != field_label.lower() and len(candidate.split()) >= 2:
                        logger.info(f"[Fallback] Extracted name from vector data: {candidate}")
                        return candidate
        
        return cleaned if cleaned else "Not available"

    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        logger.info("🧹 Response cache cleared")

    def clear_stats(self):
        """Clear performance statistics"""
        self.metrics = PerformanceMetrics()
        logger.info("📊 Performance statistics cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses
        }


# Alias for backward compatibility
UltimateSmartLLMService = SmartLLMService