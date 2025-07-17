#!/usr/bin/env python3
"""
🧠 SMART LLM SERVICE WITH LANGGRAPH - COMPREHENSIVE TEST
Tests the new intelligent LLM service that uses tool calling for ANY field type
"""

import asyncio
import time
import json
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger first
import sys
sys.path.append('.')
from app.utils.logger import configure_logger
configure_logger()

from loguru import logger
from app.services.llm_service import SmartLLMService

async def test_smart_llm_langgraph():
    """Comprehensive test of the Smart LLM Service with LangGraph tool calling"""
    
    logger.info("🧠 TESTING SMART LLM SERVICE WITH LANGGRAPH TOOL CALLING")
    logger.info("="*80)
    
    # Initialize the service
    try:
        smart_llm = SmartLLMService()
        logger.info("✅ Smart LLM Service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Smart LLM Service: {e}")
        return
    
    # Test user ID
    test_user_id = "test_user_123"
    
    # Test various field types to demonstrate intelligence
    test_fields = [
        # Professional fields (should use resume tool)
        "What is your current job title?",
        "Which company do you work for?",
        "What are your main skills?",
        
        # Personal fields (should use personal info tool)
        "What is your email address?",
        "What is your phone number?",
        "What is your home address?",
        
        # Complex fields (should use both tools or reasoning)
        "What is your preferred work location?",
        "How many years of experience do you have?",
        "What is your salary expectation?",
        
        # Creative fields (should demonstrate AI reasoning)
        "Why are you interested in this position?",
        "What makes you a good candidate?",
        "Describe your ideal work environment"
    ]
    
    results = []
    total_start_time = time.time()
    
    logger.info(f"🎯 Testing {len(test_fields)} different field types...")
    logger.info("")
    
    for i, field_label in enumerate(test_fields, 1):
        logger.info(f"📝 TEST {i}/{len(test_fields)}: '{field_label}'")
        logger.info("-" * 60)
        
        try:
            # Test the field
            result = await smart_llm.generate_field_answer(
                field_label=field_label,
                user_id=test_user_id
            )
            
            # Log results
            status = result.get("status", "unknown")
            answer = result.get("answer", "No answer")
            confidence = result.get("confidence", 0)
            data_source = result.get("data_source", "unknown")
            processing_time = result.get("processing_time", 0)
            tools_used = result.get("field_analysis", {}).get("tools_used", {})
            
            logger.info(f"   Status: {status}")
            logger.info(f"   Answer: {answer[:150]}{'...' if len(answer) > 150 else ''}")
            logger.info(f"   Confidence: {confidence:.1f}%")
            logger.info(f"   Data Source: {data_source}")
            logger.info(f"   Processing Time: {processing_time:.3f}s")
            logger.info(f"   Resume Tool Used: {tools_used.get('resume_search', False)}")
            logger.info(f"   Personal Tool Used: {tools_used.get('personal_search', False)}")
            
            # Store result
            results.append({
                "field": field_label,
                "result": result,
                "success": status == "success"
            })
            
            if status == "success":
                logger.info("   ✅ SUCCESS")
            else:
                logger.error("   ❌ FAILED")
                
        except Exception as e:
            logger.error(f"   ❌ ERROR: {str(e)}")
            results.append({
                "field": field_label,
                "result": {"error": str(e)},
                "success": False
            })
        
        logger.info("")
    
    total_time = time.time() - total_start_time
    
    # Calculate statistics
    successful_tests = sum(1 for r in results if r["success"])
    success_rate = (successful_tests / len(results)) * 100
    
    # Get performance stats
    perf_stats = smart_llm.get_performance_stats()
    
    # Print comprehensive summary
    logger.info("🎯 SMART LLM SERVICE TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"📊 Overall Results:")
    logger.info(f"   • Total Tests: {len(results)}")
    logger.info(f"   • Successful: {successful_tests}")
    logger.info(f"   • Failed: {len(results) - successful_tests}")
    logger.info(f"   • Success Rate: {success_rate:.1f}%")
    logger.info(f"   • Total Testing Time: {total_time:.3f}s")
    logger.info("")
    
    logger.info(f"🔧 Tool Usage Statistics:")
    logger.info(f"   • Resume Tool Calls: {perf_stats.get('resume_tool_calls', 0)}")
    logger.info(f"   • Personal Info Tool Calls: {perf_stats.get('personal_info_tool_calls', 0)}")
    logger.info(f"   • Average Processing Time: {perf_stats.get('average_processing_time', 0):.3f}s")
    logger.info(f"   • Service Success Rate: {perf_stats.get('success_rate', 0):.1f}%")
    logger.info("")
    
    # Analyze tool usage patterns
    resume_tool_used = sum(1 for r in results if r.get("result", {}).get("field_analysis", {}).get("tools_used", {}).get("resume_search", False))
    personal_tool_used = sum(1 for r in results if r.get("result", {}).get("field_analysis", {}).get("tools_used", {}).get("personal_search", False))
    both_tools_used = sum(1 for r in results if (
        r.get("result", {}).get("field_analysis", {}).get("tools_used", {}).get("resume_search", False) and
        r.get("result", {}).get("field_analysis", {}).get("tools_used", {}).get("personal_search", False)
    ))
    
    logger.info(f"🎯 Intelligence Analysis:")
    logger.info(f"   • Fields using Resume Tool: {resume_tool_used}/{len(results)} ({(resume_tool_used/len(results)*100):.1f}%)")
    logger.info(f"   • Fields using Personal Tool: {personal_tool_used}/{len(results)} ({(personal_tool_used/len(results)*100):.1f}%)")
    logger.info(f"   • Fields using Both Tools: {both_tools_used}/{len(results)} ({(both_tools_used/len(results)*100):.1f}%)")
    logger.info("")
    
    # Show detailed results for each category
    professional_fields = ["job title", "company", "skills"]
    personal_fields = ["email", "phone", "address"]
    
    logger.info(f"📋 Detailed Results by Category:")
    logger.info("")
    
    for i, result in enumerate(results):
        field = result["field"]
        success = "✅" if result["success"] else "❌"
        answer = result.get("result", {}).get("answer", "No answer")
        data_source = result.get("result", {}).get("data_source", "unknown")
        
        logger.info(f"   {success} {field}")
        logger.info(f"      Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        logger.info(f"      Source: {data_source}")
        logger.info("")
    
    # Save test results
    test_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_summary": {
            "total_tests": len(results),
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "total_time": total_time
        },
        "performance_stats": perf_stats,
        "tool_usage": {
            "resume_tool_usage": resume_tool_used,
            "personal_tool_usage": personal_tool_used,
            "both_tools_usage": both_tools_used
        },
        "detailed_results": results
    }
    
    report_filename = f"smart_llm_langgraph_test_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(test_report, f, indent=2, default=str)
    
    logger.info(f"📄 Test report saved to: {report_filename}")
    
    if success_rate >= 80:
        logger.info("🎉 SMART LLM SERVICE WITH LANGGRAPH: EXCELLENT PERFORMANCE!")
    elif success_rate >= 60:
        logger.info("👍 SMART LLM SERVICE WITH LANGGRAPH: GOOD PERFORMANCE!")
    else:
        logger.warning("⚠️  SMART LLM SERVICE WITH LANGGRAPH: NEEDS IMPROVEMENT")
    
    logger.info("="*80)

if __name__ == "__main__":
    asyncio.run(test_smart_llm_langgraph()) 