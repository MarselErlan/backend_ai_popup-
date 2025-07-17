#!/usr/bin/env python3
"""
🧠 ULTIMATE SMART LLM SERVICE COMPREHENSIVE TEST
Tests all features of the world-class LLM service
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
from app.services.llm_service import UltimateSmartLLMService

async def test_ultimate_smart_llm_service():
    """Comprehensive test of the Ultimate Smart LLM Service"""
    
    logger.info("🧠 STARTING ULTIMATE SMART LLM SERVICE COMPREHENSIVE TEST")
    logger.info("="*100)
    
    # Initialize the service
    try:
        llm_service = UltimateSmartLLMService()
        logger.info("✅ Ultimate Smart LLM Service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        return
    
    # Test fields covering all major types
    test_fields = [
        # Contact fields (should use pattern extraction)
        {"label": "Email Address", "expected_type": "email"},
        {"label": "Phone Number", "expected_type": "phone"},
        {"label": "Full Name", "expected_type": "name"},
        {"label": "First Name", "expected_type": "name"},
        {"label": "Last Name", "expected_type": "name"},
        {"label": "Postal Code", "expected_type": "postal_code"},
        {"label": "Home Address", "expected_type": "address"},
        
        # Professional fields (should use resume data)
        {"label": "Current Job Title", "expected_type": "general"},
        {"label": "Company Name", "expected_type": "general"},
        {"label": "Work Experience", "expected_type": "general"},
        {"label": "Technical Skills", "expected_type": "general"},
        {"label": "Education Background", "expected_type": "general"},
        
        # Complex application fields (should use LLM generation)
        {"label": "Why do you want to work at our startup?", "expected_type": "general"},
        {"label": "Describe your experience with fast-paced environments", "expected_type": "general"},
        {"label": "What makes you a good fit for this role?", "expected_type": "general"},
        {"label": "Are you authorized to work in the US?", "expected_type": "general"},
        {"label": "What are your salary expectations?", "expected_type": "general"},
    ]
    
    user_id = "test_user_ultimate"
    test_results = []
    total_start_time = time.time()
    
    logger.info(f"🎯 Testing {len(test_fields)} different field types")
    logger.info("="*100)
    
    for i, field in enumerate(test_fields, 1):
        logger.info(f"\n🔍 TEST {i}/{len(test_fields)}: {field['label']}")
        logger.info("-" * 80)
        
        field_start_time = time.time()
        
        try:
            # Test the field
            result = await llm_service.generate_field_answer(
                field_label=field["label"],
                user_id=user_id,
                field_context={"test_mode": True}
            )
            
            field_processing_time = time.time() - field_start_time
            
            # Analyze results
            success = result.get("status") == "success"
            answer = result.get("answer", "")
            data_source = result.get("data_source", "unknown")
            tier_used = result.get("performance_metrics", {}).get("tier_used", "unknown")
            confidence_score = result.get("performance_metrics", {}).get("confidence_score", 0)
            
            # Log results
            status_emoji = "✅" if success else "❌"
            logger.info(f"{status_emoji} RESULT:")
            logger.info(f"   📝 Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            logger.info(f"   📊 Data Source: {data_source}")
            logger.info(f"   🏆 Tier Used: {tier_used}")
            logger.info(f"   🎯 Confidence: {confidence_score:.1f}%")
            logger.info(f"   ⏱️  Processing Time: {field_processing_time:.3f}s")
            
            # Performance grade
            if field_processing_time < 0.5:
                performance_grade = "A+ (EXCELLENT)"
            elif field_processing_time < 1.0:
                performance_grade = "A (VERY GOOD)"
            elif field_processing_time < 2.0:
                performance_grade = "B (GOOD)"
            else:
                performance_grade = "C (ACCEPTABLE)"
            
            logger.info(f"   🚀 Performance Grade: {performance_grade}")
            
            # Store results
            test_results.append({
                "field_label": field["label"],
                "expected_type": field["expected_type"],
                "success": success,
                "answer": answer,
                "data_source": data_source,
                "tier_used": tier_used,
                "confidence_score": confidence_score,
                "processing_time": field_processing_time,
                "performance_grade": performance_grade,
                "full_result": result
            })
            
        except Exception as e:
            logger.error(f"❌ Test failed for '{field['label']}': {e}")
            test_results.append({
                "field_label": field["label"],
                "expected_type": field["expected_type"],
                "success": False,
                "error": str(e),
                "processing_time": time.time() - field_start_time
            })
    
    total_processing_time = time.time() - total_start_time
    
    # Generate comprehensive test report
    logger.info("\n" + "="*100)
    logger.info("📊 ULTIMATE SMART LLM SERVICE - COMPREHENSIVE TEST REPORT")
    logger.info("="*100)
    
    # Success metrics
    successful_tests = [r for r in test_results if r.get("success", False)]
    success_rate = len(successful_tests) / len(test_results) * 100
    
    logger.info(f"🎯 OVERALL PERFORMANCE:")
    logger.info(f"   ✅ Success Rate: {success_rate:.1f}% ({len(successful_tests)}/{len(test_results)})")
    logger.info(f"   ⏱️  Total Processing Time: {total_processing_time:.3f}s")
    logger.info(f"   📈 Average Time per Field: {total_processing_time/len(test_results):.3f}s")
    
    # Performance breakdown
    if successful_tests:
        processing_times = [r["processing_time"] for r in successful_tests]
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        logger.info(f"\n⚡ PERFORMANCE BREAKDOWN:")
        logger.info(f"   📊 Average Processing Time: {avg_time:.3f}s")
        logger.info(f"   🚀 Fastest Response: {min_time:.3f}s")
        logger.info(f"   🐌 Slowest Response: {max_time:.3f}s")
        
        # Tier usage analysis
        tier_usage = {}
        for result in successful_tests:
            tier = result.get("tier_used", "unknown")
            tier_usage[tier] = tier_usage.get(tier, 0) + 1
        
        logger.info(f"\n🏆 TIER USAGE ANALYSIS:")
        for tier, count in tier_usage.items():
            percentage = count / len(successful_tests) * 100
            logger.info(f"   {tier}: {count} times ({percentage:.1f}%)")
        
        # Data source analysis
        source_usage = {}
        for result in successful_tests:
            source = result.get("data_source", "unknown")
            source_usage[source] = source_usage.get(source, 0) + 1
        
        logger.info(f"\n📊 DATA SOURCE ANALYSIS:")
        for source, count in source_usage.items():
            percentage = count / len(successful_tests) * 100
            logger.info(f"   {source}: {count} times ({percentage:.1f}%)")
        
        # Performance grades
        grade_counts = {}
        for result in successful_tests:
            grade = result.get("performance_grade", "unknown")
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        logger.info(f"\n🚀 PERFORMANCE GRADES:")
        for grade, count in grade_counts.items():
            percentage = count / len(successful_tests) * 100
            logger.info(f"   {grade}: {count} times ({percentage:.1f}%)")
    
    # Service performance metrics
    performance_stats = llm_service.get_performance_stats()
    cache_stats = llm_service._get_cache_analytics()
    
    logger.info(f"\n💾 SYSTEM PERFORMANCE METRICS:")
    logger.info(f"   📈 Total Requests: {performance_stats.get('total_requests', 0)}")
    logger.info(f"   ⚡ Cache Hit Rate: {cache_stats.get('cache_hit_rate', 0):.1%}")
    logger.info(f"   🎯 Success Rate: {performance_stats.get('success_rate', 0):.1%}")
    logger.info(f"   🚀 Cache Exits: {performance_stats.get('cache_exits', 0)}")
    logger.info(f"   🏆 Tier 1 Exits: {performance_stats.get('tier_1_exits', 0)}")
    logger.info(f"   🥈 Tier 2 Exits: {performance_stats.get('tier_2_exits', 0)}")
    logger.info(f"   🥉 Tier 3 Completions: {performance_stats.get('tier_3_completions', 0)}")
    
    # Sample answers showcase
    logger.info(f"\n📝 SAMPLE ANSWERS SHOWCASE:")
    logger.info("-" * 80)
    
    for result in successful_tests[:5]:  # Show first 5 successful results
        field_label = result["field_label"]
        answer = result["answer"]
        data_source = result["data_source"]
        tier_used = result["tier_used"]
        
        logger.info(f"🔍 {field_label}:")
        logger.info(f"   📝 Answer: {answer}")
        logger.info(f"   📊 Source: {data_source} | Tier: {tier_used}")
        logger.info("")
    
    # Final assessment
    logger.info("="*100)
    if success_rate >= 90:
        final_grade = "🏆 EXCELLENT - Ultimate Smart LLM Service is performing at world-class level!"
    elif success_rate >= 80:
        final_grade = "🥇 VERY GOOD - Ultimate Smart LLM Service is performing well!"
    elif success_rate >= 70:
        final_grade = "🥈 GOOD - Ultimate Smart LLM Service is performing adequately!"
    else:
        final_grade = "🥉 NEEDS IMPROVEMENT - Ultimate Smart LLM Service needs optimization!"
    
    logger.info(f"🎯 FINAL ASSESSMENT: {final_grade}")
    logger.info("="*100)
    
    # Save detailed results
    test_report = {
        "test_summary": {
            "total_tests": len(test_results),
            "successful_tests": len(successful_tests),
            "success_rate": success_rate,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(test_results),
            "final_grade": final_grade
        },
        "detailed_results": test_results,
        "system_metrics": {
            "performance_stats": performance_stats,
            "cache_stats": cache_stats
        },
        "tier_usage": tier_usage if successful_tests else {},
        "data_source_usage": source_usage if successful_tests else {},
        "performance_grades": grade_counts if successful_tests else {}
    }
    
    # Save to file
    report_filename = f"ultimate_smart_llm_test_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(test_report, f, indent=2, default=str)
    
    logger.info(f"📁 Detailed test report saved to: {report_filename}")
    logger.info("🧠 ULTIMATE SMART LLM SERVICE TEST COMPLETED!")
    
    return test_report

if __name__ == "__main__":
    asyncio.run(test_ultimate_smart_llm_service()) 