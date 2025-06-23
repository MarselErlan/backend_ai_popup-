#!/usr/bin/env python3
"""
Import Verification Test
This verifies that all imports work correctly for the RealTimeUsageAnalyzer
"""

def test_imports():
    """Test all possible import methods"""
    print("🧪 Testing RealTimeUsageAnalyzer imports...")
    
    # Test 1: Direct import
    try:
        from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer
        print("✅ Test 1 PASSED: Direct import works")
        
        # Test instantiation
        analyzer = RealTimeUsageAnalyzer()
        print("✅ Test 1b PASSED: Analyzer instantiation works")
        
        # Test deep analyzer integration
        if hasattr(analyzer, 'deep_analyzer'):
            print("✅ Test 1c PASSED: Deep analyzer integration works")
        else:
            print("❌ Test 1c FAILED: Deep analyzer not found")
            
    except ImportError as e:
        print(f"❌ Test 1 FAILED: Direct import failed - {e}")
        return False
    except Exception as e:
        print(f"❌ Test 1 FAILED: Instantiation failed - {e}")
        return False
    
    # Test 2: Alternative import method
    try:
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).parent
        tests_path = project_root / "tests"
        
        if str(tests_path) not in sys.path:
            sys.path.insert(0, str(tests_path))
            
        from tests.analysis.realtime_usage_analyzer import RealTimeUsageAnalyzer as RealTimeUsageAnalyzer2
        print("✅ Test 2 PASSED: Alternative import works")
        
    except ImportError as e:
        print(f"❌ Test 2 FAILED: Alternative import failed - {e}")
    except Exception as e:
        print(f"❌ Test 2 FAILED: Alternative import error - {e}")
    
    # Test 3: Check class attributes
    try:
        analyzer = RealTimeUsageAnalyzer()
        
        required_attrs = ['deep_analyzer', 'monitoring', 'start_monitoring', 'stop_monitoring']
        for attr in required_attrs:
            if hasattr(analyzer, attr):
                print(f"✅ Test 3 PASSED: {attr} attribute exists")
            else:
                print(f"❌ Test 3 FAILED: {attr} attribute missing")
                
    except Exception as e:
        print(f"❌ Test 3 FAILED: Attribute check failed - {e}")
    
    print("\n🎉 Import verification complete!")
    return True

if __name__ == "__main__":
    test_imports() 