#!/usr/bin/env python3
"""
🧪 Test Middleware Import
Check if DeepTrackingMiddleware can be imported without errors
"""

def test_middleware_import():
    """Test if the middleware can be imported"""
    
    print("🧪 Testing DeepTrackingMiddleware Import")
    print("=" * 60)
    
    try:
        print("📦 Importing DeepTrackingMiddleware...")
        from app.services.integrated_usage_analyzer import DeepTrackingMiddleware
        print("✅ Import successful!")
        
        print("🔍 Checking middleware class...")
        print(f"   • Class: {DeepTrackingMiddleware}")
        print(f"   • Base classes: {DeepTrackingMiddleware.__bases__}")
        print(f"   • Methods: {[m for m in dir(DeepTrackingMiddleware) if not m.startswith('_')]}")
        
        # Check if it has the dispatch method
        if hasattr(DeepTrackingMiddleware, 'dispatch'):
            print("✅ dispatch method found")
        else:
            print("❌ dispatch method missing")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_middleware_import()
    if success:
        print("\n🎉 Middleware import test passed!")
    else:
        print("\n❌ Middleware import test failed!") 