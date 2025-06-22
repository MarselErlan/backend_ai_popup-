#!/usr/bin/env python3
"""
Demo script for testing function and class tracking functionality.

This script demonstrates how to use the tracking decorators and 
shows the detailed function/class usage data in reports.
"""

import time
import asyncio
from app.utils.usage_decorators import track_function, track_class, track_service_function
from app.services.integrated_usage_analyzer import get_analyzer


# Example tracked functions
@track_function
def calculate_sum(a: int, b: int) -> int:
    """Simple function to demonstrate tracking"""
    time.sleep(0.01)  # Simulate some work
    return a + b


@track_function
def process_text(text: str, uppercase: bool = False) -> str:
    """Text processing function with optional parameter"""
    time.sleep(0.005)
    if uppercase:
        return text.upper()
    return text.lower()


@track_service_function
def fetch_user_data(user_id: int, include_profile: bool = True) -> dict:
    """Simulate fetching user data from database"""
    time.sleep(0.02)  # Simulate database query
    
    if user_id <= 0:
        raise ValueError("Invalid user ID")
    
    data = {"id": user_id, "name": f"User_{user_id}"}
    if include_profile:
        data["profile"] = {"email": f"user{user_id}@example.com"}
    
    return data


# Example tracked classes
@track_class
class UserService:
    """Service class for user operations"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_count = 0
    
    def create_user(self, name: str, email: str) -> dict:
        """Create a new user"""
        time.sleep(0.01)
        self.connection_count += 1
        return {"id": self.connection_count, "name": name, "email": email}
    
    def get_user(self, user_id: int) -> dict:
        """Get user by ID"""
        time.sleep(0.008)
        self.connection_count += 1
        return {"id": user_id, "name": f"User_{user_id}"}
    
    def update_user(self, user_id: int, **kwargs) -> dict:
        """Update user data"""
        time.sleep(0.015)
        self.connection_count += 1
        return {"id": user_id, **kwargs}


@track_class
class DataProcessor:
    """Data processing utility class"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.processed_count = 0
    
    def process_batch(self, data_list: list) -> list:
        """Process a batch of data"""
        time.sleep(0.02)
        self.processed_count += len(data_list)
        return [item.upper() if isinstance(item, str) else item for item in data_list]
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return {"processed_count": self.processed_count, "batch_size": self.batch_size}


def run_function_demos():
    """Run function tracking demonstrations"""
    print("🔧 Testing Function Tracking...")
    
    # Test basic function calls
    result1 = calculate_sum(10, 20)
    result2 = calculate_sum(5, 15)
    result3 = calculate_sum(100, 200)
    
    print(f"   Sum results: {result1}, {result2}, {result3}")
    
    # Test function with different parameters
    text1 = process_text("Hello World", uppercase=True)
    text2 = process_text("Python Rocks", uppercase=False)
    text3 = process_text("FastAPI is Great")
    
    print(f"   Text results: {text1}, {text2}, {text3}")
    
    # Test function with success and error cases
    try:
        user1 = fetch_user_data(1, include_profile=True)
        user2 = fetch_user_data(2, include_profile=False)
        print(f"   User data: {user1}, {user2}")
    except Exception as e:
        print(f"   User fetch error: {e}")
    
    # Test error case
    try:
        fetch_user_data(-1)
    except ValueError as e:
        print(f"   Expected error caught: {e}")


def run_class_demos():
    """Run class tracking demonstrations"""
    print("🏗️ Testing Class Tracking...")
    
    # Test UserService class
    user_service = UserService("postgresql://localhost/test")
    
    user1 = user_service.create_user("Alice", "alice@example.com")
    user2 = user_service.create_user("Bob", "bob@example.com")
    
    retrieved_user = user_service.get_user(1)
    updated_user = user_service.update_user(1, name="Alice Smith", active=True)
    
    print(f"   Created users: {user1}, {user2}")
    print(f"   Retrieved: {retrieved_user}")
    print(f"   Updated: {updated_user}")
    
    # Test DataProcessor class
    processor1 = DataProcessor(batch_size=50)
    processor2 = DataProcessor(batch_size=100)
    
    batch1 = processor1.process_batch(["hello", "world", "python"])
    batch2 = processor2.process_batch(["fastapi", "uvicorn", "starlette"])
    
    stats1 = processor1.get_stats()
    stats2 = processor2.get_stats()
    
    print(f"   Processed batches: {batch1}, {batch2}")
    print(f"   Stats: {stats1}, {stats2}")


def run_mixed_usage_demo():
    """Demonstrate mixed function and class usage"""
    print("🔄 Testing Mixed Usage Patterns...")
    
    # Create service instance
    service = UserService("sqlite:///demo.db")
    
    # Use functions and methods together
    for i in range(1, 4):
        # Calculate some value
        calculated = calculate_sum(i * 10, i * 5)
        
        # Process text
        processed_text = process_text(f"User_{i}", uppercase=True)
        
        # Create user with calculated/processed data
        user = service.create_user(processed_text, f"user{calculated}@example.com")
        
        # Fetch user data
        user_data = fetch_user_data(i, include_profile=True)
        
        print(f"   Iteration {i}: calc={calculated}, text={processed_text}, user={user}")


async def main():
    """Main demo function"""
    print("🚀 Starting Function & Class Tracking Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = get_analyzer()
    analyzer.start_monitoring()
    
    try:
        # Run demonstrations
        run_function_demos()
        print()
        
        run_class_demos()
        print()
        
        run_mixed_usage_demo()
        print()
        
        # Let some time pass to see timing data
        print("⏱️ Waiting a moment for timing data...")
        await asyncio.sleep(1)
        
        # Show current status
        status = analyzer.get_status()
        print(f"📊 Current Analysis Status:")
        print(f"   Functions tracked: {status.get('functions_discovered', 0)}")
        print(f"   Classes tracked: {status.get('classes_discovered', 0)}")
        print(f"   Duration: {status.get('duration_minutes', 0):.2f} minutes")
        
    finally:
        # Stop monitoring and generate report
        print("\n🔄 Generating analysis report...")
        analyzer.stop_monitoring()
        
        print("✅ Demo completed! Check the reports:")
        print("   📄 tests/reports/integrated_analysis_current.json")
        print("   🌐 tests/reports/integrated_analysis_current.html")
        print("\n💡 Look for the 'Most Used Functions' and 'Most Used Classes' tables!")


if __name__ == "__main__":
    asyncio.run(main()) 