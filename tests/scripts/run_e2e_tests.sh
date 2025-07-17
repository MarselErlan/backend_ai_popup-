#!/bin/bash
"""
🚀 END-TO-END TEST RUNNER SCRIPT
Simple script to run comprehensive end-to-end tests from organized structure
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $1 in
        "info") echo -e "${BLUE}[INFO]${NC} $2" ;;
        "success") echo -e "${GREEN}[SUCCESS]${NC} $2" ;;
        "warning") echo -e "${YELLOW}[WARNING]${NC} $2" ;;
        "error") echo -e "${RED}[ERROR]${NC} $2" ;;
    esac
}

# Print header
echo "=================================================================="
echo "🚀 END-TO-END TESTING FOR MAIN.PY (ORGANIZED STRUCTURE)"
echo "=================================================================="
echo ""

# Check if we're in the scripts directory and navigate to tests root
if [ ! -d "../e2e" ]; then
    print_status "error" "Please run this script from the tests/scripts directory"
    exit 1
fi

# Navigate to tests root directory
cd ..

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_status "error" "Python 3 is required but not installed"
    exit 1
fi

# Check if required packages are available
print_status "info" "Checking required packages..."

# Check if aiohttp is available
python3 -c "import aiohttp" 2>/dev/null
if [ $? -ne 0 ]; then
    print_status "warning" "aiohttp not found, installing..."
    pip3 install aiohttp
fi

# Check if server is running
print_status "info" "Checking if server is running..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_status "success" "Server is running"
else
    print_status "error" "Server is not running on localhost:8000"
    print_status "info" "Please start the server first:"
    echo "  cd .."
    echo "  python main.py"
    exit 1
fi

echo ""
print_status "info" "Starting end-to-end tests from organized structure..."
echo ""

# Run the main end-to-end test from the e2e folder
if [ -f "e2e/test_end_to_end.py" ]; then
    print_status "info" "Running primary E2E test..."
    cd e2e
    python3 test_end_to_end.py
    e2e_result=$?
    cd ..
else
    print_status "error" "Primary E2E test not found: e2e/test_end_to_end.py"
    e2e_result=1
fi

# Check the exit code
if [ $e2e_result -eq 0 ]; then
    echo ""
    print_status "success" "Primary end-to-end test completed successfully!"
    
    # Optionally run additional tests
    echo ""
    print_status "info" "Running additional test categories..."
    
    # Run integration tests
    if [ -d "integration" ]; then
        print_status "info" "Running integration tests..."
        integration_passed=0
        integration_total=0
        
        for test_file in integration/test_*.py; do
            if [ -f "$test_file" ]; then
                integration_total=$((integration_total + 1))
                print_status "info" "Running $(basename $test_file)..."
                cd integration
                python3 $(basename $test_file) > /dev/null 2>&1
                if [ $? -eq 0 ]; then
                    integration_passed=$((integration_passed + 1))
                    print_status "success" "$(basename $test_file) - PASSED"
                else
                    print_status "error" "$(basename $test_file) - FAILED"
                fi
                cd ..
            fi
        done
        
        if [ $integration_total -gt 0 ]; then
            print_status "info" "Integration tests: $integration_passed/$integration_total passed"
        fi
    fi
    
else
    echo ""
    print_status "error" "End-to-end test failed!"
    echo ""
    print_status "info" "To run all test suites:"
    echo "  python3 scripts/run_all_tests.py"
    exit 1
fi

echo ""
echo "=================================================================="
print_status "success" "🎉 Organized testing completed!"
print_status "info" "Check the reports/ folder for detailed results"
print_status "info" "Test structure:"
echo "  📂 e2e/ - End-to-end tests"
echo "  📂 integration/ - Integration tests"
echo "  📂 unit/ - Unit tests"
echo "  📂 performance/ - Performance tests"
echo "  📂 analysis/ - Code analysis"
echo "  📂 scripts/ - Test runners"
echo "  📂 fixtures/ - Test data"
echo "  📂 reports/ - Test reports"
echo "==================================================================" 