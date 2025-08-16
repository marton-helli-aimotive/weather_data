#!/bin/bash

# Test runner script for comprehensive test execution
# Usage: ./run_tests.sh [test_type] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
TEST_TYPE="all"
COVERAGE_THRESHOLD=90
PARALLEL_WORKERS=auto
TIMEOUT=300
VERBOSE=false
FAIL_FAST=false
GENERATE_REPORT=true
UPLOAD_COVERAGE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--coverage-threshold)
            COVERAGE_THRESHOLD="$2"
            shift 2
            ;;
        -j|--parallel)
            PARALLEL_WORKERS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -x|--fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --no-report)
            GENERATE_REPORT=false
            shift
            ;;
        --upload-coverage)
            UPLOAD_COVERAGE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -t, --type TYPE           Test type: all, unit, integration, property, performance, contract, e2e"
            echo "  -c, --coverage-threshold  Minimum coverage percentage (default: 90)"
            echo "  -j, --parallel WORKERS    Number of parallel workers (default: auto)"
            echo "  --timeout SECONDS         Test timeout in seconds (default: 300)"
            echo "  -v, --verbose             Verbose output"
            echo "  -x, --fail-fast           Stop on first failure"
            echo "  --no-report               Skip report generation"
            echo "  --upload-coverage         Upload coverage to codecov"
            echo "  -h, --help                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date +'%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

print_info() {
    print_status "$BLUE" "INFO: $1"
}

print_success() {
    print_status "$GREEN" "SUCCESS: $1"
}

print_warning() {
    print_status "$YELLOW" "WARNING: $1"
}

print_error() {
    print_status "$RED" "ERROR: $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python version
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    if [[ $(echo "$python_version" | cut -d'.' -f1,2) < "3.10" ]]; then
        print_error "Python 3.10+ required, found $python_version"
        exit 1
    fi
    print_success "Python version: $python_version"
    
    # Check if pytest is installed
    if ! command -v pytest &> /dev/null; then
        print_error "pytest not found. Install with: pip install pytest"
        exit 1
    fi
    
    # Check if required packages are installed
    python -c "import pytest, coverage, hypothesis" 2>/dev/null || {
        print_error "Required test packages not installed. Run: pip install -e '.[test]'"
        exit 1
    }
    
    # Check Redis connection for integration tests
    if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "integration" || "$TEST_TYPE" == "e2e" ]]; then
        if ! python -c "import redis; r = redis.Redis(); r.ping()" 2>/dev/null; then
            print_warning "Redis not available. Integration and E2E tests may fail."
        else
            print_success "Redis connection verified"
        fi
    fi
}

# Function to setup test environment
setup_environment() {
    print_info "Setting up test environment..."
    
    # Create logs directory
    mkdir -p logs
    
    # Set environment variables for testing
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)/src"
    export WEATHER_PIPELINE_ENV="test"
    export WEATHER_PIPELINE_LOG_LEVEL="DEBUG"
    export REDIS_URL="${REDIS_URL:-redis://localhost:6379/1}"  # Use test database
    
    # Clear any existing coverage data
    if [[ -f .coverage ]]; then
        rm .coverage
    fi
    
    print_success "Environment setup complete"
}

# Function to run specific test suite
run_test_suite() {
    local suite=$1
    local suite_name=$2
    local pytest_args=("$@")
    shift 2
    
    print_info "Running $suite_name tests..."
    
    local cmd_args=()
    
    # Add parallel execution if specified
    if [[ "$PARALLEL_WORKERS" != "1" ]]; then
        cmd_args+=("-n" "$PARALLEL_WORKERS")
    fi
    
    # Add verbose output if requested
    if [[ "$VERBOSE" == "true" ]]; then
        cmd_args+=("-v")
    else
        cmd_args+=("-q")
    fi
    
    # Add fail-fast if requested
    if [[ "$FAIL_FAST" == "true" ]]; then
        cmd_args+=("-x")
    fi
    
    # Add timeout
    cmd_args+=("--timeout=$TIMEOUT")
    
    # Add coverage for unit tests
    if [[ "$suite" == "tests/unit" ]]; then
        cmd_args+=("--cov=src" "--cov-append" "--cov-report=term-missing")
    fi
    
    # Run the test suite
    if pytest "${cmd_args[@]}" "$suite" "${pytest_args[@]}"; then
        print_success "$suite_name tests passed"
        return 0
    else
        print_error "$suite_name tests failed"
        return 1
    fi
}

# Function to generate coverage report
generate_coverage_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        return 0
    fi
    
    print_info "Generating coverage report..."
    
    # Generate HTML report
    coverage html -d htmlcov/
    
    # Generate XML report for CI
    coverage xml
    
    # Check coverage threshold
    local coverage_percent=$(coverage report | tail -1 | awk '{print $4}' | sed 's/%//')
    
    if [[ $(echo "$coverage_percent >= $COVERAGE_THRESHOLD" | bc -l) == 1 ]]; then
        print_success "Coverage: ${coverage_percent}% (threshold: ${COVERAGE_THRESHOLD}%)"
    else
        print_error "Coverage: ${coverage_percent}% below threshold: ${COVERAGE_THRESHOLD}%"
        return 1
    fi
    
    print_info "Coverage report available at: htmlcov/index.html"
    
    # Upload coverage if requested
    if [[ "$UPLOAD_COVERAGE" == "true" && -n "$CODECOV_TOKEN" ]]; then
        print_info "Uploading coverage to Codecov..."
        bash <(curl -s https://codecov.io/bash) -t "$CODECOV_TOKEN"
    fi
}

# Function to generate test report
generate_test_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        return 0
    fi
    
    print_info "Generating test report..."
    
    # Create reports directory
    mkdir -p reports
    
    # Generate JUnit XML report
    local report_file="reports/test-results-$(date +%Y%m%d-%H%M%S).xml"
    
    # Re-run tests with JUnit output (for CI/CD integration)
    pytest --junitxml="$report_file" tests/ > /dev/null 2>&1 || true
    
    print_success "Test report generated: $report_file"
}

# Main execution function
main() {
    print_info "Starting test execution with type: $TEST_TYPE"
    
    # Check prerequisites
    check_prerequisites
    
    # Setup environment
    setup_environment
    
    local exit_code=0
    
    # Run tests based on type
    case $TEST_TYPE in
        "unit")
            run_test_suite "tests/unit" "Unit" || exit_code=1
            ;;
        "integration")
            run_test_suite "tests/integration" "Integration" || exit_code=1
            ;;
        "property")
            run_test_suite "tests/property" "Property-based" || exit_code=1
            ;;
        "performance")
            run_test_suite "tests/performance" "Performance" "-m" "not slow" || exit_code=1
            ;;
        "contract")
            run_test_suite "tests/contract" "Contract" || exit_code=1
            ;;
        "e2e")
            run_test_suite "tests/e2e" "End-to-end" || exit_code=1
            ;;
        "all")
            print_info "Running comprehensive test suite..."
            
            # Run all test types in order
            run_test_suite "tests/unit" "Unit" || exit_code=1
            
            if [[ $exit_code -eq 0 || "$FAIL_FAST" != "true" ]]; then
                run_test_suite "tests/integration" "Integration" || exit_code=1
            fi
            
            if [[ $exit_code -eq 0 || "$FAIL_FAST" != "true" ]]; then
                run_test_suite "tests/property" "Property-based" || exit_code=1
            fi
            
            if [[ $exit_code -eq 0 || "$FAIL_FAST" != "true" ]]; then
                run_test_suite "tests/performance" "Performance" "-m" "not slow" || exit_code=1
            fi
            
            if [[ $exit_code -eq 0 || "$FAIL_FAST" != "true" ]]; then
                run_test_suite "tests/contract" "Contract" || exit_code=1
            fi
            
            if [[ $exit_code -eq 0 || "$FAIL_FAST" != "true" ]]; then
                run_test_suite "tests/e2e" "End-to-end" || exit_code=1
            fi
            ;;
        *)
            print_error "Unknown test type: $TEST_TYPE"
            exit 1
            ;;
    esac
    
    # Generate reports
    if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "unit" ]]; then
        generate_coverage_report || exit_code=1
    fi
    
    generate_test_report
    
    # Final status
    if [[ $exit_code -eq 0 ]]; then
        print_success "All tests completed successfully!"
    else
        print_error "Some tests failed. Check the output above for details."
    fi
    
    exit $exit_code
}

# Run main function
main "$@"
