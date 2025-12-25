#!/bin/bash
# Test runner script for NI DAQ application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  NI DAQ Application Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Parse arguments
TEST_TYPE="all"
COVERAGE=true
PARALLEL=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --gui)
            TEST_TYPE="gui"
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit         Run unit tests only"
            echo "  --integration  Run integration tests only"
            echo "  --gui          Run GUI tests only"
            echo "  --no-coverage  Disable coverage report"
            echo "  --parallel     Run tests in parallel (requires pytest-xdist)"
            echo "  -v, --verbose  Verbose output"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add test selection
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD -m unit"
        echo -e "${YELLOW}Running unit tests only...${NC}"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD -m integration"
        echo -e "${YELLOW}Running integration tests only...${NC}"
        ;;
    gui)
        PYTEST_CMD="$PYTEST_CMD -m gui"
        echo -e "${YELLOW}Running GUI tests only...${NC}"
        ;;
    *)
        echo -e "${YELLOW}Running all tests...${NC}"
        ;;
esac

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=term-missing --cov-report=html"
fi

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
    echo -e "${YELLOW}Running tests in parallel...${NC}"
fi

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${RED}pytest is not installed. Installing...${NC}"
    pip install pytest pytest-cov pytest-qt pytest-timeout pytest-xdist
fi

# Check required test dependencies
echo -e "${YELLOW}Checking test dependencies...${NC}"
python -c "
import sys
missing = []

try:
    import pytest
except ImportError:
    missing.append('pytest')

try:
    import pytest_cov
except ImportError:
    missing.append('pytest-cov')

try:
    import pytest_qt
except ImportError:
    missing.append('pytest-qt')

try:
    import pytest_timeout
except ImportError:
    missing.append('pytest-timeout')

if missing:
    print(f'Missing dependencies: {\", \".join(missing)}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Installing missing test dependencies...${NC}"
    pip install pytest pytest-cov pytest-qt pytest-timeout
fi

echo ""
echo -e "${GREEN}Running:${NC} $PYTEST_CMD"
echo ""

# Run tests
eval "$PYTEST_CMD"

# Capture exit code
EXIT_CODE=$?

echo ""
echo -e "${GREEN}========================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}  All tests passed!${NC}"

    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${GREEN}Coverage report generated:${NC}"
        echo "  - Terminal summary above"
        echo "  - HTML report: htmlcov/index.html"
        echo "  - XML report: coverage.xml"
    fi
else
    echo -e "${RED}  Some tests failed!${NC}"
fi

echo -e "${GREEN}========================================${NC}"

exit $EXIT_CODE
