#!/bin/bash
# Comprehensive test runner for generic model support
# Automatically sets up LD_PRELOAD for vLLM

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Setup LD_PRELOAD for vLLM
export LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Generic Model Support - Comprehensive Test Suite${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Test 1: Basic functionality (no vLLM imports)
echo -e "${YELLOW}[1/4] Running basic functionality tests...${NC}"
python run_generic_model_tests.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Basic tests passed${NC}"
else
    echo -e "${RED}✗ Basic tests failed${NC}"
    exit 1
fi
echo ""

# Test 2: Parallelism examples
echo -e "${YELLOW}[2/4] Running parallelism examples...${NC}"
python examples/generic_model_parallelism.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Parallelism examples passed${NC}"
else
    echo -e "${RED}✗ Parallelism examples failed${NC}"
    exit 1
fi
echo ""

# Test 3: vLLM integration (model registration)
echo -e "${YELLOW}[3/4] Running vLLM integration tests...${NC}"
python test_vllm_integration.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ vLLM integration tests passed${NC}"
else
    echo -e "${RED}✗ vLLM integration tests failed${NC}"
    exit 1
fi
echo ""

# Test 4: Advanced vLLM tests (instantiation, interface detection)
echo -e "${YELLOW}[4/4] Running advanced vLLM tests...${NC}"
python test_vllm_advanced.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Advanced vLLM tests passed${NC}"
else
    echo -e "${RED}✗ Advanced vLLM tests failed${NC}"
    exit 1
fi
echo ""

echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}ALL TESTS PASSED!${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Summary:"
echo "  ✓ Flash attention with backward pass works"
echo "  ✓ Model training with PyTorch works"
echo "  ✓ Parallelism utilities work"
echo "  ✓ Model registration with vLLM works"
echo "  ✓ Model properly detected as text generation model"
echo "  ✓ vLLM can instantiate and use our custom model"
echo ""
echo "The RFC implementation is validated and ready!"
