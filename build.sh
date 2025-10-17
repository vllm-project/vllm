#!/bin/bash
set -e

# Build script for vLLM Docker image with local DeepEP (NIXL support)
# Usage: ./build.sh [OPTIONS]

# Default configuration
DEEPEP_PATH="${DEEPEP_PATH:-../DeepEP}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
IMAGE_TAG="${IMAGE_TAG:-vllm-elastic-ep:latest_local_deepep}"
DOCKERFILE="docker/Dockerfile.local_deepep"
RUN_WHEEL_CHECK="${RUN_WHEEL_CHECK:-false}"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-}"  # Empty = use temp dir
KEEP_CONTEXT="${KEEP_CONTEXT:-false}"
TARGET_STAGE="${TARGET_STAGE:-vllm-openai}"
NO_CACHE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build vLLM Docker image with local DeepEP (NIXL support)

OPTIONS:
    -d, --deepep PATH         Path to DeepEP directory (default: ../DeepEP)
    -c, --cuda VERSION        CUDA version (default: 12.8.1)
    -p, --python VERSION      Python version (default: 3.12)
    -t, --tag TAG            Docker image tag (default: vllm-elastic-ep:latest_local_deepep)
    --target STAGE           Docker target stage (default: vllm-openai)
                            Options: vllm-openai, vllm-base, build, deepep-build
    --build-context DIR      Use specific directory for build context (default: temp dir)
    --keep-context           Keep build context after build (useful for debugging)
    --run-wheel-check        Run wheel size check during build
    --no-cache               Build without Docker cache
    -h, --help               Show this help message

ENVIRONMENT VARIABLES:
    DEEPEP_PATH              Path to DeepEP directory
    CUDA_VERSION             CUDA version for the build
    PYTHON_VERSION           Python version for the build
    IMAGE_TAG                Docker image tag
    DOCKER_BUILDKIT          Enable BuildKit (default: 1)

EXAMPLES:
    # Basic build with defaults
    $0

    # Custom DeepEP path and image tag
    $0 -d /path/to/DeepEP -t my-vllm:latest

    # Build with specific CUDA and Python versions
    $0 -c 12.9.0 -p 3.11

    # Build only the base image (without OpenAI server)
    $0 --target vllm-base

    # Keep build context for inspection
    $0 --keep-context --build-context ./my-build-context
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--deepep)
            DEEPEP_PATH="$2"
            shift 2
            ;;
        -c|--cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --target)
            TARGET_STAGE="$2"
            shift 2
            ;;
        --build-context)
            BUILD_CONTEXT_DIR="$2"
            shift 2
            ;;
        --keep-context)
            KEEP_CONTEXT=true
            shift
            ;;
        --run-wheel-check)
            RUN_WHEEL_CHECK=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
         -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Enable Docker BuildKit
export DOCKER_BUILDKIT=1

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}vLLM Build with Local DeepEP (NIXL)${NC}"
echo -e "${BLUE}Using Clean Approach (Script Fork)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Configuration:"
echo "  DeepEP Path:      $DEEPEP_PATH"
echo "  CUDA Version:     $CUDA_VERSION"
echo "  Python Version:   $PYTHON_VERSION"
echo "  Image Tag:        $IMAGE_TAG"
echo "  Target Stage:     $TARGET_STAGE"
echo "  Dockerfile:       $DOCKERFILE"
echo "  Run Wheel Check:  $RUN_WHEEL_CHECK"
echo ""

# Check if DeepEP directory exists
if [ ! -d "$DEEPEP_PATH" ]; then
    echo -e "${RED}Error: DeepEP directory not found at $DEEPEP_PATH${NC}"
    echo "Please set DEEPEP_PATH environment variable or use -d option"
    exit 1
fi

# Make DeepEP path absolute
DEEPEP_PATH=$(realpath "$DEEPEP_PATH")
echo "Using DeepEP from: $DEEPEP_PATH"

# Check if NIXL support is available in DeepEP
if grep -q "nixl_buffer" "$DEEPEP_PATH/deep_ep/buffer.py" 2>/dev/null; then
    echo -e "${GREEN}✓ NIXL support found in DeepEP${NC}"
else
    echo -e "${YELLOW}Warning: NIXL support may not be available in DeepEP${NC}"
    echo -e "${YELLOW}Make sure your DeepEP has NIXL methods (nixl_buffer, update_memory_buffers)${NC}"
fi

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}Error: Dockerfile not found at $DOCKERFILE${NC}"
    echo "Please run this script from the vLLM root directory"
    exit 1
fi

# Create or use build context directory
if [ -z "$BUILD_CONTEXT_DIR" ]; then
    BUILD_CONTEXT=$(mktemp -d -t vllm-build.XXXXXX)
    echo -e "${BLUE}Creating temporary build context in $BUILD_CONTEXT${NC}"
    CLEANUP_CONTEXT=true
else
    BUILD_CONTEXT=$(realpath "$BUILD_CONTEXT_DIR")
    mkdir -p "$BUILD_CONTEXT"
    echo -e "${BLUE}Using build context in $BUILD_CONTEXT${NC}"
    CLEANUP_CONTEXT=false
    if [ "$KEEP_CONTEXT" = "true" ]; then
        CLEANUP_CONTEXT=false
    fi
fi

# Copy vLLM source to build context
echo "Copying vLLM source to build context..."
rsync -a --exclude='*.pyc' \
         --exclude='__pycache__' \
         --exclude='.git' \
         --exclude='build' \
         --exclude='dist' \
         --exclude='*.egg-info' \
         --exclude='.cache' \
         --exclude='.pytest_cache' \
         --exclude='vllm_venv' \
         --exclude='venv' \
         --exclude='*.so' \
         --exclude='*.whl' \
         . "$BUILD_CONTEXT/vllm/"

# Copy DeepEP to build context
echo "Copying DeepEP source to build context..."
rsync -a --exclude='*.pyc' \
         --exclude='__pycache__' \
         --exclude='.git' \
         --exclude='build' \
         --exclude='dist' \
         --exclude='*.egg-info' \
         --exclude='.cache' \
         --exclude='*.so' \
         --exclude='*.whl' \
         "$DEEPEP_PATH" "$BUILD_CONTEXT/vllm/DeepEP/"

# Create a .dockerignore to exclude unnecessary files
cat > "$BUILD_CONTEXT/vllm/.dockerignore" << EOF
# Python
**/__pycache__
**/*.pyc
**/*.pyo
**/*.pyd
.Python

# Git
**/.git
**/.gitignore

# Build artifacts
**/build
**/dist
**/*.egg-info
**/*.egg

# Cache
**/.cache
**/.pytest_cache
**/.tox

# Virtual environments
**/vllm_venv
**/venv
**/env
**/ENV

# IDE
**/.vscode
**/.idea
**/*.swp
**/*.swo

# OS
**/.DS_Store
**/Thumbs.db

# Compiled
**/*.so
**/*.dll
**/*.dylib

# Wheels (except in deepep-dist)
**/*.whl
!deepep-dist/*.whl
EOF

echo -e "${GREEN}Build context prepared${NC}"
echo ""

# Build the Docker image
echo -e "${BLUE}Starting Docker build...${NC}"
echo "Running: docker build -f $DOCKERFILE --target $TARGET_STAGE ..."
echo ""

cd "$BUILD_CONTEXT/vllm"

docker build $NO_CACHE \
    -f "$DOCKERFILE" \
    --target "$TARGET_STAGE" \
    --tag "$IMAGE_TAG" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg RUN_WHEEL_CHECK="$RUN_WHEEL_CHECK" \
    --progress=plain \
    .

BUILD_STATUS=$?

# Cleanup
if [ "$CLEANUP_CONTEXT" = "true" ] && [ "$KEEP_CONTEXT" != "true" ]; then
    echo ""
    echo "Cleaning up build context..."
    rm -rf "$BUILD_CONTEXT"
elif [ "$KEEP_CONTEXT" = "true" ]; then
    echo ""
    echo -e "${YELLOW}Build context kept at: $BUILD_CONTEXT${NC}"
fi

if [ $BUILD_STATUS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Image: $IMAGE_TAG"
    echo ""
    echo "To run the image with NIXL DeepEP:"
    echo ""
    echo "  docker run --gpus all -p 8000:8000 \\"
    echo "    -e VLLM_ALL2ALL_BACKEND=nixl_deepep_low_latency \\"
    echo "    -e NIXL_ETCD_ENDPOINTS=etcd-service:2379 \\"
    echo "    -e NIXL_DEEPEP_MAX_NUM_RANKS=8 \\"
    echo "    -e NIXL_UCX_IB_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 \\"
    echo "    -e NIXL_UCX_TCP_DEVICES=eth0 \\"
    echo "    $IMAGE_TAG \\"
    echo "    --model /path/to/model \\"
    echo "    --tensor-parallel-size 8"
    echo ""
    echo "For standard DeepEP (non-NIXL):"
    echo ""
    echo "  docker run --gpus all -p 8000:8000 \\"
    echo "    -e VLLM_ALL2ALL_BACKEND=deepep_low_latency \\"
    echo "    $IMAGE_TAG \\"
    echo "    --model /path/to/model \\"
    echo "    --tensor-parallel-size 8"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Docker build failed${NC}"
    echo -e "${RED}========================================${NC}"
    if [ "$KEEP_CONTEXT" != "true" ] && [ -d "$BUILD_CONTEXT" ]; then
        echo ""
        echo "To debug, run with --keep-context to preserve the build context"
    fi
    exit 1
fi
