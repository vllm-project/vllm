#!/bin/bash
# Script to build and/or install DeepGEMM from source
# Default: build and install immediately
# Optional: build wheels to a directory for later installation (useful in multi-stage builds)
set -e

# Default values
DEEPGEMM_GIT_REPO="https://github.com/deepseek-ai/DeepGEMM.git"
DEEPGEMM_GIT_REF="477618cd51baffca09c4b0b87e97c03fe827ef03"
WHEEL_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ref)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --ref requires an argument." >&2
                exit 1
            fi
            DEEPGEMM_GIT_REF="$2"
            shift 2
            ;;
        --cuda-version)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --cuda-version requires an argument." >&2
                exit 1
            fi
            CUDA_VERSION="$2"
            shift 2
            ;;
        --wheel-dir)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --wheel-dir requires a directory path." >&2
                exit 1
            fi
            WHEEL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --ref REF          Git reference to checkout (default: $DEEPGEMM_GIT_REF)"
            echo "  --cuda-version VER CUDA version (auto-detected if not provided)"
            echo "  --wheel-dir PATH   If set, build wheel into PATH but do not install"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Auto-detect CUDA version if not provided
if [ -z "$CUDA_VERSION" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        echo "Auto-detected CUDA version: $CUDA_VERSION"
    else
        echo "Warning: Could not auto-detect CUDA version. Please specify with --cuda-version"
        exit 1
    fi
fi

# Extract major and minor version numbers
CUDA_MAJOR="${CUDA_VERSION%%.*}"
CUDA_MINOR="${CUDA_VERSION#${CUDA_MAJOR}.}"
CUDA_MINOR="${CUDA_MINOR%%.*}"
echo "CUDA version: $CUDA_VERSION (major: $CUDA_MAJOR, minor: $CUDA_MINOR)"

# Check CUDA version requirement
if [ "$CUDA_MAJOR" -lt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 8 ]; }; then
    echo "Skipping DeepGEMM build/installation (requires CUDA 12.8+ but got ${CUDA_VERSION})"
    exit 0
fi

echo "Preparing DeepGEMM build..."
echo "Repository: $DEEPGEMM_GIT_REPO"
echo "Reference: $DEEPGEMM_GIT_REF"

# Create a temporary directory for the build
INSTALL_DIR=$(mktemp -d)
trap 'rm -rf "$INSTALL_DIR"' EXIT

# Clone the repository
git clone --recursive --shallow-submodules "$DEEPGEMM_GIT_REPO" "$INSTALL_DIR/deepgemm"
pushd "$INSTALL_DIR/deepgemm"

# Checkout the specific reference
git checkout "$DEEPGEMM_GIT_REF"

# Clean previous build artifacts
# (Based on https://github.com/deepseek-ai/DeepGEMM/blob/main/install.sh)
rm -rf build dist *.egg-info

# Build wheel
echo "🏗️  Building DeepGEMM wheel..."
python3 setup.py bdist_wheel

# If --wheel-dir was specified, copy wheels there and exit
if [ -n "$WHEEL_DIR" ]; then
    mkdir -p "$WHEEL_DIR"
    cp dist/*.whl "$WHEEL_DIR"/
    echo "✅ Wheel built and copied to $WHEEL_DIR"
    popd
    exit 0
fi

# Default behaviour: install built wheel
if command -v uv >/dev/null 2>&1; then
    echo "Installing DeepGEMM wheel using uv..."
    if [ -n "$VLLM_DOCKER_BUILD_CONTEXT" ]; then
        uv pip install --system dist/*.whl
    else
        uv pip install dist/*.whl
    fi
else
    echo "Installing DeepGEMM wheel using pip..."
    python3 -m pip install dist/*.whl
fi

popd
echo "✅ DeepGEMM installation completed successfully"
