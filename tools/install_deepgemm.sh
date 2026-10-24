#!/bin/bash
# Script to build and/or install DeepGEMM from source
# Default: build and install immediately
# Optional: build wheels to a directory for later installation (useful in multi-stage builds)
set -e

# Default values
# Keep DEEPGEMM_GIT_REF in sync with cmake/external_projects/deepgemm.cmake
# Inferact fork of the vLLM fork's dev branch: sm120 and sm90 paged-MQA ports,
# SwiGLU alpha/beta and SiTU Mega MoE activations, the CUDA 12.x layout header
# fix, plus Muse Spark BF16/NVFP4 MegaMoE with fused post-expert normalization.
DEEPGEMM_GIT_REPO="https://github.com/vllm-project/DeepGEMM.git"
DEEPGEMM_GIT_REF="c8aaf3bc7053db6f7d7bf762f63f3b91edd5c7bb"
PYTHON_EXECUTABLE="${VIRTUAL_ENV:-.venv}/bin/python"
WHEEL_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo|--python)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: $1 requires an argument." >&2
                exit 1
            fi
            if [[ "$1" == "--repo" ]]; then
                DEEPGEMM_GIT_REPO="$2"
            else
                PYTHON_EXECUTABLE="$2"
            fi
            shift 2
            ;;
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
            echo "  --repo URL         Git repository (default: $DEEPGEMM_GIT_REPO)"
            echo "  --ref REF          Git reference to checkout (default: $DEEPGEMM_GIT_REF)"
            echo "  --python PATH      Target Python (default: $PYTHON_EXECUTABLE)"
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

# Resolve paths before entering the temporary checkout. Build with the same
# Python and torch installation that will load the resulting extension.
PYTHON_EXECUTABLE=$("$PYTHON_EXECUTABLE" -c 'import sys; print(sys.executable)')
if [ -n "$WHEEL_DIR" ]; then
    mkdir -p "$WHEEL_DIR"
    WHEEL_DIR=$(cd "$WHEEL_DIR" && pwd)
elif ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to install into the target Python environment." >&2
    exit 1
fi

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
CUDA_MINOR="${CUDA_VERSION#"${CUDA_MAJOR}".}"
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
git clone --no-checkout "$DEEPGEMM_GIT_REPO" "$INSTALL_DIR/deepgemm"
pushd "$INSTALL_DIR/deepgemm"

# Checkout the specific reference
git checkout --detach "$DEEPGEMM_GIT_REF"
git submodule sync --recursive
git submodule update --init --recursive --depth 1

# Clean previous build artifacts
# (Based on https://github.com/deepseek-ai/DeepGEMM/blob/main/install.sh)
rm -rf -- build dist *.egg-info 2>/dev/null || true

# Build wheel
echo "🏗️  Building DeepGEMM wheel..."
DG_FORCE_BUILD=1 "$PYTHON_EXECUTABLE" setup.py bdist_wheel

# If --wheel-dir was specified, copy wheels there and exit
if [ -n "$WHEEL_DIR" ]; then
    mkdir -p "$WHEEL_DIR"
    cp dist/*.whl "$WHEEL_DIR"/
    echo "✅ Wheel built and copied to $WHEEL_DIR"
    popd
    exit 0
fi

# Keep the target environment's torch/CUDA dependencies unchanged.
uv pip install --python "$PYTHON_EXECUTABLE" --no-deps --reinstall dist/*.whl

popd
echo "✅ DeepGEMM installation completed successfully"
