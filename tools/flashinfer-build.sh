#!/usr/bin/env bash

set -ex

# Build FlashInfer with AOT kernels
# This script is used by both the Dockerfile and standalone wheel building

# FlashInfer configuration - keep FLASHINFER_GIT_REF in sync with requirements/cuda.txt
FLASHINFER_GIT_REPO="https://github.com/flashinfer-ai/flashinfer.git"
FLASHINFER_GIT_REF="${FLASHINFER_GIT_REF:-v0.3.1}"  # Must match requirements/cuda.txt
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
BUILD_WHEEL="${BUILD_WHEEL:-true}"

echo "🏗️  Building FlashInfer ${FLASHINFER_GIT_REF} for CUDA ${CUDA_VERSION}"

# Clone FlashInfer
git clone --depth 1 --recursive --shallow-submodules \
    --branch ${FLASHINFER_GIT_REF} \
    ${FLASHINFER_GIT_REPO} flashinfer

# Set CUDA arch list based on CUDA version
# Exclude CUDA arches for older versions (11.x and 12.0-12.7)
if [[ "${CUDA_VERSION}" == 11.* ]]; then
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9"
elif [[ "${CUDA_VERSION}" == 12.[0-7]* ]]; then
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a"
else
    # CUDA 12.8+ supports 10.0a and 12.0
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 12.0"
fi

echo "🏗️ Building FlashInfer AOT for arches: ${FI_TORCH_CUDA_ARCH_LIST}"

pushd flashinfer
    # Make sure the wheel is built for the correct CUDA version
    export UV_TORCH_BACKEND=cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

    # Build AOT kernels
    export FLASHINFER_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}"
    TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
        python3 -m flashinfer.aot
    
    if [[ "${BUILD_WHEEL}" == "true" ]]; then
        # Build wheel for distribution
        TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
            uv build --wheel --out-dir ../flashinfer-dist .
        # mkdir -p /output && cp /wheels/*.whl /output/
        echo "✅ FlashInfer wheel built successfully in flashinfer-dist/"
    else
        # Install directly (for Dockerfile)
        TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
            uv pip install --system --no-build-isolation --force-reinstall .
        echo "✅ FlashInfer installed successfully"
    fi
popd

# Cleanup
rm -rf flashinfer