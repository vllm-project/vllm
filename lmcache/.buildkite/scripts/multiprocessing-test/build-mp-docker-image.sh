#!/bin/bash
# Build the LMCache + vLLM docker image for multiprocessing tests
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== Building LMCache + vLLM image ==="

cd "$REPO_ROOT/docker"

# Build the vLLM + LMCache image (used for both lmcache server and vllm process)
docker build \
    --progress=plain \
    --build-arg CUDA_VERSION=12.8 \
    --build-arg UBUNTU_VERSION=24.04 \
    --build-arg VLLM_VERSION=nightly \
    --target image-build \
    --file Dockerfile \
    --tag lmcache/vllm-openai:test \
    ../

echo "=== Docker image built successfully ==="
docker images | grep "lmcache/vllm-openai" | head -5
