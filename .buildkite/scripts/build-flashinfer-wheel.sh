#!/usr/bin/env bash

set -ex

CUDA_VERSION="${1:-12.8.1}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-v0.2.9rc2}"

echo "Building FlashInfer wheel for CUDA ${CUDA_VERSION} using vLLM Dockerfile"

# Build the FlashInfer wheel using the existing Dockerfile stage
DOCKER_BUILDKIT=1 docker build \
  --build-arg max_jobs=16 \
  --build-arg USE_SCCACHE=1 \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg FLASHINFER_GIT_REF="${FLASHINFER_VERSION}" \
  --tag flashinfer-wheel-builder:${CUDA_VERSION} \
  --target flashinfer-wheel-builder \
  --progress plain \
  -f docker/Dockerfile .

# Extract the wheel
mkdir -p artifacts/dist
docker run --rm -v $(pwd)/artifacts:/output_host flashinfer-wheel-builder:${CUDA_VERSION} \
  bash -c 'cp /output/*.whl /output_host/dist/ && chmod -R a+rw /output_host'

# Upload the wheel
bash .buildkite/scripts/upload-flashinfer-wheels.sh

echo "FlashInfer wheel built and uploaded successfully for CUDA ${CUDA_VERSION}"
ls -la artifacts/dist/