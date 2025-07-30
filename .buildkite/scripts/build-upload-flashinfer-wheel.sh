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

# Upload the wheel to S3
echo "Uploading FlashInfer wheel to S3..."
wheel_files=(artifacts/dist/*.whl)

# Check that exactly one wheel is found
if [[ ${#wheel_files[@]} -ne 1 ]]; then
  echo "Error: Expected exactly one wheel file in artifacts/dist/, but found ${#wheel_files[@]}"
  exit 1
fi

# Get the single wheel file
wheel="${wheel_files[0]}"
echo "Processing FlashInfer wheel: $wheel"

# Rename 'linux' to 'manylinux1' in the wheel filename for compatibility
new_wheel="${wheel/linux/manylinux1}"
if [[ "$wheel" != "$new_wheel" ]]; then
  mv -- "$wheel" "$new_wheel"
  wheel="$new_wheel"
  echo "Renamed wheel to: $wheel"
fi

# Extract the version from the wheel
version=$(unzip -p "$wheel" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
wheel_name=$(basename "$wheel")
echo "FlashInfer version: $version"

# Upload the wheel to S3 under flashinfer directory
aws s3 cp "$wheel" "s3://vllm-wheels/flashinfer/"

echo "✅ FlashInfer wheel built and uploaded successfully for CUDA ${CUDA_VERSION}"
echo "📦 Wheel: $wheel_name (version $version)"
ls -la artifacts/dist/