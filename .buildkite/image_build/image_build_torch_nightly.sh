#!/bin/bash
set -euo pipefail

# Build vLLM image with PyTorch Nightly
# This script builds the vLLM test image using PyTorch nightly wheels

print_usage_and_exit() {
    echo "Usage: $0 <registry> <repo> <commit>"
    exit 1
}

if [[ $# -lt 3 ]]; then
    print_usage_and_exit
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3

IMAGE_TAG="${REGISTRY}/${REPO}:${BUILDKITE_COMMIT}-torch-nightly"

echo "--- :docker: Building PyTorch Nightly image"
echo "IMAGE_TAG: ${IMAGE_TAG}"

# ECR login
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${REGISTRY}"
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com

# Setup buildx
docker buildx create --name vllm-builder --driver docker-container --use || docker buildx use vllm-builder
docker buildx inspect --bootstrap
docker buildx ls

# Check if image already exists
echo "--- :mag: Checking if image exists"
if docker manifest inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Image already exists: ${IMAGE_TAG}"
    echo "Skipping build"
    exit 0
fi
echo "Image not found, proceeding with build..."

# Build the image with PyTorch nightly
echo "--- :docker: Building image with PyTorch Nightly"
docker buildx build --file docker/Dockerfile \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="${BUILDKITE_COMMIT}" \
    --build-arg USE_SCCACHE=1 \
    --build-arg PYTORCH_NIGHTLY=1 \
    --build-arg TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0" \
    --build-arg FI_TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0a 10.0a" \
    --tag "${IMAGE_TAG}" \
    --push \
    --target test \
    --progress plain .

echo "--- :white_check_mark: PyTorch Nightly image build complete"
echo "Image: ${IMAGE_TAG}"
