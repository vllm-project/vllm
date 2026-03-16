#!/bin/bash
set -euo pipefail

# Build a vLLM test image with PyTorch nightly installed.
# Called by the pipeline generator's "vLLM Against PyTorch Nightly" group.

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <registry> <repo> <commit> <branch> <image_tag>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3
BRANCH=$4
IMAGE_TAG=$5

# --- ECR login ---
echo "--- :key: ECR login"
aws ecr-public get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "$REGISTRY"
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com

# --- Set up buildx ---
echo "--- :docker: Setting up buildx"
docker buildx create --name vllm-builder --driver docker-container --use || true
docker buildx inspect --bootstrap
docker buildx ls

# --- Skip if image already exists ---
echo "--- :mag: Checking if image already exists"
if docker manifest inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Image found: $IMAGE_TAG — skipping build"
  exit 0
fi
echo "Image not found, proceeding with build..."

# --- Build ---
echo "--- :docker: Building torch nightly image"
docker buildx build --file docker/Dockerfile \
  --build-arg max_jobs=16 \
  --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
  --build-arg USE_SCCACHE=1 \
  --build-arg PYTORCH_NIGHTLY=1 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0" \
  --build-arg FI_TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0a 10.0a 12.0a" \
  --tag "$IMAGE_TAG" \
  --push \
  --target test \
  --progress plain .

echo "--- :white_check_mark: Torch nightly image build complete: $IMAGE_TAG"
