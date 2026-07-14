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

# --- Arguments ---
echo "--- :mag: Arguments"
echo "REGISTRY: ${REGISTRY}"
echo "REPO: ${REPO}"
echo "BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"
echo "BRANCH: ${BRANCH}"
echo "IMAGE_TAG: ${IMAGE_TAG}"

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

# --- Skip if image already exists (unless forcing a rebuild) ---
# The image tag is keyed on the vLLM commit SHA only, with no PyTorch nightly
# date/version in it. Reusing a cached image for the same commit would run the
# tests against a stale nightly, so default to always rebuilding here. Set
# FORCE_REBUILD=0 to opt back into cache reuse.
FORCE_REBUILD="${FORCE_REBUILD:-1}"
if [[ "$FORCE_REBUILD" != "1" ]] && docker manifest inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Image found: $IMAGE_TAG — skipping build (FORCE_REBUILD=0)"
  .buildkite/scripts/annotate-image-build.sh "$IMAGE_TAG"
  exit 0
fi
echo "Building image (FORCE_REBUILD=$FORCE_REBUILD): $IMAGE_TAG"

# --- CUDA 13.0 for nightly builds ---
# Nightly CI uses CUDA 13.0 while regular CI stays on CUDA 12.9
NIGHTLY_CUDA_VERSION="13.0.2"
NIGHTLY_BUILD_BASE_IMAGE="nvidia/cuda:${NIGHTLY_CUDA_VERSION}-devel-ubuntu22.04"
NIGHTLY_FINAL_BASE_IMAGE="nvidia/cuda:${NIGHTLY_CUDA_VERSION}-base-ubuntu22.04"

echo "--- :docker: Building torch nightly image (CUDA ${NIGHTLY_CUDA_VERSION})"
docker buildx build --file docker/Dockerfile \
  --build-arg max_jobs=16 \
  --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
  --build-arg USE_SCCACHE=1 \
  --build-arg PYTORCH_NIGHTLY=1 \
  --build-arg CUDA_VERSION="${NIGHTLY_CUDA_VERSION}" \
  --build-arg BUILD_BASE_IMAGE="${NIGHTLY_BUILD_BASE_IMAGE}" \
  --build-arg FINAL_BASE_IMAGE="${NIGHTLY_FINAL_BASE_IMAGE}" \
  --build-arg torch_cuda_arch_list="8.0 8.9 9.0 10.0 12.0" \
  --tag "$IMAGE_TAG" \
  --push \
  --target test \
  --progress plain .

echo "--- :white_check_mark: Torch nightly image build complete: $IMAGE_TAG"

.buildkite/scripts/annotate-image-build.sh "$IMAGE_TAG"
