#!/bin/bash
set -e

# Builds the Zen CPU image for vLLM tests.
#
# Two-step build: first builds the CPU `vllm-openai` image from Dockerfile.cpu,
# then layers zentorch + test deps on top via Dockerfile.zen.

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3

CPU_BASE_TAG="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu"
ZEN_TAG="$REGISTRY/$REPO:$BUILDKITE_COMMIT-zen-cpu"

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "$REGISTRY"

# skip build if image already exists
if [[ -z $(docker manifest inspect "$ZEN_TAG") ]]; then
  echo "Image not found, proceeding with build..."
else
  echo "Image found"
  exit 0
fi

# Step 1: build CPU base if not already in registry. Reuses the published
# CPU image when available so we don't pay the cost twice.
if ! docker manifest inspect "$CPU_BASE_TAG" > /dev/null 2>&1; then
  echo "--- Building CPU base image (vllm-openai target from Dockerfile.cpu)"
  docker build --file docker/Dockerfile.cpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    --build-arg VLLM_CPU_X86=true \
    --tag "$CPU_BASE_TAG" \
    --target vllm-openai \
    --progress plain .
  docker push "$CPU_BASE_TAG"
else
  echo "--- CPU base image already in registry, pulling instead of rebuilding"
  docker pull "$CPU_BASE_TAG"
fi

# Step 2: build Zen image on top of the CPU base.
echo "--- Building Zen CPU test image"
docker build --file docker/Dockerfile.zen \
  --build-arg BASE_IMAGE="$CPU_BASE_TAG" \
  --build-arg max_jobs=16 \
  --tag "$ZEN_TAG" \
  --target vllm-zen-test \
  --progress plain .

# push
docker push "$ZEN_TAG"
