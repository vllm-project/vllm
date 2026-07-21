#!/bin/bash
set -e

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3
IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-hpu"

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true

# skip build if image already exists
if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image found"
else
  echo "Image not found, proceeding with build..."
  # build
  docker build \
    --file tests/pytorch_ci_hud_benchmark/Dockerfile.hpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    --tag "$IMAGE" \
    --progress plain \
    https://github.com/vllm-project/vllm-gaudi.git
  # push
  docker push "$IMAGE"
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
