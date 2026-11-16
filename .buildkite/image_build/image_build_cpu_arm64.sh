#!/bin/bash
set -e

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3

# When TORCH_NIGHTLY=1, build against torch nightly and tag it with the
# -torch-nightly-arm64-cpu suffix the Arm CPU test steps pull on the nightly
# lane (ci-infra get_image appends -torch-nightly before -arm64-cpu).
PYTORCH_NIGHTLY_ARGS=()
if [[ "${TORCH_NIGHTLY:-0}" == "1" ]]; then
  IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-torch-nightly-arm64-cpu"
  PYTORCH_NIGHTLY_ARGS=(--build-arg PYTORCH_NIGHTLY=1)
else
  IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-arm64-cpu"
fi

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true

# skip build if image already exists
if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image found"
else
  echo "Image not found, proceeding with build..."
  # build
  docker build --file docker/Dockerfile.cpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    "${PYTORCH_NIGHTLY_ARGS[@]}" \
    --tag "$IMAGE" \
    --target vllm-test \
    --progress plain .
  # push
  docker push "$IMAGE"
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
