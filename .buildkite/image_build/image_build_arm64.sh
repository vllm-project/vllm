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
# -torch-nightly-arm64 suffix the DGX Spark test steps pull on the nightly lane
# (ci-infra get_image(arm64=True) appends -torch-nightly before -arm64).
PYTORCH_NIGHTLY_ARGS=()
if [[ "${TORCH_NIGHTLY:-0}" == "1" ]]; then
  IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-torch-nightly-arm64"
  PYTORCH_NIGHTLY_ARGS=(--build-arg PYTORCH_NIGHTLY=1)
else
  IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-arm64"
fi

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true

# skip build if image already exists
if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image found"
else
  echo "Image not found, proceeding with build..."
  # build for arm64 GPU targets: Grace/GH200 (sm_90),
  # Blackwell/Thor (sm_100/sm_103/sm_110), and DGX Spark/GB10
  # (sm_121, family-covered by 12.0 under CUDA 13)
  docker build --file docker/Dockerfile \
    --platform linux/arm64 \
    --build-arg max_jobs=16 \
    --build-arg nvcc_threads=4 \
    --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a7d9ccfc25bbc5cf17b3fcea12964a042 \
    --build-arg torch_cuda_arch_list="9.0 10.0 11.0 12.0" \
    --build-arg USE_SCCACHE=1 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    "${PYTORCH_NIGHTLY_ARGS[@]}" \
    --tag "$IMAGE" \
    --target test \
    --progress plain .
  # push
  docker push "$IMAGE"
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
