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
# Regular arm64 CI stays on CUDA 13.0 -- the cu130 *release* index still carries
# wheels. Only the nightly lane has to move, because 13.0 was dropped from
# PyTorch's binary build matrix on 2026-09-28 (pytorch/pytorch#198913) and the
# cu130 *nightly* index stopped publishing after 2.15.0.dev20260928.
ARM64_BUILD_BASE_IMAGE="pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a7d9ccfc25bbc5cf17b3fcea12964a042"
PYTORCH_NIGHTLY_ARGS=()
if [[ "${TORCH_NIGHTLY:-0}" == "1" ]]; then
  IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-torch-nightly-arm64"
  # CUDA_VERSION has to be passed explicitly: docker/Dockerfile derives the
  # wheel index from it (13.2 -> cu132), and without it this lane inherits
  # ARG CUDA_VERSION=13.0.3 and keeps pulling the frozen cu130 nightly index.
  # 13.2.1 rather than 13.2.2 because FINAL_BASE_IMAGE resolves to
  # nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04 and no 13.2.2 tag is published.
  PYTORCH_NIGHTLY_ARGS=(--build-arg PYTORCH_NIGHTLY=1 --build-arg CUDA_VERSION=13.2.1)
  # Unpinned on purpose -- see image_build_torch_nightly.sh. The non-nightly
  # pin above is left alone.
  ARM64_BUILD_BASE_IMAGE="pytorch/manylinuxaarch64-builder:cuda13.2"
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
    --build-arg BUILD_BASE_IMAGE="$ARM64_BUILD_BASE_IMAGE" \
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
