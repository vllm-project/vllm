#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the AMD Zen CPU image (vLLM + zentorch) as a two-step layered build:
#   1. a CPU base image (vLLM already installed) that Dockerfile.zen layers on
#   2. docker/Dockerfile.zen --target vllm-zen-test -> zen image on top
#
# For step 1 we prefer to reuse the shared CPU image that the `image-build-cpu`
# step already builds and pushes for this commit (`…:<commit>-cpu`, built with
# --target vllm-test). That image has vLLM fully installed in the same venv, so
# it works as the base for the zen layer. If it isn't available (e.g. the zen
# step runs before/without the CPU image build), we fall back to building our
# own base from docker/Dockerfile.cpu.
#
# See docker/Dockerfile.zen for the build workflow this mirrors.
set -e

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3

# The shared CPU image built by image_build_cpu.sh (preferred base), our private
# fallback base (only built if the shared one is missing), and the final zen image.
SHARED_CPU_IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu"
FALLBACK_BASE_IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu-base-for-zen"
IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-zen-cpu"

# ZENTORCH_VERSION is optional; when unset the Dockerfile falls back to
# installing zentorch via `vllm[zen]`.
ZENTORCH_VERSION=${ZENTORCH_VERSION:-}

# authenticate with AWS ECR
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true

# skip build if image already exists
if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image found"
else
  echo "Image not found, proceeding with build..."

  # Step 1: obtain the CPU base image that Dockerfile.zen layers on. Prefer the
  # shared `-cpu` image built by the image-build-cpu step; only build our own
  # fallback base if it isn't present.
  if docker manifest inspect "$SHARED_CPU_IMAGE" >/dev/null 2>&1; then
    echo "Reusing shared CPU image as base: $SHARED_CPU_IMAGE"
    docker pull "$SHARED_CPU_IMAGE"
    BASE_IMAGE="$SHARED_CPU_IMAGE"
  else
    echo "Shared CPU image not found, building fallback base: $FALLBACK_BASE_IMAGE"
    docker build --file docker/Dockerfile.cpu \
      --platform linux/amd64 \
      --build-arg max_jobs=16 \
      --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
      --build-arg VLLM_CPU_X86=true \
      --tag "$FALLBACK_BASE_IMAGE" \
      --target vllm-openai \
      --progress plain .
    docker push "$FALLBACK_BASE_IMAGE"
    BASE_IMAGE="$FALLBACK_BASE_IMAGE"
  fi

  # Step 2: build the zen test image on top of the CPU base.
  docker build --file docker/Dockerfile.zen \
    --platform linux/amd64 \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    ${ZENTORCH_VERSION:+--build-arg ZENTORCH_VERSION="$ZENTORCH_VERSION"} \
    --tag "$IMAGE" \
    --target vllm-zen-test \
    --progress plain .

  # push
  docker push "$IMAGE"
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
