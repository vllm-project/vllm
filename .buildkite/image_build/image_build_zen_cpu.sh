#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the AMD Zen CPU image (vLLM + zentorch) by layering docker/Dockerfile.zen
# on top of a CPU base image that already has vLLM installed.
#
# To avoid recompiling vLLM from source, prefer reusing the shared CPU image that
# the image-build-cpu step already builds and publishes for this commit
# (`<repo>:<commit>-cpu`). It lives in public ECR, which allows anonymous pulls,
# so no registry credentials are required. If the pull fails (e.g. the image
# isn't published yet), fall back to building the base from docker/Dockerfile.cpu.
#
# When a registry is provided, the built zen image is pushed (Step 3) so test
# nodes on other queues (e.g. zen5) can pull it instead of rebuilding. Pulls are
# anonymous; the push needs AWS credentials on the build agent.
#
# See docker/Dockerfile.zen for the build workflow this mirrors.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3

image_repo() {
  if [[ -n "${REGISTRY}" ]]; then
    printf '%s/%s' "${REGISTRY}" "${REPO}"
  else
    printf '%s' "${REPO}"
  fi
}

IMAGE_REPO="$(image_repo)"

# The shared CPU image published by image-build-cpu (public ECR, anonymous pull).
SHARED_CPU_IMAGE=""
if [[ -n "${REGISTRY}" ]]; then
  SHARED_CPU_IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu"
fi
FALLBACK_BASE_IMAGE="zen-cpu-base:$BUILDKITE_COMMIT"
IMAGE="$IMAGE_REPO:$BUILDKITE_COMMIT-zen-cpu"

# ZENTORCH_VERSION is optional; when unset the Dockerfile falls back to
# installing zentorch via `vllm[zen]`.
ZENTORCH_VERSION=${ZENTORCH_VERSION:-}

# Step 1: obtain the CPU base image that Dockerfile.zen layers on. Prefer pulling
# the published `-cpu` image; fall back to building it from source if unavailable.
if [[ -n "${SHARED_CPU_IMAGE}" ]] && docker pull "$SHARED_CPU_IMAGE"; then
  echo "--- :docker: Using published CPU image as base: $SHARED_CPU_IMAGE"
  BASE_IMAGE="$SHARED_CPU_IMAGE"
else
  echo "--- :docker: Published CPU image unavailable; building base from source"
  docker build --file docker/Dockerfile.cpu \
    --platform linux/amd64 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    --build-arg VLLM_CPU_X86=true \
    --tag "$FALLBACK_BASE_IMAGE" \
    --target vllm-openai \
    --progress plain .
  BASE_IMAGE="$FALLBACK_BASE_IMAGE"
fi

# Step 2: build the zen test image on top of the CPU base.
echo "--- :docker: Building Zen test image"
docker build --file docker/Dockerfile.zen \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  ${ZENTORCH_VERSION:+--build-arg ZENTORCH_VERSION="$ZENTORCH_VERSION"} \
  --tag "$IMAGE" \
  --target vllm-zen-test \
  --progress plain .

# Step 3: push the zen image so test nodes can pull it instead of rebuilding.
# Only when a registry is provided (CI); local runs keep the image local.
# Pushing to public ECR requires AWS credentials on the build agent; pulls are
# anonymous.
if [[ -n "${REGISTRY}" ]]; then
  echo "--- :docker: Pushing Zen image: $IMAGE"
  aws ecr-public get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin "$REGISTRY"
  docker push "$IMAGE"
fi
