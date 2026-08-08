#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the AMD Zen CPU image (vLLM + zentorch) as a two-step layered build:
#   1. a CPU base image (vLLM installed) built from docker/Dockerfile.cpu
#   2. docker/Dockerfile.zen --target vllm-zen-test -> zen image on top
#
# The image is (re)built from source every time, mirroring
# .buildkite/scripts/hardware_ci/run-cpu-test.sh. It is not pushed to a registry.
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

# Local image tags (not pushed).
BASE_IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu-base-for-zen"
IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-zen-cpu"

# ZENTORCH_VERSION is optional; when unset the Dockerfile falls back to
# installing zentorch via `vllm[zen]`.
ZENTORCH_VERSION=${ZENTORCH_VERSION:-}

# Step 1: build the CPU base image that Dockerfile.zen layers on.
echo "--- :docker: Building CPU base image"
docker build --file docker/Dockerfile.cpu \
  --platform linux/amd64 \
  --build-arg max_jobs=16 \
  --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
  --build-arg VLLM_CPU_X86=true \
  --tag "$BASE_IMAGE" \
  --target vllm-openai \
  --progress plain .

# Step 2: build the zen test image on top of the CPU base.
echo "--- :docker: Building Zen test image"
docker build --file docker/Dockerfile.zen \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  ${ZENTORCH_VERSION:+--build-arg ZENTORCH_VERSION="$ZENTORCH_VERSION"} \
  --tag "$IMAGE" \
  --target vllm-zen-test \
  --progress plain .
