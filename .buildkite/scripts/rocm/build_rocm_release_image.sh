#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the vLLM ROCm release Docker image from the cached ROCm base image
# and push it to ECR. Reads `rocm-base-image-tag` Buildkite meta-data and
# propagates it as `rocm-base-ecr-tag` for the downstream nightly publish step.

set -euo pipefail

# Login to ECR
aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

# Get ECR image tag from metadata (set by build-rocm-base-wheels)
ECR_IMAGE_TAG="$(buildkite-agent meta-data get rocm-base-image-tag 2>/dev/null || echo '')"
if [ -z "${ECR_IMAGE_TAG}" ]; then
  echo "ERROR: rocm-base-image-tag metadata not found"
  echo "This should have been set by the build-rocm-base-wheels job"
  exit 1
fi

echo "Pulling base Docker image from ECR: ${ECR_IMAGE_TAG}"

# Pull base Docker image from ECR
docker pull "${ECR_IMAGE_TAG}"

echo "Loaded base image: ${ECR_IMAGE_TAG}"

# Pass the base image ECR tag to downstream steps (nightly publish)
buildkite-agent meta-data set "rocm-base-ecr-tag" "${ECR_IMAGE_TAG}"

echo "========================================"
echo "Building vLLM ROCm release image with:"
echo "  BASE_IMAGE: ${ECR_IMAGE_TAG}"
echo "  BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"
echo "========================================"

# Build vLLM ROCm release image using cached base
DOCKER_BUILDKIT=1 docker build \
  --build-arg max_jobs=16 \
  --build-arg BASE_IMAGE="${ECR_IMAGE_TAG}" \
  --build-arg USE_SCCACHE=1 \
  --build-arg SCCACHE_BUCKET_NAME=vllm-build-sccache \
  --build-arg SCCACHE_REGION_NAME=us-west-2 \
  --build-arg SCCACHE_S3_NO_CREDENTIALS=0 \
  --tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm \
  --target vllm-openai \
  --progress plain \
  -f docker/Dockerfile.rocm .

# Push to ECR
docker push public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm

echo ""
echo " Successfully built and pushed ROCm release image"
echo "   Image: public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm"
echo ""
