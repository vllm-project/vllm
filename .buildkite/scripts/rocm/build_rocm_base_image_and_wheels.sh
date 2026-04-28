#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the ROCm base image and extract base wheels.
# Uses ECR (image) + S3 (wheels) as a cache keyed by Dockerfile.rocm_base hash.
# Sets Buildkite meta-data `rocm-base-image-tag` for downstream ROCm jobs.

set -euo pipefail

# Generate cache key
CACHE_KEY=$(.buildkite/scripts/rocm/cache_rocm_base_wheels.sh key)
ECR_CACHE_TAG="public.ecr.aws/q9t5s3a7/vllm-release-repo:${CACHE_KEY}-rocm-base"

echo "========================================"
echo "ROCm Base Build Configuration"
echo "========================================"
echo "  CACHE_KEY: ${CACHE_KEY}"
echo "  ECR_CACHE_TAG: ${ECR_CACHE_TAG}"
echo "========================================"

# Login to ECR
aws ecr-public get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

IMAGE_EXISTS=false
WHEELS_EXIST=false

# Check ECR for Docker image
if docker manifest inspect "${ECR_CACHE_TAG}" > /dev/null 2>&1; then
    IMAGE_EXISTS=true
    echo "ECR image cache HIT"
fi

# Check S3 for wheels
WHEEL_CACHE_STATUS=$(.buildkite/scripts/rocm/cache_rocm_base_wheels.sh check)
if [ "${WHEEL_CACHE_STATUS}" = "hit" ]; then
    WHEELS_EXIST=true
    echo "S3 wheels cache HIT"
fi


# Scenario 1: Both cached (best case)
if [ "${IMAGE_EXISTS}" = "true" ] && [ "${WHEELS_EXIST}" = "true" ]; then
    echo ""
    echo "FULL CACHE HIT - Reusing both image and wheels"
    echo ""

    # Download wheels
    .buildkite/scripts/rocm/cache_rocm_base_wheels.sh download

    # Save ECR tag for downstream jobs
    buildkite-agent meta-data set "rocm-base-image-tag" "${ECR_CACHE_TAG}"

# Scenario 2: Full rebuild needed
else
    echo ""
    echo " CACHE MISS - Building from scratch..."
    echo ""

    # Build full base image and push to ECR
    DOCKER_BUILDKIT=1 docker buildx build \
    --file docker/Dockerfile.rocm_base \
    --tag "${ECR_CACHE_TAG}" \
    --build-arg USE_SCCACHE=1 \
    --build-arg SCCACHE_BUCKET_NAME=vllm-build-sccache \
    --build-arg SCCACHE_REGION_NAME=us-west-2 \
    --build-arg SCCACHE_S3_NO_CREDENTIALS=0 \
    --push \
    .

    # Build wheel extraction stage
    DOCKER_BUILDKIT=1 docker buildx build \
    --file docker/Dockerfile.rocm_base \
    --tag rocm-base-debs:${BUILDKITE_BUILD_NUMBER} \
    --target debs_wheel_release \
    --build-arg USE_SCCACHE=1 \
    --build-arg SCCACHE_BUCKET_NAME=vllm-build-sccache \
    --build-arg SCCACHE_REGION_NAME=us-west-2 \
    --build-arg SCCACHE_S3_NO_CREDENTIALS=0 \
    --load \
    .

    # Extract and upload wheels
    mkdir -p artifacts/rocm-base-wheels
    cid=$(docker create rocm-base-debs:${BUILDKITE_BUILD_NUMBER})
    docker cp ${cid}:/app/debs/. artifacts/rocm-base-wheels/
    docker rm ${cid}

    .buildkite/scripts/rocm/cache_rocm_base_wheels.sh upload

    # Cache base docker image to ECR
    docker push "${ECR_CACHE_TAG}"

    buildkite-agent meta-data set "rocm-base-image-tag" "${ECR_CACHE_TAG}"

    echo ""
    echo " Build complete - Image and wheels cached"
fi
