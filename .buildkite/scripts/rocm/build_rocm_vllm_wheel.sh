#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the vLLM ROCm wheel against the cached ROCm base image.
# Reads `rocm-base-image-tag` Buildkite meta-data set by
# build_rocm_base_image_and_wheels.sh and writes the final wheel to
# artifacts/rocm-vllm-wheel/.

set -euo pipefail

# Ensure git tags are up-to-date (Buildkite's default fetch doesn't update tags)
# This fixes version detection when tags are moved/force-pushed
echo "Fetching latest tags from origin..."
git fetch --tags --force origin

# Log tag information for debugging version detection
echo "========================================"
echo "Git Tag Verification"
echo "========================================"
echo "Current HEAD: $(git rev-parse HEAD)"
echo "git describe --tags: $(git describe --tags 2>/dev/null || echo 'No tags found')"
echo ""
echo "Recent tags (pointing to commits near HEAD):"
git tag -l --sort=-creatordate | head -5
echo "setuptools_scm version detection:"
pip install -q setuptools_scm 2>/dev/null || true
python3 -c "import setuptools_scm; print('  Detected version:', setuptools_scm.get_version())" 2>/dev/null || echo "  (setuptools_scm not available in this environment)"
echo "========================================"

# Download wheel artifacts from current build
echo "Downloading wheel artifacts from current build"
buildkite-agent artifact download "artifacts/rocm-base-wheels/*.whl" .

# Get ECR image tag from metadata (set by build-rocm-base-wheels)
ECR_IMAGE_TAG="$(buildkite-agent meta-data get rocm-base-image-tag 2>/dev/null || echo '')"
if [ -z "${ECR_IMAGE_TAG}" ]; then
  echo "ERROR: rocm-base-image-tag metadata not found"
  echo "This should have been set by the build-rocm-base-wheels job"
  exit 1
fi

echo "Pulling base Docker image from ECR: ${ECR_IMAGE_TAG}"

# Login to ECR
aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

# Pull base Docker image from ECR
docker pull "${ECR_IMAGE_TAG}"

echo "Loaded base image: ${ECR_IMAGE_TAG}"

# Prepare base wheels for Docker build context
mkdir -p docker/context/base-wheels
touch docker/context/base-wheels/.keep
cp artifacts/rocm-base-wheels/*.whl docker/context/base-wheels/
echo "Base wheels for vLLM build:"
ls -lh docker/context/base-wheels/

echo "========================================"
echo "Building vLLM wheel with:"
echo "  BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"
echo "  BUILDKITE_BRANCH: ${BUILDKITE_BRANCH}"
echo "  BASE_IMAGE: ${ECR_IMAGE_TAG}"
echo "========================================"

# Build vLLM wheel using local checkout (REMOTE_VLLM=0)
DOCKER_BUILDKIT=1 docker build \
  --file docker/Dockerfile.rocm \
  --target export_vllm_wheel_release \
  --output type=local,dest=rocm-dist \
  --build-arg BASE_IMAGE="${ECR_IMAGE_TAG}" \
  --build-arg REMOTE_VLLM=0 \
  --build-arg GIT_REPO_CHECK=1 \
  --build-arg USE_SCCACHE=1 \
  --build-arg SCCACHE_BUCKET_NAME=vllm-build-sccache \
  --build-arg SCCACHE_REGION_NAME=us-west-2 \
  --build-arg SCCACHE_S3_NO_CREDENTIALS=0 \
  .
echo "Built vLLM wheel:"
ls -lh rocm-dist/*.whl
# Copy wheel to artifacts directory
mkdir -p artifacts/rocm-vllm-wheel
cp rocm-dist/*.whl artifacts/rocm-vllm-wheel/
echo "Final vLLM wheel:"
ls -lh artifacts/rocm-vllm-wheel/
