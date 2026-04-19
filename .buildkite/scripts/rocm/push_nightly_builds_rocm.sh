#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Push ROCm nightly base image and nightly image from ECR 
# to Docker Hub as vllm/vllm-openai-rocm:base-nightly and vllm/vllm-openai-rocm:nightly
# and vllm/vllm-openai-rocm:base-nightly-<commit> and vllm/vllm-openai-rocm:nightly-<commit>.
# Run when NIGHTLY=1 after build-rocm-release-image has pushed to ECR.
#
# Local testing (no push to Docker Hub):
#   BUILDKITE_COMMIT=<commit-with-rocm-image-in-ecr> DRY_RUN=1 bash .buildkite/scripts/rocm/push_nightly_builds_rocm.sh
# Requires: AWS CLI configured (for ECR public login), Docker. For full run: Docker Hub login.

set -ex

# Use BUILDKITE_COMMIT from env (required; set to a commit that has ROCm image in ECR for local test)
BUILDKITE_COMMIT="${BUILDKITE_COMMIT:?Set BUILDKITE_COMMIT to the commit SHA that has the ROCm image in ECR (e.g. from a previous release pipeline run)}"
DRY_RUN="${DRY_RUN:-0}"

# Get the base image ECR tag (set by build-rocm-release-image pipeline step)
BASE_ORIG_TAG="$(buildkite-agent meta-data get rocm-base-ecr-tag 2>/dev/null || echo "")"
if [ -z "$BASE_ORIG_TAG" ]; then
  echo "WARNING: rocm-base-ecr-tag metadata not found, falling back to commit-based tag"
  BASE_ORIG_TAG="public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm-base"
fi

ORIG_TAG="${BUILDKITE_COMMIT}-rocm"
BASE_TAG_NAME="base-nightly"
TAG_NAME="nightly"
BASE_TAG_NAME_COMMIT="base-nightly-${BUILDKITE_COMMIT}"
TAG_NAME_COMMIT="nightly-${BUILDKITE_COMMIT}"

echo "Pushing ROCm base image from ECR: $BASE_ORIG_TAG"
echo "Pushing ROCm release image from ECR tag: $ORIG_TAG to Docker Hub as $TAG_NAME and $TAG_NAME_COMMIT"
[[ "$DRY_RUN" == "1" ]] && echo "[DRY_RUN] Skipping push to Docker Hub"

# Login to ECR and pull the image built by build-rocm-release-image
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull "$BASE_ORIG_TAG"
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:"$ORIG_TAG"

# Tag for Docker Hub (base-nightly and base-nightly-<commit>, nightly and nightly-<commit>)
docker tag "$BASE_ORIG_TAG" vllm/vllm-openai-rocm:"$BASE_TAG_NAME"
docker tag "$BASE_ORIG_TAG" vllm/vllm-openai-rocm:"$BASE_TAG_NAME_COMMIT"
docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:"$ORIG_TAG" vllm/vllm-openai-rocm:"$TAG_NAME"
docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:"$ORIG_TAG" vllm/vllm-openai-rocm:"$TAG_NAME_COMMIT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] Would push vllm/vllm-openai-rocm:$BASE_TAG_NAME and vllm/vllm-openai-rocm:$BASE_TAG_NAME_COMMIT"
  echo "[DRY_RUN] Would push vllm/vllm-openai-rocm:$TAG_NAME and vllm/vllm-openai-rocm:$TAG_NAME_COMMIT"
  echo "[DRY_RUN] Local tags created. Exiting without push."
  exit 0
fi

# Push to Docker Hub (docker-login plugin runs before this step in CI)
docker push vllm/vllm-openai-rocm:"$BASE_TAG_NAME"
docker push vllm/vllm-openai-rocm:"$BASE_TAG_NAME_COMMIT"
docker push vllm/vllm-openai-rocm:"$TAG_NAME"
docker push vllm/vllm-openai-rocm:"$TAG_NAME_COMMIT"

echo "Pushed vllm/vllm-openai-rocm:$BASE_TAG_NAME and vllm/vllm-openai-rocm:$BASE_TAG_NAME_COMMIT"
echo "Pushed vllm/vllm-openai-rocm:$TAG_NAME and vllm/vllm-openai-rocm:$TAG_NAME_COMMIT"
