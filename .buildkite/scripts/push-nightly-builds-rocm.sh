#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Push a ROCm-family nightly base image and nightly image from ECR to Docker Hub
# as <repo>:base-nightly, <repo>:base-nightly-<commit>, <repo>:nightly and
# <repo>:nightly-<commit>.
# Run when NIGHTLY=1 after the matching build-*-release-image step has pushed to ECR.
#
# Usage: push-nightly-builds-rocm.sh [DOCKERHUB_REPO] [ECR_TAG_SUFFIX]
#   Defaults target ROCm; the ROCk images pass "vllm/vllm-openai-rock" and "rock".
#
# Local testing (no push to Docker Hub):
#   BUILDKITE_COMMIT=<commit-with-rocm-image-in-ecr> DRY_RUN=1 bash .buildkite/scripts/push-nightly-builds-rocm.sh
# Requires: AWS CLI configured (for ECR public login), Docker. For full run: Docker Hub login.

set -ex

# Use BUILDKITE_COMMIT from env (required; set to a commit that has the image in ECR for local test)
BUILDKITE_COMMIT="${BUILDKITE_COMMIT:?Set BUILDKITE_COMMIT to the commit SHA that has the image in ECR (e.g. from a previous release pipeline run)}"
DRY_RUN="${DRY_RUN:-0}"

DOCKERHUB_REPO="${1:-vllm/vllm-openai-rocm}"
ECR_TAG_SUFFIX="${2:-rocm}"
ECR_REPO="public.ecr.aws/q9t5s3a7/vllm-release-repo"

# Get the base image ECR tag (set by the build-*-release-image pipeline step)
BASE_ORIG_TAG="$(buildkite-agent meta-data get "${ECR_TAG_SUFFIX}-base-ecr-tag" 2>/dev/null || echo "")"
if [ -z "$BASE_ORIG_TAG" ]; then
  echo "WARNING: ${ECR_TAG_SUFFIX}-base-ecr-tag metadata not found, falling back to commit-based tag"
  BASE_ORIG_TAG="${ECR_REPO}:${BUILDKITE_COMMIT}-${ECR_TAG_SUFFIX}-base"
fi

ORIG_TAG="${BUILDKITE_COMMIT}-${ECR_TAG_SUFFIX}"
BASE_TAG_NAME="base-nightly"
TAG_NAME="nightly"
BASE_TAG_NAME_COMMIT="base-nightly-${BUILDKITE_COMMIT}"
TAG_NAME_COMMIT="nightly-${BUILDKITE_COMMIT}"

echo "Pushing base image from ECR: $BASE_ORIG_TAG"
echo "Pushing release image from ECR tag: $ORIG_TAG to Docker Hub as $DOCKERHUB_REPO:$TAG_NAME and $DOCKERHUB_REPO:$TAG_NAME_COMMIT"
[[ "$DRY_RUN" == "1" ]] && echo "[DRY_RUN] Skipping push to Docker Hub"

# Login to ECR and pull the images built by build-*-release-image
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull "$BASE_ORIG_TAG"
docker pull "${ECR_REPO}:${ORIG_TAG}"

# Tag for Docker Hub (base-nightly and base-nightly-<commit>, nightly and nightly-<commit>)
docker tag "$BASE_ORIG_TAG" "$DOCKERHUB_REPO":"$BASE_TAG_NAME"
docker tag "$BASE_ORIG_TAG" "$DOCKERHUB_REPO":"$BASE_TAG_NAME_COMMIT"
docker tag "${ECR_REPO}:${ORIG_TAG}" "$DOCKERHUB_REPO":"$TAG_NAME"
docker tag "${ECR_REPO}:${ORIG_TAG}" "$DOCKERHUB_REPO":"$TAG_NAME_COMMIT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] Would push $DOCKERHUB_REPO:$BASE_TAG_NAME and $DOCKERHUB_REPO:$BASE_TAG_NAME_COMMIT"
  echo "[DRY_RUN] Would push $DOCKERHUB_REPO:$TAG_NAME and $DOCKERHUB_REPO:$TAG_NAME_COMMIT"
  echo "[DRY_RUN] Local tags created. Exiting without push."
  exit 0
fi

# Push to Docker Hub (docker-login plugin runs before this step in CI)
docker push "$DOCKERHUB_REPO":"$BASE_TAG_NAME"
docker push "$DOCKERHUB_REPO":"$BASE_TAG_NAME_COMMIT"
docker push "$DOCKERHUB_REPO":"$TAG_NAME"
docker push "$DOCKERHUB_REPO":"$TAG_NAME_COMMIT"

echo "Pushed $DOCKERHUB_REPO:$BASE_TAG_NAME and $DOCKERHUB_REPO:$BASE_TAG_NAME_COMMIT"
echo "Pushed $DOCKERHUB_REPO:$TAG_NAME and $DOCKERHUB_REPO:$TAG_NAME_COMMIT"
