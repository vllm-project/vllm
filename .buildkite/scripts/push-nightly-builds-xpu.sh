#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Push XPU nightly image from ECR to Docker Hub as
# vllm/vllm-openai-xpu:nightly and vllm/vllm-openai-xpu:nightly-<commit>.
# Run when NIGHTLY=1 after build-xpu-release-image has pushed to ECR.
#
# Local testing (no push to Docker Hub):
#   BUILDKITE_COMMIT=<commit-with-xpu-image-in-ecr> DRY_RUN=1 \
#     bash .buildkite/scripts/push-nightly-builds-xpu.sh
# Requires: AWS CLI configured (for ECR public login), Docker.
# For full run: Docker Hub login.

set -ex

# Use BUILDKITE_COMMIT from env (required; set to a commit that has the
# XPU image in ECR for local test).
BUILDKITE_COMMIT="${BUILDKITE_COMMIT:?Set BUILDKITE_COMMIT to the commit SHA that has the XPU image in ECR (e.g. from a previous release pipeline run)}"
DRY_RUN="${DRY_RUN:-0}"

ORIG_TAG="${BUILDKITE_COMMIT}-xpu"
TAG_NAME="nightly"
TAG_NAME_COMMIT="nightly-${BUILDKITE_COMMIT}"

echo "Pushing XPU release image from ECR tag: $ORIG_TAG to Docker Hub as $TAG_NAME and $TAG_NAME_COMMIT"
[[ "$DRY_RUN" == "1" ]] && echo "[DRY_RUN] Skipping push to Docker Hub"

# Login to ECR and pull the image built by build-xpu-release-image.
aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:"$ORIG_TAG"

# Tag for Docker Hub (nightly and nightly-<commit>).
docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:"$ORIG_TAG" vllm/vllm-openai-xpu:"$TAG_NAME"
docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:"$ORIG_TAG" vllm/vllm-openai-xpu:"$TAG_NAME_COMMIT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] Would push vllm/vllm-openai-xpu:$TAG_NAME and vllm/vllm-openai-xpu:$TAG_NAME_COMMIT"
  echo "[DRY_RUN] Local tags created. Exiting without push."
  exit 0
fi

# Push to Docker Hub (docker-login plugin runs before this step in CI).
docker push vllm/vllm-openai-xpu:"$TAG_NAME"
docker push vllm/vllm-openai-xpu:"$TAG_NAME_COMMIT"

echo "Pushed vllm/vllm-openai-xpu:$TAG_NAME and vllm/vllm-openai-xpu:$TAG_NAME_COMMIT"
