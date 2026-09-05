#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Push CPU nightly images from ECR to Docker Hub as
# vllm/vllm-openai-cpu:nightly and vllm/vllm-openai-cpu:nightly-<commit>,
# combining the x86_64 and aarch64 architectures into a multi-arch manifest.
# Run when NIGHTLY=1 after the CPU release images have been pushed to ECR.
#
# Local testing (no push to Docker Hub):
#   BUILDKITE_COMMIT=<commit-with-cpu-images-in-ecr> DRY_RUN=1 bash .buildkite/scripts/push-nightly-builds-cpu.sh
# Requires: AWS CLI configured (for ECR public login), Docker. For full run: Docker Hub login.

set -ex

# Use BUILDKITE_COMMIT from env (required; set to a commit that has CPU images in ECR for local test)
BUILDKITE_COMMIT="${BUILDKITE_COMMIT:?Set BUILDKITE_COMMIT to the commit SHA that has the CPU images in ECR (e.g. from a previous release pipeline run)}"
DRY_RUN="${DRY_RUN:-0}"

TAG_NAME="nightly"
TAG_NAME_COMMIT="nightly-${BUILDKITE_COMMIT}"

# arch-dependent source images in ECR (pushed by the CPU release image build steps)
X86_ORIG_TAG="public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:${BUILDKITE_COMMIT}-x86_64"
ARM64_ORIG_TAG="public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:${BUILDKITE_COMMIT}-aarch64"

echo "Pushing CPU release images from ECR to Docker Hub as $TAG_NAME and $TAG_NAME_COMMIT"
echo "  x86_64:  $X86_ORIG_TAG"
echo "  aarch64: $ARM64_ORIG_TAG"
[[ "$DRY_RUN" == "1" ]] && echo "[DRY_RUN] Skipping push to Docker Hub"

# Login to ECR and pull the arch-dependent images
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull "$X86_ORIG_TAG"
docker pull "$ARM64_ORIG_TAG"

# Tag arch-dependent images for Docker Hub
docker tag "$X86_ORIG_TAG" vllm/vllm-openai-cpu:"$TAG_NAME"-x86_64
docker tag "$ARM64_ORIG_TAG" vllm/vllm-openai-cpu:"$TAG_NAME"-aarch64

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] Would push vllm/vllm-openai-cpu:$TAG_NAME-x86_64 and vllm/vllm-openai-cpu:$TAG_NAME-aarch64"
  echo "[DRY_RUN] Would create and push manifests vllm/vllm-openai-cpu:$TAG_NAME and vllm/vllm-openai-cpu:$TAG_NAME_COMMIT"
  echo "[DRY_RUN] Local tags created. Exiting without push."
  exit 0
fi

# Push arch-dependent images to Docker Hub
docker push vllm/vllm-openai-cpu:"$TAG_NAME"-x86_64
docker push vllm/vllm-openai-cpu:"$TAG_NAME"-aarch64

# Create and push the arch-independent manifests
docker manifest create vllm/vllm-openai-cpu:"$TAG_NAME" vllm/vllm-openai-cpu:"$TAG_NAME"-x86_64 vllm/vllm-openai-cpu:"$TAG_NAME"-aarch64 --amend
docker manifest create vllm/vllm-openai-cpu:"$TAG_NAME_COMMIT" vllm/vllm-openai-cpu:"$TAG_NAME"-x86_64 vllm/vllm-openai-cpu:"$TAG_NAME"-aarch64 --amend
docker manifest push vllm/vllm-openai-cpu:"$TAG_NAME"
docker manifest push vllm/vllm-openai-cpu:"$TAG_NAME_COMMIT"

echo "Pushed vllm/vllm-openai-cpu:$TAG_NAME and vllm/vllm-openai-cpu:$TAG_NAME_COMMIT"
