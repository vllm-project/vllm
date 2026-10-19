#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Push a ROCm stack's nightly base image and nightly image from ECR to Docker Hub
# under vllm/vllm-openai-rocm, as base-nightly-<variant>, nightly-<variant> and their
# -<commit> forms, then prune old nightly tags. With --default, also push the plain
# base-nightly and nightly tags for the same images.
# Run when NIGHTLY=1 after the stack's release image has been pushed to ECR.
#
# Usage: push-nightly-builds-rocm.sh <base-dockerfile> [--default]
#   e.g. "docker/Dockerfile.rocm_base --default" (ROCm 10.0, also :nightly)
#   and "docker/Dockerfile.rocm_72_base" (ROCm 7.2, :nightly-rocm72).
#
# Local testing (no push to Docker Hub):
#   BUILDKITE_COMMIT=<commit-with-image-in-ecr> DRY_RUN=1 \
#     bash .buildkite/scripts/push-nightly-builds-rocm.sh docker/Dockerfile.rocm_base --default
# Requires: AWS CLI configured (for ECR public login), Docker. For full run: Docker Hub login.

set -euxo pipefail

usage() {
  echo "Usage: $0 <base-dockerfile> [--default]" >&2
}

PUSH_DEFAULT=0
case "$#" in
  1) ;;
  2)
    if [[ "$2" != "--default" ]]; then
      usage
      exit 2
    fi
    PUSH_DEFAULT=1
    ;;
  *)
    usage
    exit 2
    ;;
esac

# Use BUILDKITE_COMMIT from env (required; set to a commit that has the image in ECR for local test)
BUILDKITE_COMMIT="${BUILDKITE_COMMIT:?Set BUILDKITE_COMMIT to the commit SHA that has the image in ECR (e.g. from a previous release pipeline run)}"
DRY_RUN="${DRY_RUN:-0}"

# shellcheck source=.buildkite/scripts/rocm/stack.sh
source "$(dirname "${BASH_SOURCE[0]}")/rocm/stack.sh" "$1"
DOCKERHUB_REPO="vllm/vllm-openai-rocm"
VARIANT="$ROCM_STACK_VARIANT"

BASE_TAGS=("base-nightly-${VARIANT}" "base-nightly-${VARIANT}-${BUILDKITE_COMMIT}")
TAGS=("nightly-${VARIANT}" "nightly-${VARIANT}-${BUILDKITE_COMMIT}")
if [[ "$PUSH_DEFAULT" == "1" ]]; then
  BASE_TAGS+=("base-nightly" "base-nightly-${BUILDKITE_COMMIT}")
  TAGS+=("nightly" "nightly-${BUILDKITE_COMMIT}")
fi

echo "Pushing base image $ROCM_STACK_ECR_BASE as: ${BASE_TAGS[*]}"
echo "Pushing release image $ROCM_STACK_ECR_IMAGE as: ${TAGS[*]}"

# Login to ECR and pull the stack's images
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull "$ROCM_STACK_ECR_BASE"
docker pull "$ROCM_STACK_ECR_IMAGE"

for tag in "${BASE_TAGS[@]}"; do docker tag "$ROCM_STACK_ECR_BASE" "$DOCKERHUB_REPO:$tag"; done
for tag in "${TAGS[@]}"; do docker tag "$ROCM_STACK_ECR_IMAGE" "$DOCKERHUB_REPO:$tag"; done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] Local tags created. Exiting without push."
  exit 0
fi

# Push to Docker Hub (docker-login plugin runs before this step in CI)
for tag in "${BASE_TAGS[@]}" "${TAGS[@]}"; do docker push "$DOCKERHUB_REPO:$tag"; done
echo "Pushed $DOCKERHUB_REPO: ${BASE_TAGS[*]} ${TAGS[*]}"

# Keep only the last 14 builds of each tag family. The plain nightly- families
# exclude the variant ones sharing their prefix; commit SHAs never start with "rocm".
CLEANUP=.buildkite/scripts/cleanup-nightly-builds.sh
bash "$CLEANUP" "nightly-${VARIANT}-" "$DOCKERHUB_REPO"
bash "$CLEANUP" "base-nightly-${VARIANT}-" "$DOCKERHUB_REPO"
if [[ "$PUSH_DEFAULT" == "1" ]]; then
  bash "$CLEANUP" "nightly-" "$DOCKERHUB_REPO" "nightly-rocm"
  bash "$CLEANUP" "base-nightly-" "$DOCKERHUB_REPO" "base-nightly-rocm"
fi
