#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Push a ROCm-family nightly base image and nightly image from ECR to Docker Hub
# under vllm/vllm-openai-rocm, as base-nightly[-<variant>], nightly[-<variant>]
# and their -<commit> forms.
# Run when NIGHTLY=1 after the matching build-*-release-image step has pushed to ECR.
#
# Usage: push-nightly-builds-rocm.sh [ECR_TAG_SUFFIX TAG_VARIANT]
#   Pass both arguments or neither. With none, the default ROCm apt build is
#   published as :nightly. The TheRock build passes "rock" (the source ECR tag
#   suffix) and "rocm714" (the Docker Hub tag flavor), giving :nightly-rocm714.
#
# Local testing (no push to Docker Hub):
#   BASE_ECR_IMAGE=<full-base-image-reference> \
#     BUILDKITE_COMMIT=<commit-with-image-in-ecr> DRY_RUN=1 \
#     bash .buildkite/scripts/push-nightly-builds-rocm.sh
# Requires: AWS CLI configured (for ECR public login), Docker. For full run: Docker Hub login.

set -euxo pipefail

usage() {
  echo "Usage: $0 [ECR_TAG_SUFFIX TAG_VARIANT]" >&2
}

DOCKERHUB_REPO="vllm/vllm-openai-rocm"

case "$#" in
  0)
    ECR_TAG_SUFFIX="rocm"
    TAG_VARIANT=""
    ;;
  2)
    if [[ -z "$1" || -z "$2" ]]; then
      usage
      exit 2
    fi
    ECR_TAG_SUFFIX="$1"
    TAG_VARIANT="$2"
    ;;
  *)
    usage
    exit 2
    ;;
esac

# Use BUILDKITE_COMMIT from env (required; set to a commit that has the image in ECR for local test)
BUILDKITE_COMMIT="${BUILDKITE_COMMIT:?Set BUILDKITE_COMMIT to the commit SHA that has the image in ECR (e.g. from a previous release pipeline run)}"
DRY_RUN="${DRY_RUN:-0}"

ECR_REPO="public.ecr.aws/q9t5s3a7/vllm-release-repo"

# Get the base image ECR tag (set by the build-*-release-image pipeline step)
BASE_METADATA_KEY="${ECR_TAG_SUFFIX}-base-ecr-tag"
if [[ -n "${BASE_ECR_IMAGE:-}" ]]; then
  if [[ "$DRY_RUN" != "1" ]]; then
    echo "ERROR: BASE_ECR_IMAGE may only be used with DRY_RUN=1" >&2
    exit 1
  fi
  BASE_ORIG_TAG="$BASE_ECR_IMAGE"
elif ! BASE_ORIG_TAG="$(buildkite-agent meta-data get "$BASE_METADATA_KEY")"; then
  echo "ERROR: Failed to read required Buildkite metadata '$BASE_METADATA_KEY'" >&2
  exit 1
fi
if [[ -z "$BASE_ORIG_TAG" ]]; then
  echo "ERROR: Required Buildkite metadata '$BASE_METADATA_KEY' is empty" >&2
  echo "Set BASE_ECR_IMAGE to the full ECR base image reference for local testing" >&2
  exit 1
fi

ORIG_TAG="${BUILDKITE_COMMIT}-${ECR_TAG_SUFFIX}"
VARIANT_SUFFIX=""
[[ -n "$TAG_VARIANT" ]] && VARIANT_SUFFIX="-$TAG_VARIANT"
BASE_TAG_NAME="base-nightly${VARIANT_SUFFIX}"
TAG_NAME="nightly${VARIANT_SUFFIX}"
BASE_TAG_NAME_COMMIT="base-nightly${VARIANT_SUFFIX}-${BUILDKITE_COMMIT}"
TAG_NAME_COMMIT="nightly${VARIANT_SUFFIX}-${BUILDKITE_COMMIT}"

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
