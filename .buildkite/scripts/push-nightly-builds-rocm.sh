#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Push a ROCm-family nightly base image and nightly image from ECR to Docker Hub
# under vllm/vllm-openai-rocm, as base-nightly-<variant>, nightly-<variant> and
# their -<commit> forms. With --default, also push the plain base-nightly and
# nightly tags (and their -<commit> forms) for the same images.
# Run when NIGHTLY=1 after the matching build-*-release-image step has pushed to ECR.
#
# Usage: push-nightly-builds-rocm.sh ECR_TAG_SUFFIX TAG_VARIANT [--default]
#   ECR_TAG_SUFFIX  source ECR tag suffix, i.e. <commit>-<suffix>
#   TAG_VARIANT     Docker Hub tag flavor, e.g. rocm100 -> :nightly-rocm100
#   e.g. "rock rocm100 --default" (ROCm 10, also :nightly) and "rocm rocm72".
#
# Local testing (no push to Docker Hub):
#   BASE_ECR_IMAGE=<full-base-image-reference> \
#     BUILDKITE_COMMIT=<commit-with-image-in-ecr> DRY_RUN=1 \
#     bash .buildkite/scripts/push-nightly-builds-rocm.sh rock rocm100 --default
# Requires: AWS CLI configured (for ECR public login), Docker. For full run: Docker Hub login.

set -euxo pipefail

usage() {
  echo "Usage: $0 ECR_TAG_SUFFIX TAG_VARIANT [--default]" >&2
}

DOCKERHUB_REPO="vllm/vllm-openai-rocm"

PUSH_DEFAULT=0
case "$#" in
  2) ;;
  3)
    if [[ "$3" != "--default" ]]; then
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
if [[ -z "$1" || -z "$2" ]]; then
  usage
  exit 2
fi
ECR_TAG_SUFFIX="$1"
TAG_VARIANT="$2"

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
ORIG_TAG="${ECR_REPO}:${BUILDKITE_COMMIT}-${ECR_TAG_SUFFIX}"

BASE_TAGS=("base-nightly-${TAG_VARIANT}" "base-nightly-${TAG_VARIANT}-${BUILDKITE_COMMIT}")
TAGS=("nightly-${TAG_VARIANT}" "nightly-${TAG_VARIANT}-${BUILDKITE_COMMIT}")
if [[ "$PUSH_DEFAULT" == "1" ]]; then
  BASE_TAGS+=("base-nightly" "base-nightly-${BUILDKITE_COMMIT}")
  TAGS+=("nightly" "nightly-${BUILDKITE_COMMIT}")
fi

echo "Pushing base image $BASE_ORIG_TAG as: ${BASE_TAGS[*]}"
echo "Pushing release image $ORIG_TAG as: ${TAGS[*]}"

# Login to ECR and pull the images built by build-*-release-image
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull "$BASE_ORIG_TAG"
docker pull "$ORIG_TAG"

for tag in "${BASE_TAGS[@]}"; do docker tag "$BASE_ORIG_TAG" "$DOCKERHUB_REPO:$tag"; done
for tag in "${TAGS[@]}"; do docker tag "$ORIG_TAG" "$DOCKERHUB_REPO:$tag"; done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] Local tags created. Exiting without push."
  exit 0
fi

# Push to Docker Hub (docker-login plugin runs before this step in CI)
for tag in "${BASE_TAGS[@]}" "${TAGS[@]}"; do docker push "$DOCKERHUB_REPO:$tag"; done
echo "Pushed $DOCKERHUB_REPO: ${BASE_TAGS[*]} ${TAGS[*]}"
