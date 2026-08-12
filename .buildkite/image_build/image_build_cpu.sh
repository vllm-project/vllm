#!/bin/bash
set -e

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3
IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu"

# replace invalid characters in Docker image tags and truncate to 128 chars
clean_docker_tag() {
    local input="$1"
    echo "$input" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-128
}

# resolve and set: CACHE_TO, CACHE_FROM, CACHE_FROM_BASE_BRANCH, CACHE_FROM_MAIN
# Reuses the same ECR cache repos as the CUDA image build, with an
# "x86_cpu" suffix on every tag so CPU and CUDA cache blobs stay isolated.
prepare_cache_tags() {
    TEST_CACHE_ECR="936637512419.dkr.ecr.us-east-1.amazonaws.com/vllm-ci-test-cache"
    MAIN_CACHE_ECR="936637512419.dkr.ecr.us-east-1.amazonaws.com/vllm-ci-postmerge-cache"

    if [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
        if [[ "${BUILDKITE_BRANCH:-}" == "main" ]]; then
            cache="${MAIN_CACHE_ECR}:latest-x86_cpu"
        else
            clean_branch=$(clean_docker_tag "${BUILDKITE_BRANCH:-unknown}")
            cache="${TEST_CACHE_ECR}:${clean_branch}-x86_cpu"
        fi
        CACHE_TO="$cache"
        CACHE_FROM="$cache"
        CACHE_FROM_BASE_BRANCH="$cache"
    else
        CACHE_TO="${TEST_CACHE_ECR}:pr-${BUILDKITE_PULL_REQUEST}-x86_cpu"
        CACHE_FROM="${TEST_CACHE_ECR}:pr-${BUILDKITE_PULL_REQUEST}-x86_cpu"
        if [[ "${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}" == "main" ]]; then
            CACHE_FROM_BASE_BRANCH="${MAIN_CACHE_ECR}:latest-x86_cpu"
        else
            clean_base=$(clean_docker_tag "${BUILDKITE_PULL_REQUEST_BASE_BRANCH}")
            CACHE_FROM_BASE_BRANCH="${TEST_CACHE_ECR}:${clean_base}-x86_cpu"
        fi
    fi

    CACHE_FROM_MAIN="${MAIN_CACHE_ECR}:latest-x86_cpu"
}

# authenticate with AWS ECR (public, for the image; private, for the cache repos)
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com || true

# skip build if image already exists
if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image found"
else
  echo "Image not found, proceeding with build..."

  prepare_cache_tags
  echo "--- :mag: Cache tags"
  echo "CACHE_TO: ${CACHE_TO}"
  echo "CACHE_FROM: ${CACHE_FROM}"
  echo "CACHE_FROM_BASE_BRANCH: ${CACHE_FROM_BASE_BRANCH}"
  echo "CACHE_FROM_MAIN: ${CACHE_FROM_MAIN}"

  # dedupe cache-from refs so identical branch/main fallbacks aren't repeated
  CACHE_FROM_ARGS=()
  SEEN_REFS=""
  for ref in "$CACHE_FROM" "$CACHE_FROM_BASE_BRANCH" "$CACHE_FROM_MAIN"; do
    if [[ "$SEEN_REFS" != *"|${ref}|"* ]]; then
      CACHE_FROM_ARGS+=(--cache-from "type=registry,ref=${ref}")
      SEEN_REFS="${SEEN_REFS}|${ref}|"
    fi
  done

  echo "--- :docker: Setting up buildx"
  docker buildx create --name vllm-cpu-builder --driver docker-container --use || true
  docker buildx inspect --bootstrap

  # build and push
  docker buildx build --file docker/Dockerfile.cpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    --build-arg VLLM_CPU_X86=true \
    --build-arg USE_SCCACHE=1 \
    --tag "$IMAGE" \
    --target vllm-test \
    "${CACHE_FROM_ARGS[@]}" \
    --cache-to "type=registry,ref=${CACHE_TO},mode=max" \
    --push \
    --progress plain .
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
