#!/bin/bash
set -e

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit>"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3

# When TORCH_NIGHTLY=1, build the CPU image against torch nightly and tag it
# with the -torch-nightly-cpu suffix the test steps pull on the nightly lane.
PYTORCH_NIGHTLY_ARGS=()
if [[ "${TORCH_NIGHTLY:-0}" == "1" ]]; then
  IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-torch-nightly-cpu"
  PYTORCH_NIGHTLY_ARGS=(--build-arg PYTORCH_NIGHTLY=1)
else
  IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu"
fi

# replace invalid characters in Docker image tags and truncate to 128 chars
clean_docker_tag() {
    local input="$1"
    echo "$input" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-128
}

# resolve and set: CACHE_TO, CACHE_FROM, CACHE_FROM_BASE_BRANCH, CACHE_FROM_MAIN
# Reuses the same ECR cache repos as the CUDA image build, with an
# "x86_cpu" suffix on every tag so CPU and CUDA cache blobs stay isolated.
# The nightly lane appends "-nightly": it resolves a different torch, so
# sharing a cache ref with the regular lane would have the two evict each
# other from the same ref on every run.
prepare_cache_tags() {
    TEST_CACHE_ECR="936637512419.dkr.ecr.us-east-1.amazonaws.com/vllm-ci-test-cache"
    MAIN_CACHE_ECR="936637512419.dkr.ecr.us-east-1.amazonaws.com/vllm-ci-postmerge-cache"
    CACHE_SUFFIX="x86_cpu"
    if [[ "${TORCH_NIGHTLY:-0}" == "1" ]]; then
        CACHE_SUFFIX="x86_cpu-nightly"
    fi

    if [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
        if [[ "${BUILDKITE_BRANCH:-}" == "main" ]]; then
            cache="${MAIN_CACHE_ECR}:latest-${CACHE_SUFFIX}"
        else
            clean_branch=$(clean_docker_tag "${BUILDKITE_BRANCH:-unknown}")
            cache="${TEST_CACHE_ECR}:${clean_branch}-${CACHE_SUFFIX}"
        fi
        CACHE_TO="$cache"
        CACHE_FROM="$cache"
        CACHE_FROM_BASE_BRANCH="$cache"
    else
        CACHE_TO="${TEST_CACHE_ECR}:pr-${BUILDKITE_PULL_REQUEST}-${CACHE_SUFFIX}"
        CACHE_FROM="${TEST_CACHE_ECR}:pr-${BUILDKITE_PULL_REQUEST}-${CACHE_SUFFIX}"
        if [[ "${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}" == "main" ]]; then
            CACHE_FROM_BASE_BRANCH="${MAIN_CACHE_ECR}:latest-${CACHE_SUFFIX}"
        else
            clean_base=$(clean_docker_tag "${BUILDKITE_PULL_REQUEST_BASE_BRANCH}")
            CACHE_FROM_BASE_BRANCH="${TEST_CACHE_ECR}:${clean_base}-${CACHE_SUFFIX}"
        fi
    fi

    CACHE_FROM_MAIN="${MAIN_CACHE_ECR}:latest-${CACHE_SUFFIX}"
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
  # network=host so the isolated docker-container builder can still reach the
  # EC2 instance metadata service for sccache's S3 credentials; recreate on
  # every run in case a stale builder without this driver-opt already exists
  # on a warm CI host.
  docker buildx rm vllm-cpu-builder >/dev/null 2>&1 || true
  docker buildx create --name vllm-cpu-builder --driver docker-container --driver-opt network=host --use
  docker buildx inspect --bootstrap

  # build and push
  docker buildx build --file docker/Dockerfile.cpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    --build-arg VLLM_CPU_X86=true \
    --build-arg USE_SCCACHE=1 \
    "${PYTORCH_NIGHTLY_ARGS[@]}" \
    --tag "$IMAGE" \
    --target vllm-test \
    "${CACHE_FROM_ARGS[@]}" \
    --cache-to "type=registry,ref=${CACHE_TO},mode=max" \
    --push \
    --progress plain .
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
