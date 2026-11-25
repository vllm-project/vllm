#!/bin/bash
set -e

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <registry> <repo> <commit> [x86|arm64]"
  exit 1
fi

REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3

# SUFFIX names the image tag and the cache refs.
case "${4:-x86}" in
  x86)
    SUFFIX="cpu"
    BUILD_ARGS=(--build-arg VLLM_CPU_X86=true --build-arg USE_SCCACHE=1)
    # When TORCH_NIGHTLY=1, build the CPU image against torch nightly and tag it
    # with the -torch-nightly-cpu suffix the test steps pull on the nightly lane.
    # Its cache gets its own ref too: it resolves a different torch, so sharing
    # one with the regular lane would have the two evict each other on every run.
    if [[ "${TORCH_NIGHTLY:-0}" == "1" ]]; then
      SUFFIX="torch-nightly-cpu"
      BUILD_ARGS+=(--build-arg PYTORCH_NIGHTLY=1)
    fi
    ;;
  arm64)  # a native build, without the torch nightly lane
    SUFFIX="arm64-cpu"
    BUILD_ARGS=(--build-arg USE_SCCACHE=1)
    ;;
  *)
    echo "Unknown arch: $4 (expected x86 or arm64)" >&2
    exit 1
    ;;
esac
IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-$SUFFIX"

# replace invalid characters in Docker image tags and truncate to 128 chars
clean_docker_tag() {
    local input="$1"
    echo "$input" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-128
}

# resolve and set: CACHE_TO, CACHE_FROM, CACHE_FROM_BASE_BRANCH, CACHE_FROM_MAIN
# Reuses the same ECR cache repos as the CUDA image build, with SUFFIX on
# every tag so CPU and CUDA cache blobs stay isolated.
prepare_cache_tags() {
    TEST_CACHE_ECR="936637512419.dkr.ecr.us-east-1.amazonaws.com/vllm-ci-test-cache"
    MAIN_CACHE_ECR="936637512419.dkr.ecr.us-east-1.amazonaws.com/vllm-ci-postmerge-cache"

    if [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
        if [[ "${BUILDKITE_BRANCH:-}" == "main" ]]; then
            cache="${MAIN_CACHE_ECR}:latest-${SUFFIX}"
        else
            clean_branch=$(clean_docker_tag "${BUILDKITE_BRANCH:-unknown}")
            cache="${TEST_CACHE_ECR}:${clean_branch}-${SUFFIX}"
        fi
        CACHE_TO="$cache"
        CACHE_FROM="$cache"
        CACHE_FROM_BASE_BRANCH="$cache"
    else
        CACHE_TO="${TEST_CACHE_ECR}:pr-${BUILDKITE_PULL_REQUEST}-${SUFFIX}"
        CACHE_FROM="${TEST_CACHE_ECR}:pr-${BUILDKITE_PULL_REQUEST}-${SUFFIX}"
        if [[ "${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}" == "main" ]]; then
            CACHE_FROM_BASE_BRANCH="${MAIN_CACHE_ECR}:latest-${SUFFIX}"
        else
            clean_base=$(clean_docker_tag "${BUILDKITE_PULL_REQUEST_BASE_BRANCH}")
            CACHE_FROM_BASE_BRANCH="${TEST_CACHE_ECR}:${clean_base}-${SUFFIX}"
        fi
    fi

    CACHE_FROM_MAIN="${MAIN_CACHE_ECR}:latest-${SUFFIX}"
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

  # build and push. ignore-error=true covers the cache export only: a failed
  # cache write is a warning, a failed build or image push still fails the step.
  docker buildx build --file docker/Dockerfile.cpu \
    --build-arg max_jobs=16 \
    --build-arg buildkite_commit="$BUILDKITE_COMMIT" \
    "${BUILD_ARGS[@]}" \
    --tag "$IMAGE" \
    --target vllm-test \
    "${CACHE_FROM_ARGS[@]}" \
    --cache-to "type=registry,ref=${CACHE_TO},mode=max,ignore-error=true" \
    --push \
    --progress plain .
fi

.buildkite/scripts/annotate-image-build.sh "$IMAGE"
