#!/bin/bash

# Run the AMD Zen CPU kernel tests. The zen5 hardware is scarce, so this resolves
# the test image through three tiers, cheapest first, to keep the box testing
# rather than compiling:
#   Tier 1: pull the fully pre-built zen image (image-build-zen-cpu). No build.
#   Tier 2: pull the shared CPU base (image-build-cpu) and build only the thin
#           zen layer (zentorch + test deps) on top. Skips the from-source
#           vLLM compile -- the expensive part.
#   Tier 3: build everything from source locally (local dev / total ECR outage).
set -euox pipefail

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-48-95}
NUMA_NODE=${NUMA_NODE:-1}
IMAGE_NAME="zen-cpu-test-$NUMA_NODE"
BASE_IMAGE_NAME="$IMAGE_NAME-base"
TIMEOUT_VAL=$1
TEST_COMMAND=$2

# Commit-SHA-addressed images published to ECR. Only resolvable when the
# Buildkite registry env vars are present (i.e. in CI, not local runs).
PREBUILT_ZEN_IMAGE=""
SHARED_CPU_IMAGE=""
if [ -n "${REGISTRY:-}" ] && [ -n "${REPO:-}" ] && [ -n "${BUILDKITE_COMMIT:-}" ]; then
    PREBUILT_ZEN_IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-zen-cpu"
    SHARED_CPU_IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu"
fi

ecr_login() {
    aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" || true
}

# Build the zen layer (zentorch + test deps via docker/Dockerfile.zen) on top of
# an existing base image, tagging the result as IMAGE_NAME. Mirrors
# .buildkite/image_build/image_build_zen_cpu.sh.
build_zen_layer() {
    local base="$1"
    echo "--- :docker: Building Zen test image on base $base"
    docker build --progress plain --tag "$IMAGE_NAME" --build-arg BASE_IMAGE="$base" --target vllm-zen-test -f docker/Dockerfile.zen .
}

# Tier 3: build the CPU base from source, then the zen layer on top.
build_from_source() {
    echo "--- :docker: Building CPU base image from source"
    docker build --progress plain --tag "$BASE_IMAGE_NAME" --target vllm-openai -f docker/Dockerfile.cpu .
    build_zen_layer "$BASE_IMAGE_NAME"
}

if [ -n "$PREBUILT_ZEN_IMAGE" ]; then
    ecr_login
    if docker pull "$PREBUILT_ZEN_IMAGE"; then
        # Tier 1: fully pre-built zen image; no build needed.
        echo "--- :docker: Tier 1: using pre-built zen image $PREBUILT_ZEN_IMAGE"
        docker tag "$PREBUILT_ZEN_IMAGE" "$IMAGE_NAME"
    elif docker pull "$SHARED_CPU_IMAGE"; then
        # Tier 2: zen image missing; reuse the CPU base and build only the layer.
        echo "--- :docker: Tier 2: zen image unavailable; building zen layer on pulled CPU base $SHARED_CPU_IMAGE"
        build_zen_layer "$SHARED_CPU_IMAGE"
    else
        # Tier 3: nothing to pull; build everything from source.
        echo "--- :docker: Tier 3: no pre-built images available; building from source"
        build_from_source
    fi
else
    # No registry access (e.g. local dev): build everything from source.
    echo "--- :docker: Registry env vars unset; building from source"
    build_from_source
fi

# Run the image, setting --shm-size=4g for tensor parallel.
docker run --rm --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_CI_ENV=1 -e VLLM_CPU_SIM_MULTI_NUMA=1 --shm-size=4g "$IMAGE_NAME" \
        timeout "$TIMEOUT_VAL" bash -c "set -euox pipefail; echo \"--- Print packages\"; pip list; echo \"--- Running tests\"; ${TEST_COMMAND}"