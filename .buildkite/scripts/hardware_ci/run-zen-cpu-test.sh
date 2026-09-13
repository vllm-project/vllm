#!/bin/bash

# Run the AMD Zen CPU kernel tests on the zen5 hardware. This builds the zen test
# image by layering docker/Dockerfile.zen on a CPU base, then runs the given test
# command inside the container with NUMA/cpuset pinning.
#
# To spare the scarce zen5 box from recompiling vLLM, prefer reusing the shared
# <repo>:<commit>-cpu image that image-build-cpu publishes to public ECR (anonymous
# pull, no credentials needed) as the base; fall back to building it from source.
# This mirrors .buildkite/image_build/image_build_zen_cpu.sh.
set -euox pipefail

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-0-47}
NUMA_NODE=${NUMA_NODE:-0}
IMAGE_NAME="zen-cpu-test-$NUMA_NODE"
FALLBACK_BASE_IMAGE="zen-cpu-base-$NUMA_NODE"
TIMEOUT_VAL=$1
TEST_COMMAND=$2

# The shared CPU image published by image-build-cpu. Only resolvable when the
# Buildkite registry env vars are present (i.e. in CI, not local runs).
SHARED_CPU_IMAGE=""
if [ -n "${REGISTRY:-}" ] && [ -n "${REPO:-}" ] && [ -n "${BUILDKITE_COMMIT:-}" ]; then
    SHARED_CPU_IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu"
fi

# Step 1: obtain the CPU base image that Dockerfile.zen layers on. Prefer pulling
# the published `-cpu` image; fall back to building it from source if unavailable.
if [ -n "$SHARED_CPU_IMAGE" ] && docker pull "$SHARED_CPU_IMAGE"; then
    echo "--- :docker: Using published CPU image as base: $SHARED_CPU_IMAGE"
    BASE_IMAGE="$SHARED_CPU_IMAGE"
else
    echo "--- :docker: Published CPU image unavailable; building base from source"
    docker build --progress plain --tag "$FALLBACK_BASE_IMAGE" \
        --target vllm-openai -f docker/Dockerfile.cpu .
    BASE_IMAGE="$FALLBACK_BASE_IMAGE"
fi

# Step 2: build the zen test image on top of the CPU base.
echo "--- :docker: Building Zen test image"
docker build --progress plain --tag "$IMAGE_NAME" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --target vllm-zen-test -f docker/Dockerfile.zen .

# Run the image, setting --shm-size=4g for tensor parallel.
docker run --rm --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_CI_ENV=1 -e VLLM_CPU_SIM_MULTI_NUMA=1 --shm-size=4g "$IMAGE_NAME" \
        timeout "$TIMEOUT_VAL" bash -c "set -euox pipefail; echo \"--- Print packages\"; pip list; echo \"--- Running tests\"; ${TEST_COMMAND}"
