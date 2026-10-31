#!/bin/bash

# Run the AMD Zen CPU tests on the zen5 hardware.

set -euox pipefail

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-0-47}
NUMA_NODE=${NUMA_NODE:-0}
IMAGE_NAME="zen-cpu-test-$NUMA_NODE"
FALLBACK_BASE_IMAGE="zen-cpu-base-$NUMA_NODE"
TIMEOUT_VAL=$1
TEST_COMMAND=$2

# Install zentorch directly in Dockerfile.zen instead of `uv pip install
# "vllm[zen]"`, which makes uv re-resolve vLLM and fail on its transitive
# triton-cpu URL dependency. Keep in sync with the `zen` extra in setup.py.
ZENTORCH_VERSION="${ZENTORCH_VERSION:-2.14.0.1}"

CLEANUP_IMAGES=()
cleanup() {
    if [ "${#CLEANUP_IMAGES[@]}" -gt 0 ]; then
        docker rmi -f "${CLEANUP_IMAGES[@]}" > /dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# Published images (resolvable only when the Buildkite registry env vars are
# present, i.e. in CI). Pulls from public ECR are anonymous.
ZEN_IMAGE=""
SHARED_CPU_IMAGE=""
if [ -n "${REGISTRY:-}" ] && [ -n "${REPO:-}" ] && [ -n "${BUILDKITE_COMMIT:-}" ]; then
    ZEN_IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-zen-cpu"
    SHARED_CPU_IMAGE="$REGISTRY/$REPO:$BUILDKITE_COMMIT-cpu"
fi

if [ -n "$ZEN_IMAGE" ] && docker pull "$ZEN_IMAGE"; then
    # Fast path: reuse the prebuilt zen image; no local build.
    echo "--- :docker: Using published Zen image: $ZEN_IMAGE"
    IMAGE_NAME="$ZEN_IMAGE"
    CLEANUP_IMAGES+=("$ZEN_IMAGE")
else
    # Step 1: obtain the CPU base image that Dockerfile.zen layers on. Prefer
    # pulling the published `-cpu` image; fall back to building it from source.
    if [ -n "$SHARED_CPU_IMAGE" ] && docker pull "$SHARED_CPU_IMAGE"; then
        echo "--- :docker: Using published CPU image as base: $SHARED_CPU_IMAGE"
        BASE_IMAGE="$SHARED_CPU_IMAGE"
        CLEANUP_IMAGES+=("$SHARED_CPU_IMAGE")
    else
        echo "--- :docker: Published CPU image unavailable; building base from source"
        docker build --progress plain --tag "$FALLBACK_BASE_IMAGE" \
            --target vllm-openai -f docker/Dockerfile.cpu .
        BASE_IMAGE="$FALLBACK_BASE_IMAGE"
        CLEANUP_IMAGES+=("$FALLBACK_BASE_IMAGE")
    fi

    # Step 2: build the zen test image on top of the CPU base.
    echo "--- :docker: Building Zen test image"
    docker build --progress plain --tag "$IMAGE_NAME" \
        --build-arg BASE_IMAGE="$BASE_IMAGE" \
        --build-arg ZENTORCH_VERSION="$ZENTORCH_VERSION" \
        --target vllm-zen-test -f docker/Dockerfile.zen .
    CLEANUP_IMAGES+=("$IMAGE_NAME")
fi

# Run the image, setting --shm-size=4g for tensor parallel. The in-container
# command verifies the installed zentorch matches the pinned version and that
# its torch.compile backend is registered (i.e. not the stable-only build), so
# a mismatch fails loudly instead of silently testing a different zentorch.
docker run --rm \
    --cpuset-cpus="$CORE_RANGE" \
    --cpuset-mems="$NUMA_NODE" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --privileged=true \
    -e HF_TOKEN \
    -e VLLM_CPU_KVCACHE_SPACE=16 \
    -e VLLM_CPU_CI_ENV=1 \
    -e VLLM_CPU_SIM_MULTI_NUMA=1 \
    --shm-size=4g \
    "$IMAGE_NAME" \
    timeout "$TIMEOUT_VAL" bash -c "set -euox pipefail
echo \"--- Print packages\"
pip list
echo \"--- Verify zentorch\"
python -c \"
import importlib.metadata as m, torch, zentorch
from torch._dynamo import list_backends
v = m.version('zentorch')
assert v == '${ZENTORCH_VERSION}', (
    'zentorch ' + v + ' != expected ${ZENTORCH_VERSION}'
)
assert 'zentorch' in list_backends(), (
    'zentorch torch.compile backend not registered (stable-only build?)'
)
\"
echo \"--- Running tests\"
${TEST_COMMAND}"
