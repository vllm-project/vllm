#!/bin/bash

# Run the AMD Zen CPU tests on the zen5 hardware.

set -euox pipefail

# The zen5 host's agent environment hook gives each agent its own socket
# (CORE_RANGE, NUMA_NODE) and slot (AGENT_SLOT); the defaults are for local runs.
if [ -n "${BUILDKITE:-}" ] && [ -z "${CORE_RANGE:-}" ]; then
    echo "WARNING: the agent did not set CORE_RANGE; jobs on this host may share cores"
fi
CORE_RANGE=${CORE_RANGE:-0-47}
NUMA_NODE=${NUMA_NODE:-0}
AGENT_SLOT=${AGENT_SLOT:-}
LOCAL_IMAGE="zen-cpu-test-${NUMA_NODE}${AGENT_SLOT:+-${AGENT_SLOT}}"
FALLBACK_BASE_IMAGE="zen-cpu-base-${NUMA_NODE}${AGENT_SLOT:+-${AGENT_SLOT}}"
TIMEOUT_VAL=$1
TEST_COMMAND=$2

# Install zentorch directly in Dockerfile.zen instead of `uv pip install
# "vllm[zen]"`, which makes uv re-resolve vLLM and fail on its transitive
# triton-cpu URL dependency. Keep in sync with the `zen` extra in setup.py.
ZENTORCH_VERSION="${ZENTORCH_VERSION:-2.14.0.2}"

# Same disk policy as run-cpu-test.sh: prune only when the Docker root is under
# pressure, keeping anything used within CACHE_MAX_AGE. Pulled registry images
# are shared by the agents on the host, so they are left to this prune instead
# of being removed when a job ends.
DISK_USAGE_THRESHOLD=${DISK_USAGE_THRESHOLD:-80}
CACHE_MAX_AGE=${CACHE_MAX_AGE:-24h}
prune_if_disk_pressure() {
    local docker_root disk_usage
    docker_root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)
    if [ -z "$docker_root" ]; then
        return 0
    fi
    disk_usage=$(df "$docker_root" 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')
    if [ "${disk_usage:-0}" -gt "$DISK_USAGE_THRESHOLD" ]; then
        echo "--- :broom: Disk usage ${disk_usage}% exceeds ${DISK_USAGE_THRESHOLD}%, reclaiming space"
        docker system prune --force --all --filter "until=${CACHE_MAX_AGE}" || true
    fi
}

cleanup() {
    docker image rm -f "$LOCAL_IMAGE" "$FALLBACK_BASE_IMAGE" > /dev/null 2>&1 || true
    prune_if_disk_pressure
}
trap cleanup EXIT
prune_if_disk_pressure

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
    RUN_IMAGE="$ZEN_IMAGE"
else
    # Step 1: obtain the CPU base image that Dockerfile.zen layers on. Prefer
    # pulling the published `-cpu` image; fall back to building it from source.
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
    docker build --progress plain --tag "$LOCAL_IMAGE" \
        --build-arg BASE_IMAGE="$BASE_IMAGE" \
        --build-arg ZENTORCH_VERSION="$ZENTORCH_VERSION" \
        --target vllm-zen-test -f docker/Dockerfile.zen .
    RUN_IMAGE="$LOCAL_IMAGE"
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
    "$RUN_IMAGE" \
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
