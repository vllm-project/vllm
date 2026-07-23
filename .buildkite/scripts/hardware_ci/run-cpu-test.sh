#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -euox pipefail

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-48-95}
NUMA_NODE=${NUMA_NODE:-1}
AGENT_SLOT=${AGENT_SLOT:-}
IMAGE_NAME="cpu-test-${NUMA_NODE}${AGENT_SLOT:+-${AGENT_SLOT}}"
TIMEOUT_VAL=$1
TEST_COMMAND=$2

# Disk hygiene knobs. Reclaim space only once the Docker root filesystem crosses
# DISK_USAGE_THRESHOLD percent, and cap the shared BuildKit cache at
# BUILDKIT_CACHE_MAX so subsequent builds keep reusing the hottest layers.
DISK_USAGE_THRESHOLD=${DISK_USAGE_THRESHOLD:-70}
BUILDKIT_CACHE_MAX=${BUILDKIT_CACHE_MAX:-80GB}

# Reclaim disk only when the host is under pressure. We trim (not purge) the
# shared BuildKit cache so cross-job/cross-agent reuse stays intact, and only
# touch dangling images; other agents' uniquely tagged images are left alone.
prune_if_disk_pressure() {
    local docker_root disk_usage
    docker_root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)
    if [ -z "$docker_root" ]; then
        return 0
    fi
    disk_usage=$(df "$docker_root" 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')
    if [ "${disk_usage:-0}" -gt "$DISK_USAGE_THRESHOLD" ]; then
        echo "--- :broom: Disk usage ${disk_usage}% exceeds ${DISK_USAGE_THRESHOLD}%, reclaiming space"
        docker image prune -f || true
        docker builder prune -f --keep-storage="$BUILDKIT_CACHE_MAX" || true
    else
        echo "Disk usage ${disk_usage:-unknown}% within ${DISK_USAGE_THRESHOLD}% threshold; skipping prune"
    fi
}

# Always drop this agent's image once the job ends (the default builder never
# uses it as a cache source, so removing it costs no rebuild speed), then
# reclaim space if needed. Guard every docker call with `|| true` so the trap
# never overrides the test's exit code.
cleanup() {
    docker image rm -f "$IMAGE_NAME" || true
    prune_if_disk_pressure
}
trap cleanup EXIT

# Free space up front so a nearly-full host doesn't fail the build.
prune_if_disk_pressure

# building the docker image
echo "--- :docker: Building Docker image"
docker build --progress plain --tag "$IMAGE_NAME" --target vllm-test -f docker/Dockerfile.cpu .

# Run the image, setting --shm-size=4g for tensor parallel.
docker run --rm --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_CI_ENV=1 -e VLLM_CPU_SIM_MULTI_NUMA=1 -e VLLM_CPU_ATTN_SPLIT_KV=0 --shm-size=4g "$IMAGE_NAME" \
        timeout "$TIMEOUT_VAL" bash -c "set -euox pipefail; echo \"--- Print packages\"; pip list; echo \"--- Running tests\"; ${TEST_COMMAND}"
