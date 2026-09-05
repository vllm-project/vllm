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
DISK_USAGE_THRESHOLD=${DISK_USAGE_THRESHOLD:-80}
BUILDKIT_CACHE_MAX=${BUILDKIT_CACHE_MAX:-200GB}

# Reclaim disk only when the host is under pressure. We trim (not purge) the
# shared BuildKit cache so cross-job/cross-agent reuse stays intact, and only
# touch dangling images; other agents' uniquely tagged images are left alone.
# Cache mounts (exec.cachemount: uv/cargo/apt) are excluded so they survive
# pruning -- they're what let installs reuse packages instead of hitting the
# network, and are otherwise reclaimable like any other build cache record.
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
        docker builder prune -f --keep-storage="$BUILDKIT_CACHE_MAX" --filter type!=exec.cachemount || true
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

# Opportunistically warm the local base image cache; `docker build` below
# omits `--pull` so it already prefers a cached image over the network. A
# single attempt only -- the retry loop below covers a miss here too.
echo "--- :docker: Pre-fetching base image"
docker pull ubuntu:22.04 || true

# building the docker image
echo "--- :docker: Building Docker image"
BUILD_RETRY_PATTERN='dial tcp|i/o timeout|failed to authorize|TLS handshake timeout|connection reset|Could not resolve host|Temporary failure in name resolution|dns error: failed to lookup address information|client error \(Connect\)|error sending request for url|Failed to fetch:'
BUILD_MAX_ATTEMPTS=4          # 1 initial + 3 retries
BUILD_RETRY_WAITS=(10 20 40)  # seconds to wait before retry 1/2/3
build_log="$(mktemp)"
attempt=1
while true; do
    if docker build --progress plain --tag "$IMAGE_NAME" --target vllm-test \
            --build-arg USE_SCCACHE=1 --build-arg SCCACHE_LOCAL_ONLY=1 \
            -f docker/Dockerfile.cpu . 2>&1 | tee "$build_log"; then
        break
    fi
    if [ "$attempt" -ge "$BUILD_MAX_ATTEMPTS" ] || ! grep -qE "$BUILD_RETRY_PATTERN" "$build_log"; then
        rm -f "$build_log"
        exit 1
    fi
    wait_s="${BUILD_RETRY_WAITS[$((attempt - 1))]}"
    echo "--- :docker: Transient network error during build (attempt $attempt/$BUILD_MAX_ATTEMPTS), retrying in ${wait_s}s"
    sleep "$wait_s"
    attempt=$((attempt + 1))
done
rm -f "$build_log"

# Run the image, setting --shm-size=4g for tensor parallel.
docker run --rm --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_CI_ENV=1 -e VLLM_CPU_SIM_MULTI_NUMA=1 -e VLLM_CPU_ATTN_SPLIT_KV=0 --shm-size=4g "$IMAGE_NAME" \
        timeout "$TIMEOUT_VAL" bash -c "set -euox pipefail; echo \"--- Print packages\"; pip list; echo \"--- Running tests\"; ${TEST_COMMAND}"
