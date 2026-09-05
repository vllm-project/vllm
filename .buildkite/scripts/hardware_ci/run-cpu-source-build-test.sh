#!/bin/bash

# This script builds the CPU source docker image (the vllm-src stage of
# docker/Dockerfile.cpu: full source tree plus build dependencies, but no
# prebuilt vLLM wheel) and verifies that a local source build,
# `pip install -e .`, succeeds inside it, followed by an import /
# `vllm serve --help` smoke check.
#
# It guards the developer local-build workflow (#9129), which the prebuilt
# wheel-based vllm-test image used by the other CPU test steps cannot catch.
set -euox pipefail

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-48-95}
NUMA_NODE=${NUMA_NODE:-1}
AGENT_SLOT=${AGENT_SLOT:-}
IMAGE_NAME="cpu-src-build-${NUMA_NODE}${AGENT_SLOT:+-${AGENT_SLOT}}"
TIMEOUT_VAL=${TIMEOUT_VAL:-90m}

cleanup() {
    docker image rm -f "$IMAGE_NAME" || true
}
trap cleanup EXIT

# Building the docker image. Layers up to and including vllm-src are shared
# with the regular cpu-test image build, so base/build-dependency layers are
# usually served from the BuildKit cache and only the source copy is new.
echo "--- :docker: Building Docker image (vllm-src stage)"
docker build --progress plain --tag "$IMAGE_NAME" --target vllm-src -f docker/Dockerfile.cpu .

# Run the editable source build and smoke checks inside the image.
docker run --rm --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" --shm-size=4g "$IMAGE_NAME" \
    timeout "$TIMEOUT_VAL" bash -c '
        set -euox pipefail
        cd /vllm-workspace
        echo "--- :hammer_and_wrench: pip install -e . (VLLM_TARGET_DEVICE=cpu)"
        VLLM_TARGET_DEVICE=cpu uv pip install --no-build-isolation -e .
        # Import from outside the source tree so the editable install itself is exercised.
        cd /
        echo "--- :mag: Smoke test: import vllm"
        python3 -c "import vllm; print(f\"vLLM version: {vllm.__version__}\")"
        echo "--- :mag: Smoke test: vllm serve --help"
        vllm serve --help > /dev/null
    '
