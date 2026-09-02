#!/bin/bash

# This script builds the Zen CPU docker image and runs the test command
# inside the container. Mirrors run-cpu-test.sh's "build locally, then run"
# structure for parity. A follow-up may switch to pulling the pre-built
# image from registry (see image_build_zen_cpu.sh).
set -euox pipefail

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-48-95}
NUMA_NODE=${NUMA_NODE:-1}
IMAGE_NAME="zen-cpu-test-$NUMA_NODE"
CPU_BASE_IMAGE="vllm-openai-cpu-local:$NUMA_NODE"
TIMEOUT_VAL=$1
TEST_COMMAND=$2

# Build CPU base image (Dockerfile.zen layers on top of this).
echo "--- :docker: Building CPU base image"
docker build --progress plain \
    --tag "$CPU_BASE_IMAGE" \
    --target vllm-openai \
    -f docker/Dockerfile.cpu .

# Build the Zen CPU test image on top of the CPU base.
echo "--- :docker: Building Zen CPU test image"
docker build --progress plain \
    --build-arg BASE_IMAGE="$CPU_BASE_IMAGE" \
    --tag "$IMAGE_NAME" \
    --target vllm-zen-test \
    -f docker/Dockerfile.zen .

# Run the image, setting --shm-size=4g for tensor parallel.
docker run --rm --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_CI_ENV=1 -e VLLM_CPU_SIM_MULTI_NUMA=1 --shm-size=4g "$IMAGE_NAME" \
        timeout "$TIMEOUT_VAL" bash -c "set -euox pipefail; echo \"--- Print packages\"; pip list; echo \"--- Running tests\"; ${TEST_COMMAND}"
