#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -euox pipefail

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-48-95}
NUMA_NODE=${NUMA_NODE:-1}
IMAGE_NAME="cpu-test-$NUMA_NODE"
TIMEOUT_VAL=$1
TEST_COMMAND=$2

# building the docker image
echo "--- :docker: Building Docker image"
docker build --progress plain --tag "$IMAGE_NAME" --target vllm-test -f docker/Dockerfile.cpu .

# Run the image, setting --shm-size=4g for tensor parallel.
docker run --rm --cpuset-cpus=$CORE_RANGE --cpuset-mems=$NUMA_NODE -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_CI_ENV=1 -e VLLM_CPU_SIM_MULTI_NUMA=1 --shm-size=4g $IMAGE_NAME \
        timeout $TIMEOUT_VAL bash -c "set -euox pipefail; echo \"--- Print packages\"; pip list; echo \"--- Running tests\"; ${TEST_COMMAND}"
