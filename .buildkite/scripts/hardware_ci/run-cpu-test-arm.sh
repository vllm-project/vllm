#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-0-16}
OMP_CORE_RANGE=${OMP_CORE_RANGE:-0-16}

export CMAKE_BUILD_PARALLEL_LEVEL=16

# Setup cleanup
remove_docker_container() {
    set -e;
    docker rm -f cpu-test || true;
}
trap remove_docker_container EXIT
remove_docker_container

# Try building the docker image
docker build --tag cpu-test --target vllm-test -f docker/Dockerfile.cpu .

# Run the image
docker run -itd --cpuset-cpus="$CORE_RANGE" --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=16 --env VLLM_CPU_CI_ENV=1 -e E2E_OMP_THREADS="$OMP_CORE_RANGE" --shm-size=4g --name cpu-test cpu-test

function cpu_tests() {
  set -e

  docker exec cpu-test bash -c "
    set -e
    pip list"

  # offline inference
  docker exec cpu-test bash -c "
    set -e
    python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m"

  # Run kernel tests
  docker exec cpu-test bash -c "
    set -e
    pytest -x -v -s tests/kernels/test_onednn.py
    pytest -x -v -s tests/kernels/attention/test_cpu_attn.py"

  # basic online serving
  docker exec cpu-test bash -c '
    set -e
    VLLM_CPU_OMP_THREADS_BIND=$E2E_OMP_THREADS vllm serve Qwen/Qwen3-0.6B --max-model-len 2048 &
    server_pid=$!
    timeout 600 bash -c "until curl localhost:8000/v1/models; do sleep 1; done" || exit 1
    vllm bench serve \
      --backend vllm \
      --dataset-name random \
      --model Qwen/Qwen3-0.6B \
      --num-prompts 20 \
      --endpoint /v1/completions
    kill -s SIGTERM $server_pid &'
}

# All of CPU tests are expected to be finished less than 40 mins.
export -f cpu_tests
timeout 2h bash -c cpu_tests
