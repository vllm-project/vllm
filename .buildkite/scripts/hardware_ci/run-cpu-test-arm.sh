#!/bin/bash

# Run the Arm CPU test suites against the image built by cpu-arm64-image-build.
set -euxo pipefail

SHARD_ID=${1:?Usage: run-cpu-test-arm.sh SHARD_ID IMAGE}
IMAGE=${2:?Usage: run-cpu-test-arm.sh SHARD_ID IMAGE}
if [[ ! "$SHARD_ID" =~ ^[0-2]$ ]]; then
    echo "SHARD_ID must be 0, 1, or 2" >&2
    exit 2
fi

JOB_SUFFIX=${BUILDKITE_JOB_ID:-local-$SHARD_ID}
CONTAINER_NAME="cpu-test-${JOB_SUFFIX//[^a-zA-Z0-9_.-]/-}"

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-0-31}
OMP_CORE_RANGE=${OMP_CORE_RANGE:-0-31}

export CMAKE_BUILD_PARALLEL_LEVEL=32

# Setup cleanup
remove_docker_container() {
    set -e;
    docker rm -f "$CONTAINER_NAME" || true;
}
trap remove_docker_container EXIT
remove_docker_container

docker pull "$IMAGE"

# Run the image
docker run -itd --cpuset-cpus="$CORE_RANGE" --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=16 --env VLLM_CPU_CI_ENV=1 -e E2E_OMP_THREADS="$OMP_CORE_RANGE" --shm-size=4g --name "$CONTAINER_NAME" "$IMAGE"

print_packages() {
  docker exec "$CONTAINER_NAME" bash -c "
    set -e
    pip list"
}

kernel_tests() {
  set -e
  docker exec "$CONTAINER_NAME" bash -c "
    set -e
    pytest -x -v -s tests/kernels/test_onednn.py
    pytest -x -v -s tests/kernels/attention/test_cpu_attn.py
    pytest -x -v -s tests/kernels/core/test_cpu_activation.py
    pytest -x -v -s tests/kernels/moe/test_cpu_fused_moe.py
    pytest -x -v -s tests/kernels/mamba/cpu/test_cpu_gdn_ops.py
    pytest -x -v -s tests/kernels/moe/test_cpu_int4_moe.py
    pytest -x -v -s tests/kernels/mamba/test_cpu_short_conv.py
    pytest -x -v -s tests/kernels/mamba/test_causal_conv1d.py
    pytest -x -v -s tests/kernels/mamba/test_mamba_ssm.py"
}

model_tests() {
  set -e
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN is not set. Skipping tests that require model downloads."
    return
  fi

  docker exec "$CONTAINER_NAME" bash -c "
    set -e
    python3 examples/basic/offline_inference/generate.py --model facebook/opt-125m"

  docker exec "$CONTAINER_NAME" bash -c "
    set -e
    pytest -x -v -s tests/models/multimodal/generation/test_whisper.py -m cpu_model
    pytest -x -v -s 'tests/models/language/pooling/test_embedding.py::test_models[sentence-transformers/all-MiniLM-L12-v2]'"

  docker exec "$CONTAINER_NAME" bash -c "
    set -e
    pytest -x -v -s tests/quantization/test_compressed_tensors.py::test_compressed_tensors_w8a8_logprobs"
}

serving_tests() {
  set -e
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN is not set. Skipping tests that require model downloads."
    return
  fi

  docker exec "$CONTAINER_NAME" bash -c '
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

  docker exec "$CONTAINER_NAME" bash -c '
    set -e
    VLLM_CPU_OMP_THREADS_BIND=$E2E_OMP_THREADS vllm serve Qwen/Qwen3.5-0.8B --max-model-len 2048 &
    server_pid=$!
    timeout 600 bash -c "until curl localhost:8000/v1/models; do sleep 1; done" || exit 1
    vllm bench serve \
      --backend vllm \
      --dataset-name random \
      --model Qwen/Qwen3.5-0.8B \
      --num-prompts 20 \
      --endpoint /v1/completions
    kill -s SIGTERM $server_pid &'
}

print_packages
export CONTAINER_NAME
export -f kernel_tests model_tests serving_tests
case "$SHARD_ID" in
  0) timeout 30m bash -c kernel_tests ;;
  1) timeout 30m bash -c model_tests ;;
  2) timeout 30m bash -c serving_tests ;;
esac
