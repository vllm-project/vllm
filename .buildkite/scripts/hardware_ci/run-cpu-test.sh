#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-48-95}
OMP_CORE_RANGE=${OMP_CORE_RANGE:-48-95}
NUMA_NODE=${NUMA_NODE:-1}

export CMAKE_BUILD_PARALLEL_LEVEL=32

# Setup cleanup
remove_docker_container() { 
    set -e; 
    docker rm -f cpu-test-"$NUMA_NODE" cpu-test-"$NUMA_NODE"-avx2 || true; 
}
trap remove_docker_container EXIT
remove_docker_container

# Try building the docker image
numactl -C "$CORE_RANGE" -N "$NUMA_NODE" docker build --tag cpu-test-"$NUMA_NODE" --target vllm-test -f docker/Dockerfile.cpu .
numactl -C "$CORE_RANGE" -N "$NUMA_NODE" docker build --build-arg VLLM_CPU_DISABLE_AVX512="true" --tag cpu-test-"$NUMA_NODE"-avx2 --target vllm-test -f docker/Dockerfile.cpu .

# Run the image, setting --shm-size=4g for tensor parallel.
docker run -itd --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --env VLLM_CPU_OMP_THREADS_BIND="$OMP_CORE_RANGE" --env VLLM_CPU_CI_ENV=1 --shm-size=4g --name cpu-test-"$NUMA_NODE" cpu-test-"$NUMA_NODE"
docker run -itd --cpuset-cpus="$CORE_RANGE" --cpuset-mems="$NUMA_NODE" --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --env VLLM_CPU_OMP_THREADS_BIND="$OMP_CORE_RANGE" --env VLLM_CPU_CI_ENV=1 --shm-size=4g --name cpu-test-"$NUMA_NODE"-avx2 cpu-test-"$NUMA_NODE"-avx2

function cpu_tests() {
  set -e
  export NUMA_NODE=$2

  # list packages
  docker exec cpu-test-"$NUMA_NODE"-avx2 bash -c "
    set -e
    pip list"

  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    pip list"

  # offline inference
  docker exec cpu-test-"$NUMA_NODE"-avx2 bash -c "
    set -e
    python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m"

  # Run basic model test
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    pytest -v -s tests/kernels/attention/test_cache.py -m cpu_model
    pytest -v -s tests/kernels/attention/test_mla_decode_cpu.py -m cpu_model
    pytest -v -s tests/models/language/generation -m cpu_model
    pytest -v -s tests/models/language/pooling -m cpu_model
    pytest -v -s tests/models/multimodal/generation \
                --ignore=tests/models/multimodal/generation/test_mllama.py \
                --ignore=tests/models/multimodal/generation/test_pixtral.py \
                -m cpu_model"

  # Run compressed-tensor test
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    pytest -s -v \
    tests/quantization/test_compressed_tensors.py::test_compressed_tensors_w8a8_static_setup \
    tests/quantization/test_compressed_tensors.py::test_compressed_tensors_w8a8_dynamic_per_token"

  # Run AWQ test
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    VLLM_USE_V1=0 pytest -s -v \
    tests/quantization/test_ipex_quant.py"

  # Run chunked-prefill and prefix-cache test
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    pytest -s -v -k cpu_model \
    tests/basic_correctness/test_chunked_prefill.py"  

  # online serving
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --dtype half & 
    timeout 600 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1
    VLLM_CPU_CI_ENV=0 python3 benchmarks/benchmark_serving.py \
      --backend vllm \
      --dataset-name random \
      --model facebook/opt-125m \
      --num-prompts 20 \
      --endpoint /v1/completions \
      --tokenizer facebook/opt-125m"

  # Run multi-lora tests
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    pytest -s -v \
    tests/lora/test_qwen2vl.py"
}

# All of CPU tests are expected to be finished less than 40 mins.
export -f cpu_tests
timeout 1h bash -c "cpu_tests $CORE_RANGE $NUMA_NODE"
