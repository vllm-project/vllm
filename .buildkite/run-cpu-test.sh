#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# allow to bind to different cores
CORE_RANGE=${CORE_RANGE:-48-95}
NUMA_NODE=${NUMA_NODE:-1}

# Try building the docker image
numactl -C "$CORE_RANGE" -N "$NUMA_NODE" docker build -t cpu-test -f Dockerfile.cpu .
numactl -C "$CORE_RANGE" -N "$NUMA_NODE" docker build --build-arg VLLM_CPU_DISABLE_AVX512="true" -t cpu-test-avx2 -f Dockerfile.cpu .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test-"$NUMA_NODE" cpu-test-avx2-"$NUMA_NODE" || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image, setting --shm-size=4g for tensor parallel.
docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --cpuset-cpus="$CORE_RANGE"  \
 --cpuset-mems="$NUMA_NODE" --privileged=true --network host -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --shm-size=4g --name cpu-test-"$NUMA_NODE" cpu-test
docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --cpuset-cpus="$CORE_RANGE" \
 --cpuset-mems="$NUMA_NODE" --privileged=true --network host -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --shm-size=4g --name cpu-test-avx2-"$NUMA_NODE" cpu-test-avx2

function cpu_tests() {
  set -e

  # offline inference
  docker exec cpu-test-avx2-"$NUMA_NODE" bash -c "
    set -e
    python3 examples/offline_inference.py"

  # Run basic model test
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    pip install pytest pytest-asyncio \
      decord einops librosa peft Pillow sentence-transformers soundfile \
      transformers_stream_generator matplotlib datamodel_code_generator
    pip install torchvision --index-url https://download.pytorch.org/whl/cpu
    pytest -v -s tests/models/decoder_only/language -m cpu_model
    pytest -v -s tests/models/embedding/language -m cpu_model
    pytest -v -s tests/models/encoder_decoder/language -m cpu_model
    pytest -v -s tests/models/decoder_only/audio_language -m cpu_model
    pytest -v -s tests/models/decoder_only/vision_language -m cpu_model"

  # Run compressed-tensor test
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    pytest -s -v \
    tests/quantization/test_compressed_tensors.py::test_compressed_tensors_w8a8_static_setup \
    tests/quantization/test_compressed_tensors.py::test_compressed_tensors_w8a8_dynamic_per_token"

  # Run AWQ test
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    pytest -s -v \
    tests/quantization/test_ipex_quant.py"

  # online inference
  docker exec cpu-test-"$NUMA_NODE" bash -c "
    set -e
    export VLLM_CPU_KVCACHE_SPACE=10 
    export VLLM_CPU_OMP_THREADS_BIND=$1
    python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --dtype half & 
    timeout 600 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1
    python3 benchmarks/benchmark_serving.py \
      --backend vllm \
      --dataset-name random \
      --model facebook/opt-125m \
      --num-prompts 20 \
      --endpoint /v1/completions \
      --tokenizer facebook/opt-125m"
}

# All of CPU tests are expected to be finished less than 25 mins.
export -f cpu_tests
timeout 25m bash -c "cpu_tests $CORE_RANGE"
