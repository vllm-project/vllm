#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t cpu-test -f Dockerfile.ppc64le .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image, setting --shm-size=4g for tensor parallel.
source /etc/environment
#docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true --network host -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --shm-size=4g --name cpu-test cpu-test
docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true --network host -e HF_TOKEN="$HF_TOKEN" --name cpu-test cpu-test

function cpu_tests() {
  set -e

  # Run basic model test
  docker exec cpu-test bash -c "
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

  # online inference
  docker exec cpu-test bash -c "
    set -e
    python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-125m & 
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
timeout 25m bash -c "cpu_tests"
