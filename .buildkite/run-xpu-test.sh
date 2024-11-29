#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t xpu-test -f Dockerfile.xpu .

# Setup cleanup
remove_docker_container() { docker rm -f xpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image and test offline inference/tensor parallel
docker run -it -d --name xpu-test --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path xpu-test /bin/bash
docker exec xpu-test bash -c "python3 examples/offline_inference.py"
docker exec xpu-test bash -c "
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python3 benchmarks/benchmark_throughput.py \
  --backend=vllm \
  --dataset=./ShareGPT_V3_unfiltered_cleaned_split.json \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --num-prompts=10 \
  -tp=2 \
  --trust-remote-code \
  --device=xpu \
  --dtype=float16 \
  --enforce-eager \
  --distributed-executor-backend=ray \
  --max-model-len=4096
"
