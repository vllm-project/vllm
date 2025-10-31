#!/bin/bash

# Start vLLM Docker container with all necessary configurations
dzdo docker run --gpus all -it --rm \
  --name vllm-run \
  --ipc=host --shm-size=20g --ulimit memlock=-1 \
  -p 8001:8001 \
  -v "$HOME:$HOME" \
  -v "$HOME/Documents/MLSystems/vllm-distributed:/workspace" \
  -v /mnt/nvme/hf_cache:/mnt/nvme/hf_cache \
  -w /workspace \
  -e HF_HOME=/mnt/nvme/hf_cache \
  --entrypoint bash \
  susavlsh10/vllm-tknp:v1