#!/bin/bash

# This script should be run inside the vllm container. Enter the latest vllm container by
# docker run -it --runtime nvidia --gpus all --env "HF_TOKEN=<your HF TOKEN>"     --entrypoint /bin/bash  vllm/vllm-openai:latest
# (please modify `<your HF TOKEN>` to your own huggingface token in the above command
# Then, copy-paste this file into the docker and execute it using bash.
# Benchmarking results will be inside /vllm-workspace/vllm/benchmarks/*.txt

set -xe

# Get the number of GPUs
gpu_count=$(nvidia-smi --list-gpus | wc -l)

if [[ $gpu_count -gt 0 ]]; then
  echo "GPU found. Continue"
else
  echo "Need at least 1 GPU to run benchmarking."
  exit 1
fi


# Check if HF_TOKEN exists and starts with "hf_"
if [[ -z "$HF_TOKEN" ]]; then
  echo "Error: HF_TOKEN is not set."
  exit 1
elif [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
  echo "Error: HF_TOKEN does not start with 'hf_'."
  exit 1
else
  echo "HF_TOKEN is set and valid."
fi

# install wget and curl
(which wget && which curl) || (apt-get update && apt-get install -y wget curl)

# clone vllm repo for benchmarking
git clone https://github.com/vllm-project/vllm.git
cd vllm/benchmarks/
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

echo "Run Llama 7B benchmarking."
# offline -- 1GPU, Llama 7B
CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --model meta-llama/Llama-2-7b-hf 2>&1 | tee offline_7B.txt
# online -- 1 GPU, Llama 7B
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --swap-space 16 --disable-log-requests &
server_pid=$!
timeout 200 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1
python3 benchmark_serving.py --backend vllm --model meta-llama/Llama-2-7b-chat-hf --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 2>&1 | tee online_7B.txt
kill $server_pid



if [[ $gpu_count -gt 3 ]]; then
  echo "Run Llama 70B benchmarking."
  # offline -- 4GPU, Llama 70B
	CUDA_VISIBLE_DEVICES=0,1,2,3 python3 benchmark_throughput.py --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --model meta-llama/Llama-2-70b-hf -tp 4  2>&1 | tee offline_70B.txt
	# online -- 4 GPU, Llama 70B
	CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-70b-chat-hf --swap-space 16 --disable-log-requests -tp 4 &
	server_pid=$!
	# Use a small timeout, as the model is already cached during offline benchmarking.
	timeout 300 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1
	python3 benchmark_serving.py --backend vllm --model meta-llama/Llama-2-70b-chat-hf --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 2>&1 | tee online_70B.txt
	kill $server_pid
else
  echo "Skip Llama 70B benchmarking as there are less than 4 GPUs."
  exit 1
fi
