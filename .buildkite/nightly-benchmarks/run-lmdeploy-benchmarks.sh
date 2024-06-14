#!/bin/bash

# This script should be run inside the tgi container. Enter the latest tgi container by
# docker run --gpus all -e "HF_TOKEN=<your HF TOKEN>" -v ~/.cache/huggingface:/root/.cache/huggingface --entrypoint /bin/bash openmmlab/lmdeploy:latest 
# lmdeploy serve api_server internlm/internlm2-chat-7b
# docker run -it --gpus all -e "HF_TOKEN=<your HF TOKEN>"   --shm-size 1g --entrypoint /bin/bash ghcr.io/huggingface/text-generation-inference:2.0
# (please modify `<your HF TOKEN>` to your own huggingface token in the above command
# Then, copy-paste this file into any directory you prefer in the docker and execute it using bash.



set -ex
set -o pipefail

# install conda
(which wget && which curl) || (apt-get update && apt-get install -y wget curl)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
eval "$(cat ~/.bashrc | tail -n +15)"

# create conda environment for vllm
conda create -n vllm python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate vllm
pip install vllm

# clone vllm repo
cd /
git clone https://github.com/vllm-project/vllm.git
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# launch TGI server
lmdeploy serve api_server meta-llama/Llama-2-7b-hf  --server-port 8000 &
tgi_pid=$!
timeout 600 bash -c 'until curl localhost:8000/v1/completion; do sleep 1; done' || exit 1

# gradually reduce the request rate from 20, untill all request successed
request_rate=20
echo "Running benchmark with request rate $request_rate..."
python3 vllm/benchmarks/benchmark_serving.py --backend lmdeploy --model meta-llama/Llama-2-7b-chat-hf --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 200 --request-rate $request_rate --port 8000 --save-result 2>&1 | tee benchmark_serving.txt
kill $tgi_pid
