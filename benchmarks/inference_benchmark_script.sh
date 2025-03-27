#!/bin/bash

wget --no-verbose https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
pip install --upgrade google-cloud-storage && rm -rf inference-benchmark
git clone https://github.com/AI-Hypercomputer/inference-benchmark
echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update && apt-get install -y google-cloud-sdk
apt-get -y install jq
export PJRT_DEVICE=TPU
python inference-benchmark/benchmark_serving.py --save-json-results --port=8009 --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer=meta-llama/Meta-Llama-3-8B --request-rate=1 --backend=vllm --num-prompts=300 --max-input-length=1024 --max-output-length=1024 --file-prefix=benchmark --models=meta-llama/Meta-Llama-3-8B --stream-request --output-bucket="$0"
ls
