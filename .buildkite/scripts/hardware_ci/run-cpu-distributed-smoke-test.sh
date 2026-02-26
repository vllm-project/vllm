#!/bin/bash
set -euox pipefail

echo "--- PP+TP"
vllm serve meta-llama/Llama-3.2-3B-Instruct -tp=2 -pp=2 &
server_pid=$!
timeout 600 bash -c "until curl localhost:8000/v1/models; do sleep 1; done" || exit 1
vllm bench serve \
    --backend vllm \
    --dataset-name random \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --num-prompts 20 \
    --endpoint /v1/completions
kill -s SIGTERM $server_pid &

echo "--- DP+TP"
vllm serve meta-llama/Llama-3.2-3B-Instruct -tp=2 -dp=2 &
server_pid=$!
timeout 600 bash -c "until curl localhost:8000/v1/models; do sleep 1; done" || exit 1
vllm bench serve \
    --backend vllm \
    --dataset-name random \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --num-prompts 20 \
    --endpoint /v1/completions
kill -s SIGTERM $server_pid &
