#!/bin/bash

python vllm/separated_encoder/encoder_cache_transfer/broker.py &

CUDA_VISIBLE_DEVICES="0" python vllm/entrypoints/openai/api_server.py --model /workspace/vllm/Qwen2.5-VL-3B-Instruct --port 19534 --enable-request-id-headers --max-num-seqs 128 --instance-type "encode" --connector-workers-num 128 &

CUDA_VISIBLE_DEVICES="1" python vllm/entrypoints/openai/api_server.py --model /workspace/vllm/Qwen2.5-VL-3B-Instruct --port 19535 --enable-request-id-headers --max-num-seqs 256 --instance-type "prefill+decode" --connector-workers-num 128 &

wait