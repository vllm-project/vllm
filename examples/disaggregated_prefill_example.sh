
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_PORT=12345
# export VLLM_TRACE_FUNCTION=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export GLOO_LOGGING_LEVEL=TRACE

# prefilling instance
VLLM_DISAGG_PREFILL_ROLE=prefill CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 \
    --port 8100 \
    -tp 4 \
    --enable-prefix-caching &

# decoding instance
VLLM_DISAGG_PREFILL_ROLE=decode CUDA_VISIBLE_DEVICES=4,5,6,7 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 \
    --port 8200 \
    -tp 4 \
    --enable-prefix-caching &


