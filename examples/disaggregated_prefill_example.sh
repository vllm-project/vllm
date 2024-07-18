
export VLLM_LOGGING_LEVEL=DEBUG

# prefilling instance
VLLM_DISAGG_PREFILL_ROLE=prefill CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 \
    --port 8100 \
    -tp 4 \
    --disable-log-stats \
    --disable-log-requests \
    --enable-chunked-prefill &

sleep 10

VLLM_DISAGG_PREFILL_ROLE=decode CUDA_VISIBLE_DEVICES=4,5,6,7 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 \
    --port 8200 \
    -tp 4 \
    --disable-log-stats \
    --disable-log-requests \
    --enable-chunked-prefill &


