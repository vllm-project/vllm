model_path="/mnt/disk5/hf_models/DeepSeek-R1-BF16"

# skip warmup for now
export VLLM_SKIP_WARMUP="true"

QUANT_CONFIG=inc_quant_with_fp8kv_one_node_config.json \
VLLM_EP_SIZE=8 \
VLLM_TP_SIZE=8 \
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8858 \
    --block-size 128 \
    --model ${model_path} \
    --tokenizer ${model_path} \
    --device hpu --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 16384 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384 \
    --use-padding-aware-scheduling \
    --use-v2-block-manager \
    --distributed_executor_backend mp \
    --gpu_memory_utilization 0.90 \
    --quantization inc \
    --weights_load_device cpu \
    --kv_cache_dtype fp8_inc 2>&1 | tee benchmark_logs/vllm_ep8_tp8.log