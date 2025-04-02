# QUANT_CONFIG=scripts/inc_quant_with_fp8kv_config.json \
# VLLM_REQUANT_FP8_INC=1 \
# VLLM_ENABLE_RUNTIME_DEQUANT=1 \
# VLLM_USE_FP8_MATMUL=true \
VLLM_SKIP_WARMUP=true \
VLLM_EP_SIZE=8 \
VLLM_MOE_N_SLICE=1 \
VLLM_DELAYED_SAMPLING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_MLA_DISABLE_REQUANTIZATION=1 \
lm_eval --model vllm \
  --model_args "pretrained=/data/models/DeepSeek-R1-static,tensor_parallel_size=8,distributed_executor_backend=mp,trust_remote_code=true,max_model_len=4096,kv_cache_dtype=fp8_inc,use_v2_block_manager=True,dtype=bfloat16" \
  --tasks gsm8k --num_fewshot "5" --limit "256" \
  --batch_size "128"