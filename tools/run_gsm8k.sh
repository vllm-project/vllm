#!/bin/bash
# GSM8K 5-shot via lm-eval-harness across kv_cache_dtype settings.
set -e
MODEL=${MODEL:-/data/modelscope/DeepSeek-V2-Lite}
TP=${TP:-4}
DTYPES=${DTYPES:-"auto fp8 turboquant_k8v4 turboquant_4bit_nc turboquant_3bit_nc"}
OUT=tools/reports/p3_gsm8k
GPU_MEM=${GPU_MEM:-0.75}
mkdir -p $OUT
for DT in $DTYPES; do
  echo "=== GSM8K 5-shot: kv_cache_dtype=$DT ==="
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,tensor_parallel_size=$TP,kv_cache_dtype=$DT,max_model_len=4096,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=$GPU_MEM,max_num_seqs=64,enforce_eager=True" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path $OUT/$DT 2>&1 | tee $OUT/$DT.log | tail -20
done
