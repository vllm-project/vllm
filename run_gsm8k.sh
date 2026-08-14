#!/bin/bash
set -ux
cd /home/woosuk/workspace/vllm
export HF_HOME=/mnt/lustre/hf-models
OUT="${1:-gsm8k_baseline}"
.venv/bin/lm_eval --model local-completions \
  --model_args "model=GLM-5.2,base_url=http://0.0.0.0:8005/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=128,timeout=5000,max_length=8192" \
  --tasks gsm8k --num_fewshot 5 --output_path "$OUT"
