#!/bin/bash

# MODEL="neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8"
# MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL="Qwen/Qwen2-57B-A14B-Instruct"
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"

lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=2,distributed_executor_backend="ray",enforce_eager=True,quantization="fp8" \
  --tasks gsm8k --num_fewshot 5 --limit 250 --batch_size "auto"
