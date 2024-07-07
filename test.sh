TP_SIZE=4
FEWSHOT=5
LIMIT=250
BATCH_SIZE="auto"

# # Mixtral
# MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
# lm_eval --model vllm \
#   --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray" \
#   --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT --batch_size $BATCH_SIZE

# # Mixtral - Quantized in place
# MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
# lm_eval --model vllm \
#   --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray",quantization="fp8" \
#   --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT --batch_size $BATCH_SIZE

# # Mixtral - Quantized
# MODEL="neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8"
# lm_eval --model vllm \
#   --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray",quantization="fp8" \
#   --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT --batch_size $BATCH_SIZE

# Qwen
MODEL="Qwen/Qwen2-57B-A14B-Instruct"
lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray" \
  --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT --batch_size $BATCH_SIZE

# Qwen - Quantized in place
MODEL="Qwen/Qwen2-57B-A14B-Instruct"
lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray",quantization="fp8" \
  --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT --batch_size $BATCH_SIZE

# Qwen - Quantized
# MODEL="nm-testing/Qwen2-57B-A14B-Instruct-FP8-KV"
# lm_eval --model vllm \
#   --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray",quantization="fp8" \
#   --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT --batch_size $BATCH_SIZE

# Llama3
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray" \
  --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT --batch_size $BATCH_SIZE

# Llama-3 Quantized in place
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray",quantization="fp8" \
  --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT --batch_size $BATCH_SIZE

# Llama-3 - Quantized
MODEL="neuralmagic/Meta-Llama-3-8B-Instruct-FP8"
lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,distributed_executor_backend="ray",quantization="fp8" \
  --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT --batch_size $BATCH_SIZE