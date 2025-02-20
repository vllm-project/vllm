#!/usr/bin/env bash

MODEL=Snowflake/Llama-3.1-SwiftKV-405B-Instruct-FP8

EVAL_CMD=$(cat <<EOF
python -m lm_eval \
  --model vllm \
  --model_args pretrained=${MODEL},dtype=auto,max_model_len=4096,enable_chunked_prefill=True,tensor_parallel_size=8 \
  --gen_kwargs max_gen_toks=1024 \
  --batch_size auto \
  --output_path ./swiftkv-eval
EOF
)

${EVAL_CMD} \
  --tasks truthfulqa_mc2 \
  --num_fewshot 0

${EVAL_CMD} \
  --tasks winogrande \
  --num_fewshot 5

${EVAL_CMD} \
  --tasks hellaswag \
  --num_fewshot 10

${EVAL_CMD} \
  --tasks arc_challenge_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0

${EVAL_CMD} \
  --tasks gsm8k_cot_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 8

${EVAL_CMD} \
  --tasks mmlu_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5

${EVAL_CMD} \
  --tasks mmlu_cot_0shot_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0
