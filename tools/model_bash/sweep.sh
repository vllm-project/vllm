#!/usr/bin/env bash
set -euo pipefail

# You can export these before running, or edit defaults here:
MODEL="${MODEL:-openai/gpt-oss-120b}"
INPUT_LEN="${INPUT_LEN:-2000}"
OUTPUT_LEN="${OUTPUT_LEN:-200}"

# Concurrency levels to test
concurrencies=(10 32 64 128 256)

for c in "${concurrencies[@]}"; do
  num_prompts=$((10 * c))
  seed="$(date +%s)"

  echo "=== Running: concurrency=${c}, num_prompts=${num_prompts}, seed=${seed} ==="

  vllm bench serve \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --max-concurrency "$c" \
    --num-prompts "$num_prompts" \
    --seed "$seed" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 90,95 \
    --ignore-eos
done
