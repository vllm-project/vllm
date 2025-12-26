#!/bin/bash
set -euo pipefail

export MODEL="meta-llama/Llama-3.1-405B-Instruct"
#export MODEL="meta-llama/Llama-3.1-70B-Instruct"
#export MODEL="meta-llama/Llama-3.1-8B-Instruct"
#export MODEL="Qwen/Qwen2.5-14B-Instruct"
#export NUM_PROMPTS=64
#export CONCURRENT_REQ=64
#DOCKER_IMAGE="vllm-gaudi-cd:latest"
#"vault.habana.ai/gaudi-docker/1.22.0/ubuntu22.04/habanalabs/vllm-installer-2.7.1:latest"

# Define input/output token pairs
#declare -a TOK_PAIRS=(
#  "2048:2048"
#  "4096:128"
#  "128:4096"
#)
declare -a TOK_PAIRS=(
  "2048:2048"
)

# Define CPU counts
#CPU_LIST=(6 12 18 24 144)
CPU_LIST=(4 8 16 24 128)

for pair in "${TOK_PAIRS[@]}"; do
	  IFS=":" read -r INPUT_TOK OUTPUT_TOK <<< "$pair"
  export INPUT_TOK OUTPUT_TOK

  echo "=== Running benchmarks for INPUT_TOK=$INPUT_TOK, OUTPUT_TOK=$OUTPUT_TOK ==="

  for NUMCPU in "${CPU_LIST[@]}"; do
    export NUM_CPUS="$NUMCPU"
    echo "--- Generating CPU binding for NUMCPU=$NUMCPU ---"

    python3 generate_cpu_binding_from_csv.py \
      --output ./docker-compose.override.yml

    echo "--- Launching benchmark (NUMCPU=$NUMCPU) ---"
    docker compose up \
      --abort-on-container-exit

    LOG_SRC="logs/perftest_inp${INPUT_TOK}_out${OUTPUT_TOK}.log"
    LOG_DST="logs/perftest_inp${INPUT_TOK}_out${OUTPUT_TOK}_cpu${NUMCPU}.log"

    if [[ -f "$LOG_SRC" ]]; then
      mv "$LOG_SRC" "$LOG_DST"
      echo "Renamed log: $LOG_DST"
    else
      echo "Warning: expected log $LOG_SRC not found!"
    fi
    echo "---Closing benchmark (NUMCPU=$NUMCPU) ---"
    docker compose down

    echo "--- Completed NUMCPU=$NUMCPU ---"
  done
done

echo "✅ All benchmarks completed."

