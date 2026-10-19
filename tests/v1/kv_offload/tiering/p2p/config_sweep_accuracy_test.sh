#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run the p2p tier accuracy + transfer test across a matrix of deployments.
# One slice per CI step, selected by env var (mirrors
# tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh):
#   (default)  dense model: mixed TP, both handshake modes, multi-instance
#   MLA=1      MLA model with mixed TP (needs ~80 GB GPUs: V2-Lite is 16B)
#   DP_EP=1    data-parallel / expert-parallel prefillers and decoders
#   FEATURES=1 cache types beyond plain attention: hybrid Mamba/GDN state,
#              sliding window, fp8 KV
# Every config sizes for a single 4-GPU host.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT="${SCRIPT_DIR}/run_accuracy_test.sh"

DENSE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
# deepseek-vl2-tiny is MoE but not MLA (use_mla=false); the smallest real MLA
# model is DeepSeek-V2-Lite.
MOE_MODEL="deepseek-ai/deepseek-vl2-tiny"
MLA_MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat"
# Hybrid GatedDeltaNet + full attention: several KV cache groups plus SSM state.
HYBRID_MODEL="Qwen/Qwen3.5-0.8B"
# Sliding window of 512 tokens, shorter than the transfer and gsm8k prompts,
# so out-of-window blocks are actually dropped.
SWA_MODEL="google/gemma-3-1b-it"

# Mixed prefiller/decoder TP needs CANONICAL_LAYOUT=1: the direct host layout
# is not byte-identical across TP sizes, so the tier must either refuse the
# peer (default model runner) or, if it pairs them, hands over unusable KV.
tp_configs=(
  "P2P_MODE=pd  MODEL_NAMES=${DENSE_MODEL}"
  "P2P_MODE=p2p MODEL_NAMES=${DENSE_MODEL}"
  "P2P_MODE=pd  CANONICAL_LAYOUT=1 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=${DENSE_MODEL}"
  "P2P_MODE=p2p CANONICAL_LAYOUT=1 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1 MODEL_NAMES=${DENSE_MODEL}"
  # Direct layout with mixed TP must fall back to recompute until that layout
  # is byte-portable across TP; flip to a transfer check once it is.
  "P2P_MODE=pd  EXPECT_TRANSFER=0 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=${DENSE_MODEL}"
  "P2P_MODE=pd  NUM_PREFILL_INSTANCES=2 NUM_DECODE_INSTANCES=2 GPU_MEMORY_UTILIZATION=0.4 MODEL_NAMES=${DENSE_MODEL}"
)
# Decoder-first proxy ordering, run as a CLI flag rather than an env knob.
decoder_first_configs=(
  "P2P_MODE=pd  MODEL_NAMES=${DENSE_MODEL}"
)
mla_configs=(
  "P2P_MODE=pd  GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=${MLA_MODEL}"
  "P2P_MODE=p2p CANONICAL_LAYOUT=1 GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=${MLA_MODEL}"
  "P2P_MODE=pd  CANONICAL_LAYOUT=1 GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1 MODEL_NAMES=${MLA_MODEL}"
)
dp_ep_configs=(
  # P-TP1, D-DPEP=2 (TP=1)
  "DP_EP=1 P2P_MODE=pd  GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=${MOE_MODEL}"
  # P-TP2, D-DPEP=2 (TP=1)
  "DP_EP=1 P2P_MODE=p2p CANONICAL_LAYOUT=1 GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2 MODEL_NAMES=${MOE_MODEL}"
  # P-DP2 (dense replicas), D-TP1: the proxy pins each prefill to a replica
  "P2P_MODE=pd  PREFILLER_DP_SIZE=2 MODEL_NAMES=${DENSE_MODEL}"
)
feature_configs=(
  "P2P_MODE=p2p MODEL_NAMES=${HYBRID_MODEL}"
  "P2P_MODE=pd  MODEL_NAMES=${SWA_MODEL}"
  # fp8 pages through the canonical mixed-TP path that bf16 already covers.
  "P2P_MODE=pd  CANONICAL_LAYOUT=1 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 VLLM_SERVE_EXTRA_ARGS=--kv-cache-dtype,fp8 MODEL_NAMES=${DENSE_MODEL}"
)

run_tests() {
  local label=$1
  local extra_args=$2
  shift 2
  local -a configs=("$@")

  echo "=== Running tests (${label}) ==="
  for cfg in "${configs[@]}"; do
    local -a cfg_parts extra_args_parts
    read -r -a cfg_parts <<< "$cfg"
    read -r -a extra_args_parts <<< "$extra_args"

    echo "-> Running with ${cfg} ${extra_args:+and ${extra_args}}"
    # 'env' sets the knobs without eval; arrays keep argv splitting SC2086-clean.
    if ! env "${cfg_parts[@]}" bash "${SCRIPT}" "${extra_args_parts[@]}"; then
      echo "❌ Test failed for config: ${cfg} ${extra_args:+(${extra_args})}"
      exit 1
    fi
  done
  echo "✅ All ${label} tests passed!"
}

cmdline_args=""
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  echo "ATTENTION_BACKEND is set, running with --attention-backend ${ATTENTION_BACKEND}"
  cmdline_args="--attention-backend ${ATTENTION_BACKEND}"
fi

# The selector vars are consumed here; unset them so they do not leak into
# run_accuracy_test.sh (DP_EP is a knob there too, set per config).
if [[ -n "${MLA:-}" ]]; then
  unset MLA
  run_tests "MLA" "${cmdline_args}" "${mla_configs[@]}"
elif [[ -n "${DP_EP:-}" ]]; then
  unset DP_EP
  run_tests "DP/EP" "${cmdline_args}" "${dp_ep_configs[@]}"
elif [[ -n "${FEATURES:-}" ]]; then
  unset FEATURES
  run_tests "features" "${cmdline_args}" "${feature_configs[@]}"
else
  run_tests "TP" "${cmdline_args}" "${tp_configs[@]}"
  run_tests "decoder-first" "${cmdline_args} --decoder-first" \
    "${decoder_first_configs[@]}"
fi
