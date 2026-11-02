#!/usr/bin/env bash
set -euo pipefail

# Core PD-disagg sweep sized for a 2-GPU pod so it can run on every PR; the
# full 4-GPU sweeps (TP2/TP2, FlashInfer, the wider DP-EP and SW sets,
# cross-layer, Mamba1) run nightly only (see .buildkite/test_areas/
# disaggregated.yaml). run_accuracy_test.sh spreads engines over the visible
# GPUs, so with 2 GPUs every TP>1 config co-locates prefill and decode ranks
# on the same GPUs. Shared configs cap GPU_MEMORY_UTILIZATION at 0.4 so two
# engines fit side by side; the TP1/TP1 configs keep one dedicated GPU per
# side at the utilization the nightly sweep uses for the same model.
SCRIPT="v1/kv_connector/nixl_integration/run_accuracy_test.sh"
IMPORT_CANARY="v1/kv_connector/nixl_integration/test_nixl_imports.py"

echo "=== Running NIXL import canary ==="
python3 -m pytest -s -x "${IMPORT_CANARY}"

# TP shapes and modes on the dense default model (Qwen3-0.6B). TP1/TP1 gets
# one dedicated GPU per side. The hetero-TP configs (P TP1/D TP2 and
# P TP2/D TP1) share one GPU between the prefiller and one decoder rank
# (0.35 each), as does PP prefill in pull and push modes (P TP1xPP2, D TP2,
# sharing both GPUs). DP-attention decode runs the nightly sweep's MoE model
# (deepseek-vl2-tiny, ~6.5 GB bf16): P TP1 and one D DP2 rank share GPU0, so
# it needs 0.4 per engine to keep room for KV and CUDA contexts.
dense_configs=(
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=1"
  "GPU_MEMORY_UTILIZATION=0.35 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2"
  "GPU_MEMORY_UTILIZATION=0.35 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1"
  "DP_EP=1 GPU_MEMORY_UTILIZATION=0.4 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny"
  "GPU_MEMORY_UTILIZATION=0.35 PREFILLER_PP_SIZE=2 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2"
  "KV_CONNECTOR=NixlPushConnector GPU_MEMORY_UTILIZATION=0.35 PREFILLER_PP_SIZE=2 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2"
)

# Engine modes on the dense default model, TP1/TP1 on dedicated GPUs: the
# sleep/wake round-trip, previously nightly only.
mode_configs=(
  "SLEEP_WAKE=1 GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=1"
)

# Architecture coverage at TP1/TP1 on dedicated GPUs: MLA, hybrid SSM
# conv-state transfer (Mamba2 and GDN), and sliding-window attention
# combined with cross-layer KV sharing (gemma-4-E2B-it declares
# num_kv_shared_layers in its HF config). Configs mirror the nightly
# sweeps; the gemma config was previously unreachable, as no CI step
# selects SW_ATTN.
arch_configs=(
  "GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny"
  "VLLM_SSM_CONV_STATE_LAYOUT=DS GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=ibm-granite/granite-4.0-h-tiny VLLM_SERVE_EXTRA_ARGS=--max-model-len,8192,--trust-remote-code"
  "VLLM_SSM_CONV_STATE_LAYOUT=DS GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=Qwen/Qwen3.5-0.8B"
  "GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=google/gemma-4-E2B-it VLLM_SERVE_EXTRA_ARGS=--max-model-len,8192"
)

run_tests() {
  local label=$1
  local extra_args=$2

  echo "=== Running tests (${label}) ==="
  for cfg in "${configs[@]}"; do
    local -a cfg_parts extra_args_parts
    read -r -a cfg_parts <<< "$cfg"
    read -r -a extra_args_parts <<< "$extra_args"

    echo "-> Running with ${cfg} ${extra_args:+and ${extra_args}}"
    # Use 'env' to safely set variables without eval
    # keep argv splitting safe and SC2086-clean via arrays.
    if ! env "${cfg_parts[@]}" bash "${SCRIPT}" "${extra_args_parts[@]}"; then
      echo "❌ Test failed for config: ${cfg} ${extra_args:+(${extra_args})}"
      exit 1
    fi
  done
  echo "✅ All ${label} tests passed!"
}

configs=("${dense_configs[@]}" "${mode_configs[@]}" "${arch_configs[@]}")
run_tests "default backend" ""
