#!/usr/bin/env bash
# One owner and independent vLLM P TP2 / D TP2 workers on four local GPUs.
set -euo pipefail

role=${1:?usage: launch.sh owner|prefill|decode}
: "${MODEL_DIR:?set MODEL_DIR to the local DeepSeek-V4.1-Flash checkpoint}"
: "${TOKEN_FILE:?set TOKEN_FILE to a private token file shared with consumers}"

python_bin=${VLLM_PYTHON:-python}
owner_endpoint=${OWNER_ENDPOINT:-127.0.0.1:56000}
owner_port=${owner_endpoint##*:}

run() {
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    exec "$@"
  fi
}

if [[ ! -r "$TOKEN_FILE" ]]; then
  echo "TOKEN_FILE is not readable: $TOKEN_FILE" >&2
  exit 2
fi

case "$role" in
  owner)
    run "$python_bin" -m vllm.distributed.engram_share.vmm.owner \
      --model-dir "$MODEL_DIR" \
      --listen-host "${OWNER_BIND_HOST:-127.0.0.1}" \
      --listen-port "$owner_port" \
      --token-file "$TOKEN_FILE" \
      --device "${OWNER_DEVICE:-0}"
    ;;
  prefill|decode)
    : "${GENERATION:?copy generation from ENGRAM_OWNER_READY line}"
    if [[ "$role" == prefill ]]; then
      export CUDA_VISIBLE_DEVICES=${P_GPUS:-0,1}
      export VLLM_PORT=${P_INTERNAL_PORT:-20000}
      export VLLM_NIXL_SIDE_CHANNEL_PORT=${P_NIXL_PORT:-5559}
      port=${P_PORT:-8100}
      kv_role=kv_producer
    else
      export CUDA_VISIBLE_DEVICES=${D_GPUS:-2,3}
      export VLLM_PORT=${D_INTERNAL_PORT:-30000}
      export VLLM_NIXL_SIDE_CHANNEL_PORT=${D_NIXL_PORT:-5659}
      port=${D_PORT:-8200}
      kv_role=kv_consumer
    fi
    export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
    export VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP:-skip}
    export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp}
    export UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
    engram_json=$(OWNER_ENDPOINT="$owner_endpoint" GENERATION="$GENERATION" \
      TOKEN_FILE="$TOKEN_FILE" "$python_bin" -c \
      'import json, os; print(json.dumps({"cpu_offload": True, "shared_host_endpoint": os.environ["OWNER_ENDPOINT"], "shared_host_token_file": os.environ["TOKEN_FILE"], "shared_host_generation": os.environ["GENERATION"]}))')
    kv_json=$(KV_ROLE="$kv_role" "$python_bin" -c \
      'import json, os; print(json.dumps({"kv_connector": "NixlConnector", "kv_role": os.environ["KV_ROLE"]}))')
    run "$python_bin" -m vllm.entrypoints.cli.main serve "$MODEL_DIR" \
      --host "${SERVE_HOST:-127.0.0.1}" --port "$port" \
      --served-model-name "${SERVED_MODEL:-deepseek-v41-shared}" \
      --language-model-only --tensor-parallel-size 2 --enforce-eager \
      --shutdown-timeout 120 --disable-prefix-caching \
      --max-model-len 32768 --max-num-seqs 2 --max-num-batched-tokens 8192 \
      --block-size 128 --kv-cache-memory-bytes 2147483648 --kv-cache-dtype fp8 \
      --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
      --kernel-config '{"enable_jit_warmup":false,"enable_cutedsl_warmup":false,"enable_flashinfer_autotune":false}' \
      --engram-config "$engram_json" --load-format safetensors \
      --safetensors-load-strategy lazy --kv-transfer-config "$kv_json"
    ;;
  *)
    echo "usage: launch.sh owner|prefill|decode" >&2
    exit 2
    ;;
esac
