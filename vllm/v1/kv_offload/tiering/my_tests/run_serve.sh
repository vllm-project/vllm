#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

MODEL="${MODEL:-/root/models/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8081}"
ASCEND_DEVICE="${ASCEND_DEVICE:-0}"
NPU_KV_BYTES="${NPU_KV_BYTES:-268435456}"
CPU_KV_BYTES="${CPU_KV_BYTES:-1073741824}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
FS_ROOT="${FS_ROOT:-/tmp/vllm_kv_tiering}"
LIFECYCLE_TTL="${LIFECYCLE_TTL:-10}"
DELETE_EXPIRED_SECONDARY="${DELETE_EXPIRED_SECONDARY:-false}"
VLLM_BIN="${VLLM_BIN:-/root/miniconda3/envs/vllm-hust-dev/bin/vllm-hust}"

EXTRA_ARGS=()
if [[ "${ENFORCE_EAGER:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

mkdir -p "${FS_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICE}"
export VLLM_ASCEND_TORCH_PREFLIGHT="${VLLM_ASCEND_TORCH_PREFLIGHT:-0}"
export VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP="${VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP:-1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${VLLM_BIN}" serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --generation-config vllm \
  --max-model-len "${MAX_MODEL_LEN}" \
  --kv-cache-memory-bytes "${NPU_KV_BYTES}" \
  --prefix-caching-hash-algo sha256 \
  "${EXTRA_ARGS[@]}" \
  --kv-transfer-config "{
    \"kv_connector\": \"OffloadingConnector\",
    \"kv_role\": \"kv_both\",
    \"kv_connector_extra_config\": {
      \"spec_name\": \"TieringOffloadingSpec\",
      \"cpu_bytes_to_use\": ${CPU_KV_BYTES},
      \"block_size\": 128,
      \"eviction_policy\": \"lru\",
      \"lifecycle_idle_ttl_sec\": ${LIFECYCLE_TTL},
      \"lifecycle_delete_expired_secondary\": ${DELETE_EXPIRED_SECONDARY},
      \"secondary_tiers\": [{
        \"type\": \"fs\",
        \"root_dir\": \"${FS_ROOT}\",
        \"n_read_threads\": 4,
        \"n_write_threads\": 4
      }]
    }
  }"
