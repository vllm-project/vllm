#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_CACHE="${HF_HUB_CACHE:-/models/huggingface_hub}"
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export HSA_NO_SCRATCH_RECLAIM=1
export SAFETENSORS_FAST_GPU=1

MODEL_PATH="${MODEL_PATH:-/models/huggingface_hub/}"

# Keep SHM transport enabled and stay below the GPU-IPC mapping ceiling seen
# with multi-terabyte pinned pools.
LMCACHE_L1_SHM_FRACTION="${LMCACHE_L1_SHM_FRACTION:-0.60}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-$(python - "$LMCACHE_L1_SHM_FRACTION" <<'PY'
import os
import sys

fraction = float(sys.argv[1])
stats = os.statvfs("/dev/shm")
free_gib = stats.f_bavail * stats.f_frsize / (1024**3)
print(max(1, int(free_gib * fraction)))
PY
)}"
echo "LMCache L1: ${LMCACHE_L1_SIZE_GB} GiB (${LMCACHE_L1_SHM_FRACTION} of free /dev/shm)"

# LMCache's ROCm GPU cache-registration path requires CuPy. Kimi-K3 also
# requires the unified Mamba and subpaged-MLA cache-view edits from 0.5.3.
python - <<'PY'
import cupy  # noqa: F401
from lmcache.integration.vllm import kv_cache_group_edits

edits = {type(edit).__name__ for edit in getattr(kv_cache_group_edits, "_EDITS", ())}
required = {"_MambaUnifiedViewEdit", "_SubpagedMLAAttentionViewEdit"}
missing = required - edits
if missing:
    raise SystemExit(f"LMCache lacks Kimi-K3 cache edits: {sorted(missing)}")
print(f"LMCache Kimi-K3 edits present: {sorted(required)}")
PY

cleanup() {
  trap - EXIT INT TERM
  kill "${vllm_pid:-}" "${lmcache_pid:-}" 2>/dev/null || true
  wait "${vllm_pid:-}" "${lmcache_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

lmcache server \
  --host 127.0.0.1 \
  --port 5555 \
  --http-host 127.0.0.1 \
  --http-port 8080 \
  --l1-size-gb "$LMCACHE_L1_SIZE_GB" \
  --l1-init-size-gb 20 \
  --l1-read-ttl-seconds 7200 \
  --chunk-size 1536 \
  --max-workers 8 \
  --eviction-trigger-watermark 0.85 \
  --eviction-ratio 0.10 \
  --eviction-policy LRU &
lmcache_pid=$!

python - "$lmcache_pid" <<'PY'
import os
import socket
import sys
import time

pid = int(sys.argv[1])
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("LMCache server exited before port 5555 became ready")
    try:
        with socket.create_connection(("127.0.0.1", 5555), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit("Timed out waiting for LMCache server on port 5555")
PY

vllm serve "$MODEL_PATH" \
  --served-model-name moonshotai/Kimi-K3 \
  --host 0.0.0.0 \
  --port 8888 \
  --trust-remote-code \
  --moe-backend auto \
  --tensor-parallel-size 8 \
  --load-format auto \
  --gpu-memory-utilization 0.85 \
  --mm-encoder-tp-mode data \
  --max-num-seqs 8 \
  --max-num-batched-tokens 3000 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":2,"method":"dspark","attention_backend":"TRITON_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5555}}' &
vllm_pid=$!

wait -n "$lmcache_pid" "$vllm_pid"
