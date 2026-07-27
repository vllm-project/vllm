#!/usr/bin/env bash
# FA-in-graph canary smoke (FLASH_ATTN + FULL XPU Graph).
# Run on an XPU host (oneAPI 2026.0+) with vllm installed. Does not touch
# Ornith production. Rollback: VLLM_XPU_GRAPH_FORCE_PIECEWISE=1 or eager.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED="${SERVED:-fa-graph}"
PORT="${PORT:-8020}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"
CG_MODE="${CG_MODE:-FULL}"
DTYPE="${DTYPE:-bfloat16}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/deploy/xpu-fa-graph-canary/results}"
mkdir -p "${RESULTS_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SERVE_LOG="${RESULTS_DIR}/serve_${STAMP}.log"
SMOKE_LOG="${RESULTS_DIR}/smoke_${STAMP}.log"

export VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}"
export VLLM_XPU_GRAPH_FORCE_PIECEWISE="${VLLM_XPU_GRAPH_FORCE_PIECEWISE:-0}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"

VLLM_BIN="${VLLM_BIN:-$(command -v vllm)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

echo "=== FA-in-graph smoke MODEL=${MODEL} PORT=${PORT} CG=${CG_MODE} ===" | tee "${SMOKE_LOG}"
echo "VLLM_XPU_ENABLE_XPU_GRAPH=${VLLM_XPU_ENABLE_XPU_GRAPH}" | tee -a "${SMOKE_LOG}"
echo "VLLM_XPU_GRAPH_FORCE_PIECEWISE=${VLLM_XPU_GRAPH_FORCE_PIECEWISE}" | tee -a "${SMOKE_LOG}"

"${PYTHON_BIN}" - <<'PY' | tee -a "${SMOKE_LOG}"
import torch
from vllm.utils.torch_utils import supports_xpu_graph, supports_xpu_fa_in_graph
print("torch", torch.__version__)
print("xpu_version", getattr(torch.version, "xpu", None))
print("supports_xpu_graph", supports_xpu_graph())
print("supports_xpu_fa_in_graph", supports_xpu_fa_in_graph())
if not supports_xpu_fa_in_graph():
    raise SystemExit("FAIL: supports_xpu_fa_in_graph() is False (need oneAPI 2026.0+)")
PY

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

# shellcheck disable=SC2086
nohup "${VLLM_BIN}" serve "${MODEL}" \
  --host "${HOST}" --port "${PORT}" \
  --served-model-name "${SERVED}" \
  --attention-backend FLASH_ATTN \
  -cc.cudagraph_mode="${CG_MODE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --dtype "${DTYPE}" \
  >"${SERVE_LOG}" 2>&1 &
SERVE_PID=$!
echo "SERVE_PID=${SERVE_PID}" | tee -a "${SMOKE_LOG}"

cleanup() {
  kill "${SERVE_PID}" 2>/dev/null || true
  wait "${SERVE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for i in $(seq 1 180); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "READY after ${i}s" | tee -a "${SMOKE_LOG}"
    ready=1
    break
  fi
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "SERVE_DIED" | tee -a "${SMOKE_LOG}"
    tail -80 "${SERVE_LOG}" | tee -a "${SMOKE_LOG}"
    exit 1
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "NOT_READY" | tee -a "${SMOKE_LOG}"
  tail -80 "${SERVE_LOG}" | tee -a "${SMOKE_LOG}"
  exit 1
fi

# Log hygiene: the exact failure this feature must not regress.
if grep -E "work_group_scratch_memory|not yet available for use with the SYCL Graph" \
  "${SERVE_LOG}"; then
  echo "FAIL: SYCL Graph scratch error in serve log" | tee -a "${SMOKE_LOG}"
  exit 1
fi
# Confirm FA-in-graph actually engaged (not silently clamped to PIECEWISE).
if ! grep -Eq "FlashAttention-in-graph enabled" "${SERVE_LOG}"; then
  echo "WARN: 'FlashAttention-in-graph enabled' not found; capture may have been clamped" \
    | tee -a "${SMOKE_LOG}"
fi
grep -E "Flash Attention|FlashAttention-in-graph|cudagraph_mode|XPU Graph|FORCE_PIECEWISE|PIECEWISE|FULL" \
  "${SERVE_LOG}" | head -40 | tee -a "${SMOKE_LOG}" || true

BASE="http://${HOST}:${PORT}/v1"
"${PYTHON_BIN}" - <<PY | tee -a "${SMOKE_LOG}"
import json, re, urllib.request

BASE = "${BASE}"
MODEL = "${SERVED}"

def chat(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"], \
        data["choices"][0].get("finish_reason")

ok = True
prompt = "What is 2+2? Reply with only the number."
t1, f1 = chat(prompt, 32)
t2, f2 = chat(prompt, 32)
print("=== SHORT1 ===", repr(t1), f1)
print("=== SHORT2 ===", repr(t2), f2)
if not t1.strip() or "4" not in t1:
    print("FAIL short answer"); ok = False
if t1.strip() != t2.strip():
    print("FAIL temp0 mismatch"); ok = False

long_prompt = "Write three short sentences about the ocean."
tl, fl = chat(long_prompt, 256)
print("=== LONG ===", repr(tl[:400]), fl)
if len(tl.strip()) < 20:
    print("FAIL long too short"); ok = False
if re.search(r"!{4,}", tl) or set(tl.strip()) <= {"!", ".", " "}:
    print("FAIL garbage/! loop"); ok = False

print("SMOKE_OK" if ok else "SMOKE_FAIL")
raise SystemExit(0 if ok else 1)
PY

echo "PASS logs=${SMOKE_LOG} serve=${SERVE_LOG}" | tee -a "${SMOKE_LOG}"
