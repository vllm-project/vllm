#!/usr/bin/env bash
# Ornith XPU graphs canary — single-arm serve + correctness smokes (S1-S6)
# + optional perf bench. Run on an XPU host with this branch's vllm.
# ARM=A (eager) | B (PIECEWISE) | C (FA-in-graph FULL). Never touches the
# production DaemonSet. On exit the server is always killed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM="${ARM:-A}"
MODEL="${MODEL:-/models/Ornith-1.0-35B-MXFP4}"
SERVED="${SERVED:-ornith-canary}"
PORT="${PORT:-8021}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
GPU_UTIL="${GPU_UTIL:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
PERF="${PERF:-1}"
N_ITERS="${N_ITERS:-8}"
GEN_TOKENS="${GEN_TOKENS:-128}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/deploy/xpu-ornith-graphs-canary/results}"
mkdir -p "${RESULTS_DIR}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SERVE_LOG="${RESULTS_DIR}/serve_${ARM}_${STAMP}.log"
SMOKE_LOG="${RESULTS_DIR}/smoke_${ARM}_${STAMP}.log"
OUT_JSON="${RESULTS_DIR}/outputs_${ARM}_${STAMP}.json"
ARM_JSON="${RESULTS_DIR}/arm_${ARM}_${STAMP}.json"

case "${ARM}" in
  A)
    export VLLM_XPU_ENABLE_XPU_GRAPH=0
    ARM_ARGS="--enforce-eager"
    ;;
  B)
    export VLLM_XPU_ENABLE_XPU_GRAPH=1
    export VLLM_XPU_GRAPH_FORCE_PIECEWISE=1
    ARM_ARGS="--attention-backend FLASH_ATTN -cc.cudagraph_mode=PIECEWISE"
    ;;
  C)
    export VLLM_XPU_ENABLE_XPU_GRAPH=1
    export VLLM_XPU_GRAPH_FORCE_PIECEWISE=0
    ARM_ARGS="--attention-backend FLASH_ATTN -cc.cudagraph_mode=FULL"
    ;;
  *) echo "unknown ARM=${ARM}" >&2; exit 2 ;;
esac
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"

VLLM_BIN="${VLLM_BIN:-$(command -v vllm)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

echo "=== Ornith smoke ARM=${ARM} MODEL=${MODEL} PORT=${PORT} ===" | tee "${SMOKE_LOG}"
env | grep -E "VLLM_XPU|ZE_AFFINITY" | tee -a "${SMOKE_LOG}"

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

T_START=$(date +%s)
# shellcheck disable=SC2086
nohup "${VLLM_BIN}" serve "${MODEL}" \
  --host "${HOST}" --port "${PORT}" \
  --served-model-name "${SERVED}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --dtype "${DTYPE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":0}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  ${ARM_ARGS} >"${SERVE_LOG}" 2>&1 &
SERVE_PID=$!
echo "SERVE_PID=${SERVE_PID}" | tee -a "${SMOKE_LOG}"

cleanup() {
  kill "${SERVE_PID}" 2>/dev/null || true
  wait "${SERVE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# --- S1: startup ---
ready=0
for i in $(seq 1 $((READY_TIMEOUT / 5))); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    READY_S=$(($(date +%s) - T_START))
    echo "S1 READY after ${READY_S}s" | tee -a "${SMOKE_LOG}"
    ready=1
    break
  fi
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "S1 FAIL: SERVE_DIED" | tee -a "${SMOKE_LOG}"
    tail -100 "${SERVE_LOG}" | tee -a "${SMOKE_LOG}"
    exit 1
  fi
  sleep 5
done
if [[ "${ready}" -ne 1 ]]; then
  echo "S1 FAIL: NOT_READY after ${READY_TIMEOUT}s" | tee -a "${SMOKE_LOG}"
  tail -100 "${SERVE_LOG}" | tee -a "${SMOKE_LOG}"
  exit 1
fi

# --- S1: log hygiene (the exact historical failure is a hard fail) ---
if grep -E "work_group_scratch_memory|not yet available for use with the SYCL Graph" \
  "${SERVE_LOG}"; then
  echo "S1 FAIL: SYCL Graph scratch error in serve log" | tee -a "${SMOKE_LOG}"
  exit 1
fi
if [[ "${ARM}" == "C" ]] && ! grep -q "FlashAttention-in-graph enabled" "${SERVE_LOG}"; then
  echo "S1 FAIL: arm C without 'FlashAttention-in-graph enabled' (clamped?)" \
    | tee -a "${SMOKE_LOG}"
  exit 1
fi
grep -E "FlashAttention-in-graph|cudagraph_mode|XPU Graph|FORCE_PIECEWISE|Capturing|cudagraph" \
  "${SERVE_LOG}" | head -40 | tee -a "${SMOKE_LOG}" || true

# --- S2-S6 correctness smokes ---
BASE="http://${HOST}:${PORT}/v1"
"${PYTHON_BIN}" - "${ARM}" "${BASE}" "${SERVED}" "${OUT_JSON}" <<'PY' | tee -a "${SMOKE_LOG}"
import json, re, sys, urllib.request

arm, base, model, out_path = sys.argv[1:5]

def chat(prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]

def garbage(text):
    if not text.strip():
        return "empty"
    if re.search(r"!{4,}", text):
        return "!-loop"
    if "nan" in text.lower().split() or set(text.strip()) <= {"!", ".", " "}:
        return "nan/garbage"
    return None

ok = True
outputs = {}

def record(name, text, fail=None):
    global ok
    outputs[name] = text
    g = garbage(text)
    if g:
        print(f"{name} FAIL garbage: {g}"); ok = False
    elif fail:
        print(f"{name} FAIL: {fail}"); ok = False
    else:
        print(f"{name} OK ({len(text)} chars): {text[:120]!r}")

# S2 short + S4 determinism
p2 = "What is 2+2? Reply with only the number."
t1 = chat(p2, 32)
t2 = chat(p2, 32)
record("S2_short", t1, None if "4" in t1 else "expected 4")
if t1 != t2:
    print("S4 FAIL: temp=0 short repeat mismatch"); ok = False
else:
    print("S4a OK: short repeat byte-identical")

# S3 long + repeat
p3 = ("Write a detailed, structured explanation of how ocean tides work, "
      "covering gravity, the moon, spring and neap tides.")
l1 = chat(p3, 512)
l2 = chat(p3, 512)
record("S3_long", l1, None if len(l1.strip()) >= 200 else "too short")
if l1 != l2:
    print("S4 FAIL: temp=0 long repeat mismatch"); ok = False
else:
    print("S4b OK: long repeat byte-identical")

# S5 MoE routing stress (distinct expert domains)
s5 = {
    "S5_code": "Write a Python function that reverses a singly linked list.",
    "S5_math": "Compute 17*23 step by step, then state the final answer.",
    "S5_prose": "Describe an autumn morning in exactly two sentences.",
    "S5_multilingual": (
        "Translate 'The weather is nice today' into French and German."),
}
for name, prompt in s5.items():
    record(name, chat(prompt, 192))

# S6 GDN/Mamba state stress: plant facts, bury them under ~2k tokens of
# filler, require exact recall of the secret.
secret = "MAGNETIC-YELLOW-42"
owner = "Dr. Elena Vasquez"
filler = " ".join(
    f"Log entry {i}: routine sensor sweep {i} completed with no anomalies "
    f"detected in sector {i % 7}." for i in range(220))
p6 = (
    f"Remember these facts: the secret code is {secret} and it belongs to "
    f"{owner}. Now read this log:\n{filler}\n"
    "Question: What is the secret code and who does it belong to? "
    "Answer in one sentence.")
a6 = chat(p6, 96)
fail6 = None
if secret not in a6:
    fail6 = f"secret code not recalled (got: {a6[:200]!r})"
elif "Vasquez" not in a6:
    fail6 = f"owner not recalled (got: {a6[:200]!r})"
record("S6_state_recall", a6, fail6)

json.dump({"arm": arm, "outputs": outputs}, open(out_path, "w"), indent=2)
print("SMOKE_OK" if ok else "SMOKE_FAIL")
raise SystemExit(0 if ok else 1)
PY

# --- perf bench (single stream, greedy) ---
if [[ "${PERF}" == "1" ]]; then
  "${PYTHON_BIN}" - "${ARM}" "${BASE}" "${SERVED}" "${ARM_JSON}" \
    "${N_ITERS}" "${GEN_TOKENS}" "${READY_S}" <<'PY' | tee -a "${SMOKE_LOG}"
import json, sys, time, urllib.request

arm, base, model, out_path = sys.argv[1:5]
n_iters, gen_tokens, ready_s = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
prompt = "Explain what a GPU does in two sentences."

def stream(maxt):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": maxt,
                       "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(f"{base}/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; toks = 0; txt = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            ch = j["choices"][0].get("text", "")
            if ch:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                toks += 1; txt += ch
    dur = time.perf_counter() - t0
    dec = (toks - 1) / (dur - ttft) if ttft and toks > 1 and dur > ttft else 0.0
    return ttft or 0.0, dec, txt

stream(16)  # warmup
ttfts, decs, texts = [], [], []
for _ in range(n_iters):
    a, b, c = stream(gen_tokens)
    ttfts.append(a); decs.append(b); texts.append(c)

def med(x):
    x = sorted(x); n = len(x)
    return x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2

out = {"arm": arm, "ready_s": ready_s,
       "ttft_ms_mean": sum(ttfts) / len(ttfts) * 1000,
       "ttft_ms_p50": med(ttfts) * 1000,
       "decode_tok_s_mean": sum(decs) / len(decs),
       "decode_tok_s_p50": med(decs),
       "text": texts[0], "text_stable": len(set(texts)) == 1}
json.dump(out, open(out_path, "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "text"}, indent=2))
PY
fi

echo "PASS ARM=${ARM} outputs=${OUT_JSON} smoke=${SMOKE_LOG}" | tee -a "${SMOKE_LOG}"
