#!/usr/bin/env bash
# A/B: eager baseline (A) vs FlashAttention-in-graph FULL (B) on a small dense
# model. Single-stream latency + decode tok/s, greedy; asserts identical output
# text A vs B. Writes AB_COMPARE_<stamp>.{json,md} under results/. XPU host only;
# does not touch Ornith production.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED="${SERVED:-ab}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8021}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_UTIL="${GPU_UTIL:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
N_ITERS="${N_ITERS:-8}"
GEN_TOKENS="${GEN_TOKENS:-128}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/deploy/xpu-fa-graph-canary/results}"
mkdir -p "${RESULTS_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

VLLM_BIN="${VLLM_BIN:-$(command -v vllm)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"

run_arm() { # name  extra_env  extra_args
  local name="$1"; shift
  local serve_log="${RESULTS_DIR}/serve_${name}_${STAMP}.log"
  echo "[${name}] starting serve" >&2
  if command -v fuser >/dev/null 2>&1; then fuser -k "${PORT}/tcp" 2>/dev/null || true; fi
  # shellcheck disable=SC2086
  env ${ARM_ENV} nohup "${VLLM_BIN}" serve "${MODEL}" \
    --host "${HOST}" --port "${PORT}" --served-model-name "${SERVED}" \
    --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_UTIL}" \
    --dtype "${DTYPE}" ${ARM_ARGS} >"${serve_log}" 2>&1 &
  SERVE_PID=$!
  for i in $(seq 1 180); do
    curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${SERVE_PID}" 2>/dev/null || { echo "[${name}] serve died"; tail -60 "${serve_log}"; return 1; }
    sleep 2
  done
  if grep -Eq "work_group_scratch_memory|not yet available for use with the SYCL Graph" "${serve_log}"; then
    echo "[${name}] FAIL scratch-in-graph error"; kill "${SERVE_PID}" 2>/dev/null || true; return 1
  fi
  "${PYTHON_BIN}" - "$name" <<PY
import json, sys, time, urllib.request
name=sys.argv[1]
BASE="http://${HOST}:${PORT}/v1"; MODEL="${SERVED}"
N=${N_ITERS}; MAXT=${GEN_TOKENS}
prompt="Explain what a GPU does in two sentences."
def stream(maxt):
    body=json.dumps({"model":MODEL,"prompt":prompt,"max_tokens":maxt,
        "temperature":0.0,"stream":True}).encode()
    req=urllib.request.Request(f"{BASE}/completions",data=body,
        headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); ttft=None; toks=0; txt=""
    with urllib.request.urlopen(req,timeout=300) as r:
        for raw in r:
            line=raw.decode().strip()
            if not line.startswith("data:"): continue
            data=line[5:].strip()
            if data=="[DONE]": break
            j=json.loads(data); ch=j["choices"][0].get("text","")
            if ch:
                if ttft is None: ttft=time.perf_counter()-t0
                toks+=1; txt+=ch
    dur=time.perf_counter()-t0
    dec=(toks-1)/(dur-ttft) if ttft and toks>1 and dur>ttft else 0.0
    return ttft or 0.0, dec, txt
stream(16)  # warmup
ttfts=[]; decs=[]; texts=[]
for _ in range(N):
    a,b,c=stream(MAXT); ttfts.append(a); decs.append(b); texts.append(c)
def med(x): x=sorted(x); n=len(x); return (x[n//2] if n%2 else (x[n//2-1]+x[n//2])/2)
out={"arm":name,"ttft_ms_mean":sum(ttfts)/len(ttfts)*1000,
     "ttft_ms_p50":med(ttfts)*1000,"decode_tok_s_mean":sum(decs)/len(decs),
     "decode_tok_s_p50":med(decs),"text":texts[0],
     "text_stable":len(set(texts))==1}
json.dump(out,open("${RESULTS_DIR}/arm_"+name+"_${STAMP}.json","w"),indent=2)
print(json.dumps(out,indent=2))
PY
  kill "${SERVE_PID}" 2>/dev/null || true; wait "${SERVE_PID}" 2>/dev/null || true
  sleep 3
}

ARM_ENV="VLLM_XPU_ENABLE_XPU_GRAPH=0"; ARM_ARGS="--enforce-eager"
run_arm A

ARM_ENV="VLLM_XPU_ENABLE_XPU_GRAPH=1 VLLM_XPU_GRAPH_FORCE_PIECEWISE=0"
ARM_ARGS="--attention-backend FLASH_ATTN -cc.cudagraph_mode=FULL"
run_arm B

"${PYTHON_BIN}" - <<PY
import json
A=json.load(open("${RESULTS_DIR}/arm_A_${STAMP}.json"))
B=json.load(open("${RESULTS_DIR}/arm_B_${STAMP}.json"))
same=A["text"].strip()==B["text"].strip()
def dpct(a,b): return (b-a)/a*100 if a else 0.0
summary={"stamp":"${STAMP}","model":"${MODEL}",
  "A_eager":A,"B_fa_in_graph":B,
  "text_identical_A_vs_B":same,
  "ttft_delta_pct":dpct(A["ttft_ms_mean"],B["ttft_ms_mean"]),
  "decode_tok_s_delta_pct":dpct(A["decode_tok_s_mean"],B["decode_tok_s_mean"])}
json.dump(summary,open("${RESULTS_DIR}/AB_COMPARE_${STAMP}.json","w"),indent=2)
md=f'''# FA-in-graph A/B — ${STAMP}

**Model:** `${MODEL}`  (greedy, {A["ttft_ms_mean"]:.0f} vs {B["ttft_ms_mean"]:.0f} ms TTFT)

| Arm | TTFT ms mean | TTFT ms p50 | Decode tok/s mean | Decode tok/s p50 |
| --- | --- | --- | --- | --- |
| A: eager (graphs off) | {A["ttft_ms_mean"]:.1f} | {A["ttft_ms_p50"]:.1f} | {A["decode_tok_s_mean"]:.1f} | {A["decode_tok_s_p50"]:.1f} |
| B: FA-in-graph FULL | {B["ttft_ms_mean"]:.1f} | {B["ttft_ms_p50"]:.1f} | {B["decode_tok_s_mean"]:.1f} | {B["decode_tok_s_p50"]:.1f} |

**Decode tok/s delta (B vs A):** {summary["decode_tok_s_delta_pct"]:+.1f}%
**TTFT delta (B vs A):** {summary["ttft_delta_pct"]:+.1f}%
**Output text identical A vs B:** {same}
'''
open("${RESULTS_DIR}/AB_COMPARE_${STAMP}.md","w").write(md)
print(md)
PY
echo "AB done: ${RESULTS_DIR}/AB_COMPARE_${STAMP}.md"
