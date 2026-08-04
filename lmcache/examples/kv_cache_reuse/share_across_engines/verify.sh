#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_VENV="${SGLANG_VENV:-${EXAMPLE_DIR}/.venv-sglang}"
VLLM_VENV="${VLLM_VENV:-${EXAMPLE_DIR}/.venv-vllm}"
MODEL="Qwen/Qwen2.5-32B-Instruct"
MODEL_REVISION="5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
LMCACHE_PORT=5556
LMCACHE_HTTP_PORT=7000
SGLANG_PORT=30000
VLLM_PORT=8000
WORK_DIR="$(mktemp -d -t lmcache-sglang-vllm-XXXXXX)"

LMCACHE_PID=""
SGLANG_PID=""
VLLM_PID=""

cleanup() {
    trap - EXIT INT TERM

    for pid in "${VLLM_PID}" "${SGLANG_PID}" "${LMCACHE_PID}"; do
        if [[ -n "${pid}" ]]; then
            kill -TERM -- "-${pid}" 2>/dev/null || true
        fi
    done

    for pid in "${VLLM_PID}" "${SGLANG_PID}" "${LMCACHE_PID}"; do
        if [[ -z "${pid}" ]]; then
            continue
        fi
        for _ in {1..150}; do
            if ! kill -0 -- "-${pid}" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        kill -KILL -- "-${pid}" 2>/dev/null || true
    done

    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! command -v setsid >/dev/null 2>&1; then
    echo "Missing executable: setsid" >&2
    exit 1
fi

for executable in \
    "${SGLANG_VENV}/bin/python" \
    "${VLLM_VENV}/bin/python" \
    "${VLLM_VENV}/bin/lmcache" \
    "${VLLM_VENV}/bin/vllm"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Missing executable: ${executable}" >&2
        exit 1
    fi
done

wait_for_health() {
    local name="$1" port="$2" pid="$3" log_file="$4" path="${5:-health}"
    for ((i = 0; i < 600; i++)); do
        if curl -fsS "http://127.0.0.1:${port}/${path}" >/dev/null 2>&1; then
            echo "${name} ready after ${i}s"
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "${name} exited during startup" >&2
            tail -n 100 "${log_file}" >&2
            return 1
        fi
        sleep 1
    done
    echo "Timed out waiting for ${name}" >&2
    tail -n 100 "${log_file}" >&2
    return 1
}

metric_sum() {
    local metric="$1"
    curl -fsS "http://127.0.0.1:${LMCACHE_HTTP_PORT}/metrics" |
        awk -v metric="${metric}" \
            'index($1, metric) == 1 && (length($1) == length(metric) || substr($1, length(metric) + 1, 1) == "{") { total += $NF } END { print total + 0 }'
}

echo "Starting LMCache"
setsid "${VLLM_VENV}/bin/lmcache" server \
    --host 127.0.0.1 \
    --port "${LMCACHE_PORT}" \
    --http-port "${LMCACHE_HTTP_PORT}" \
    --chunk-size 256 \
    --l1-size-gb 8 \
    --eviction-policy LRU \
    >"${WORK_DIR}/lmcache.log" 2>&1 &
LMCACHE_PID=$!
wait_for_health "LMCache" "${LMCACHE_HTTP_PORT}" "${LMCACHE_PID}" \
    "${WORK_DIR}/lmcache.log" "healthcheck"

echo "Starting SGLang on GPU 0"
CUDA_VISIBLE_DEVICES=0 FLASHINFER_USE_CUDA_NORM=1 \
    setsid "${SGLANG_VENV}/bin/python" -m sglang.launch_server \
    --model-path "${MODEL}" \
    --revision "${MODEL_REVISION}" \
    --host 127.0.0.1 \
    --port "${SGLANG_PORT}" \
    --tp 1 \
    --page-size 16 \
    --mem-fraction-static 0.8 \
    --disable-cuda-graph \
    --enable-lmcache \
    --lmcache-config-file "${EXAMPLE_DIR}/lmcache.yaml" \
    >"${WORK_DIR}/sglang.log" 2>&1 &
SGLANG_PID=$!

echo "Starting vLLM on GPU 1"
CUDA_VISIBLE_DEVICES=1 setsid "${VLLM_VENV}/bin/vllm" serve "${MODEL}" \
    --revision "${MODEL_REVISION}" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --block-size 16 \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --kv-transfer-config \
        '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5556}}' \
    >"${WORK_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

wait_for_health "SGLang" "${SGLANG_PORT}" "${SGLANG_PID}" \
    "${WORK_DIR}/sglang.log"
wait_for_health "vLLM" "${VLLM_PORT}" "${VLLM_PID}" \
    "${WORK_DIR}/vllm.log"

"${VLLM_VENV}/bin/python" - "${WORK_DIR}/request.json" <<'PY'
import json
import sys

sentence = (
    "LMCache lets independent inference engines reuse previously computed "
    "key and value tensors for a shared prompt while preserving the model output. "
)
request = {
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "prompt": sentence * 160 + "State the main idea in one sentence.",
    "max_tokens": 16,
    "temperature": 0,
}
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(request, output)
with open(sys.argv[1].replace("request.json", "cold-request.json"), "w", encoding="utf-8") as output:
    json.dump({**request, "cache_salt": "cold-reference"}, output)
PY

echo "Computing a cache-isolated vLLM reference"
COLD_READS_BEFORE="$(metric_sum lmcache_mp_l1_read_chunks_total)"
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    --data-binary "@${WORK_DIR}/cold-request.json" \
    >"${WORK_DIR}/vllm-cold-response.json"
COLD_READS_AFTER="$(metric_sum lmcache_mp_l1_read_chunks_total)"

echo "Populating LMCache from SGLang"
curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    --data-binary "@${WORK_DIR}/request.json" \
    >"${WORK_DIR}/sglang-response.json"

READS_BEFORE="$(metric_sum lmcache_mp_l1_read_chunks_total)"
echo "Reusing the SGLang KV cache from vLLM"
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    --data-binary "@${WORK_DIR}/request.json" \
    >"${WORK_DIR}/vllm-response.json"
READS_AFTER="$(metric_sum lmcache_mp_l1_read_chunks_total)"

"${VLLM_VENV}/bin/python" - \
    "${WORK_DIR}/sglang-response.json" \
    "${WORK_DIR}/vllm-cold-response.json" \
    "${WORK_DIR}/vllm-response.json" \
    "${COLD_READS_BEFORE}" "${COLD_READS_AFTER}" \
    "${READS_BEFORE}" "${READS_AFTER}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    sglang = json.load(source)
with open(sys.argv[2], encoding="utf-8") as source:
    vllm_cold = json.load(source)
with open(sys.argv[3], encoding="utf-8") as source:
    vllm = json.load(source)

cold_before, cold_after = int(float(sys.argv[4])), int(float(sys.argv[5]))
before, after = int(float(sys.argv[6])), int(float(sys.argv[7]))
cold_read_delta = cold_after - cold_before
read_delta = after - before
sglang_tokens = sglang["usage"]["prompt_tokens"]
vllm_cold_tokens = vllm_cold["usage"]["prompt_tokens"]
vllm_tokens = vllm["usage"]["prompt_tokens"]
vllm_cold_output = vllm_cold["choices"][0]["text"]
vllm_output = vllm["choices"][0]["text"]

if cold_read_delta != 0:
    raise SystemExit(
        f"FAIL: cache-isolated reference unexpectedly read {cold_read_delta} chunks"
    )
if read_delta <= 0:
    raise SystemExit(f"FAIL: LMCache L1 read delta is {read_delta}")
if sglang_tokens != vllm_tokens or vllm_cold_tokens != vllm_tokens:
    raise SystemExit(
        "FAIL: tokenizers disagree: "
        f"SGLang={sglang_tokens}, cold vLLM={vllm_cold_tokens}, "
        f"cache-hit vLLM={vllm_tokens}"
    )
if vllm_cold_output != vllm_output:
    raise SystemExit(
        f"FAIL: cold and cache-hit vLLM outputs disagree: "
        f"cold={vllm_cold_output!r}, cache-hit={vllm_output!r}"
    )

print("PASS")
print(f"prompt_tokens={sglang_tokens}")
print(f"lmcache_l1_read_chunk_delta={read_delta}")
print(f"generated_text={vllm_output!r}")
PY

echo "Logs: ${WORK_DIR}"
