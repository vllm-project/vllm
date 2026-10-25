#!/usr/bin/env bash
# QOS / AgentX sweep for MiniMax-M3.1-Flash on amd-agentx-4-qos.
#
# Expected allocation: 1x MI350X/MI355X node, 4 GPUs (TP4).
# Claim only MXFP4 experts + FP8 KV/indexer in published results.
#
# Stages:
#   1. serve the official checkpoint with the AMD recipe
#   2. model-load / generation smoke
#   3. long-context indexer/KV check
#   4. Prometheus memory snapshot
#   5. AgentX TP4 concurrency sweep (c1 c4 c8 c16 c32 c64)
#   6. optional accuracy compare against REFERENCE_URL
set -euo pipefail

ACCOUNT="${ACCOUNT:-amd-agentx}"
QOS="${QOS:-amd-agentx-4-qos}"
PARTITION="${PARTITION:-amd-spur}"
TIME_LIMIT="${TIME_LIMIT:-4:00:00}"
MODEL="${MODEL:-MiniMaxAI/MiniMax-M3.1-Flash}"
CONCURRENCIES="${CONCURRENCIES:-1 4 8 16 32 64}"
SERVER_PORT="${SERVER_PORT:-8000}"
BENCHMARK_DURATION="${BENCHMARK_DURATION:-180}"
LONG_CONTEXT_TOKENS="${LONG_CONTEXT_TOKENS:-8192}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/minimax-m31-flash-rocm}"
IMAGE="${IMAGE:-}"
REFERENCE_URL="${REFERENCE_URL:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_SCRIPT="${SCRIPT_DIR}/minimax_m31_flash_rocm.sh"
VALIDATE_SCRIPT="${SCRIPT_DIR}/minimax_m31_flash_rocm_validate.py"

mkdir -p "${OUTPUT_DIR}"

log() {
  printf '[minimax-m31-qos] %s\n' "$*"
}

write_summary_header() {
  cat > "${OUTPUT_DIR}/SUMMARY.md" <<EOF
# MiniMax-M3.1-Flash AMD QOS

- Served precision: official NVFP4 checkpoint, routed experts requantized
  NVFP4→MXFP4 at load, FP8 main KV + FP8 indexer.
- Not native NVFP4 serving.
- Model: \`${MODEL}\`
- Recipe: \`${SERVE_SCRIPT}\`
- Image: \`${IMAGE:-local}\`
- Concurrencies: \`${CONCURRENCIES}\`
- DSpark: nested \`{model}/dspark\`, \`num_speculative_tokens=3\`, block size 128
- Monokernel / DBO: disabled

EOF
}

run_validate() {
  python3 "${VALIDATE_SCRIPT}" \
    --base-url "http://127.0.0.1:${SERVER_PORT}/v1" \
    --output "${OUTPUT_DIR}/validate.json" \
    --long-context-tokens "${LONG_CONTEXT_TOKENS}"
}

run_accuracy() {
  if [[ -z "${REFERENCE_URL}" ]]; then
    log "REFERENCE_URL unset; recording this server's smoke as the candidate only."
    return 0
  fi
  python3 - "${REFERENCE_URL}" "http://127.0.0.1:${SERVER_PORT}/v1" "${OUTPUT_DIR}/accuracy.json" <<'PY'
import json, sys, urllib.request

ref_base, cand_base, out = sys.argv[1:]
prompt = "Write a one-sentence description of sparse mixture-of-experts routing."

def complete(base):
    payload = json.dumps({
        "model": json.loads(urllib.request.urlopen(base + "/models").read())["data"][0]["id"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"]

result = {
    "served_precision": "mxfp4_experts_fp8_kv_indexer",
    "prompt": prompt,
    "reference": complete(ref_base.rstrip("/")),
    "candidate": complete(cand_base.rstrip("/")),
}
result["exact_match"] = result["reference"] == result["candidate"]
open(out, "w", encoding="utf-8").write(json.dumps(result, indent=2))
print("wrote", out, "exact_match", result["exact_match"])
PY
}

run_agentx() {
  local conc="$1"
  local art="${OUTPUT_DIR}/c${conc}"
  mkdir -p "${art}"
  aiperf profile \
    --scenario inferencex-agentx-mvp \
    --url "http://127.0.0.1:${SERVER_PORT}" \
    --endpoint /v1/chat/completions \
    --endpoint-type chat \
    --streaming \
    --model "${MODEL}" \
    --tokenizer "${MODEL}" \
    --tokenizer-trust-remote-code \
    --apply-chat-template \
    --concurrency "${conc}" \
    --benchmark-duration "${BENCHMARK_DURATION}" \
    --random-seed 42 \
    --failed-request-threshold 0.10 \
    --use-server-token-count \
    --no-gpu-telemetry \
    --public-dataset semianalysis_cc_traces_weka_062126 \
    --output-artifact-dir "${art}"
}

run_suite() {
  write_summary_header
  if [[ -n "${IMAGE}" ]]; then
    log "Using container image: ${IMAGE}"
  fi
  log "Starting MiniMax-M3.1-Flash TP4 server (MXFP4 experts, FP8 KV/indexer)."
  bash "${SERVE_SCRIPT}" --port "${SERVER_PORT}" &
  local server_pid=$!
  trap 'kill ${server_pid} 2>/dev/null || true' EXIT
  run_validate
  run_accuracy
  for conc in ${CONCURRENCIES}; do
    log "AgentX concurrency=${conc}"
    run_agentx "${conc}"
  done
  {
    echo "## Artifacts"
    echo
    echo "- \`validate.json\`: smoke, long-context, memory metrics"
    echo "- \`accuracy.json\`: optional reference comparison"
    echo "- \`c{N}/\`: AgentX artifacts"
    echo
    echo "Publish only MXFP4 experts + FP8 KV/indexer. Do not claim native NVFP4."
  } >> "${OUTPUT_DIR}/SUMMARY.md"
  log "done. results in ${OUTPUT_DIR}"
}

submit_qos() {
  local job="${OUTPUT_DIR}/qos.sbatch"
  mkdir -p "${OUTPUT_DIR}"
  cat > "${job}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SCRIPT_DIR}/../.."
OUTPUT_DIR="${OUTPUT_DIR}" MODEL="${MODEL}" SERVER_PORT="${SERVER_PORT}" \\
  CONCURRENCIES="${CONCURRENCIES}" BENCHMARK_DURATION="${BENCHMARK_DURATION}" \\
  LONG_CONTEXT_TOKENS="${LONG_CONTEXT_TOKENS}" IMAGE="${IMAGE}" \\
  REFERENCE_URL="${REFERENCE_URL}" \\
  bash "${SCRIPT_DIR}/minimax_m31_flash_rocm_qos.sh" local
EOF
  sbatch \
    --job-name minimax-m31-flash-tp4 \
    --account "${ACCOUNT}" \
    --qos "${QOS}" \
    --partition "${PARTITION}" \
    --time "${TIME_LIMIT}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:4 \
    --output="${OUTPUT_DIR}/qos-%j.out" \
    --error="${OUTPUT_DIR}/qos-%j.err" \
    "${job}"
}

case "${1:-local}" in
  submit)
    submit_qos
    ;;
  local)
    run_suite
    ;;
  *)
    echo "Usage: $0 [local|submit]"
    exit 1
    ;;
esac
