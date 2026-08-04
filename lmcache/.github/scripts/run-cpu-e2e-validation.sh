#!/usr/bin/env bash
set -euo pipefail

echo "Build ID: ${BUILDKITE_BUILD_ID:-local}"
echo "Python: $(python3 --version 2>&1 || true)"
if command -v uv >/dev/null 2>&1; then
  echo "uv: $(uv --version 2>&1 || true)"
else
  echo "uv: not installed"
fi

BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
VENV_DIR=".venv-${BUILD_ID}"
SHARED_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMCACHE_LOG="${LMCACHE_LOG_FILE:-/tmp/build_${BUILD_ID}_lmcache_cpu_validation.log}"
VLLM_LOG="${VLLM_LOG_FILE:-/tmp/build_${BUILD_ID}_vllm_cpu_validation.log}"
LMCACHE_PID=""
VLLM_PID=""
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
LMCACHE_ZMQ_PORT="${LMCACHE_ZMQ_PORT:-5555}"
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-1}"
LMCACHE_EVICTION_POLICY="${LMCACHE_EVICTION_POLICY:-LRU}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-128}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.1}"
# CPU-only KV cache pool size in GiB. vLLM defaults to 4 GiB when unset;
# 1 GiB is plenty for opt-125m e2e validation.
VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-1}"
# Cap context length and concurrent sequences to shrink scheduler buffers.
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
LMCACHE_HEALTHCHECK_TIMEOUT="${LMCACHE_HEALTHCHECK_TIMEOUT:-30}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-120}"
# Model selection knobs. Defaults keep the historical opt-125m path so
# existing callers see no behaviour change; matrix entries for other
# models (e.g. deepseek-ai/DeepSeek-V2-Lite-Chat) override these.
#   VLLM_MODEL_ID        HF repo id, passed to `vllm serve`
#   VLLM_LOAD_FORMAT     empty = default (real weights), "dummy" = random init
#                        (matches vLLM's --load-format flag)
#   VLLM_HF_OVERRIDES    optional JSON, passed via --hf-overrides. Used with
#                        dummy load to shrink layer/expert counts so large
#                        MoE models (MLA etc.) fit in CI runners.
#   VLLM_DOWNLOAD_SCRIPT relative filename under .github/scripts/ that
#                        prepares the model files locally. Defaults to
#                        the shared GH-Release downloader (which expects
#                        MODEL_ID/GH_REPO/GH_TAG/SNAPSHOT/TARBALL_NAME/
#                        TARBALL_TOP_DIR env vars, see the script for
#                        details). Set to "" to skip the step entirely
#                        (e.g. when weights are random-initialised and
#                        no HF metadata is needed).
#   VLLM_TARBALL_URL     Full https URL to the tar.gz consumed by the
#                        downloader. Defaults to the opt-125m GitHub
#                        release, so buildkite / historical callers keep
#                        working unchanged. GHA overrides this per-model
#                        via the matrix.
#   VLLM_TARBALL_CACHE_MARKER  Cache-hit marker filename passed through
#                              to the downloader. Defaults to
#                              `pytorch_model.bin`, which is what the
#                              opt-125m mirror ships.
VLLM_MODEL_ID="${VLLM_MODEL_ID:-facebook/opt-125m}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-}"
VLLM_HF_OVERRIDES="${VLLM_HF_OVERRIDES:-}"
VLLM_DOWNLOAD_SCRIPT="${VLLM_DOWNLOAD_SCRIPT:-download_gh_release_model.sh}"
VLLM_TARBALL_URL="${VLLM_TARBALL_URL:-https://github.com/LMCache/opt-125m/releases/download/v1.0/opt-125m.tar.gz}"
VLLM_TARBALL_CACHE_MARKER="${VLLM_TARBALL_CACHE_MARKER:-pytorch_model.bin}"
# Transport mode selection:
#   LMCACHE_MP_TRANSFER_MODE=engine_driven  -> engine-driven data path,
#       sub-selected by LMCACHE_SHM_NAME:
#       LMCACHE_SHM_NAME=""              -> pickle transport
#       LMCACHE_SHM_NAME=__default__     -> shm transport (default)
#   LMCACHE_MP_TRANSFER_MODE=lmcache_driven -> lmcache-driven IPC handle path
LMCACHE_SHM_NAME="${LMCACHE_SHM_NAME-__default__}"
LMCACHE_MP_TRANSFER_MODE="${LMCACHE_MP_TRANSFER_MODE:-engine_driven}"
# Set SKIP_INSTALL=1 to skip Phase 1 (install) — useful when the caller
# has already installed everything (e.g. macOS CI workflow steps).
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# Directory to collect artifacts before workspace is deleted
ARTIFACT_DIR="/tmp/build_${BUILD_ID}_artifacts"
mkdir -p "${ARTIFACT_DIR}"

upload_artifacts() {
  # Copy logs to artifact dir (which survives workspace deletion)
  cp -f "${LMCACHE_LOG}" "${ARTIFACT_DIR}/lmcache_cpu_validation.log" 2>/dev/null || true
  cp -f "${VLLM_LOG}" "${ARTIFACT_DIR}/vllm_cpu_validation.log" 2>/dev/null || true

  if [ -n "${BUILDKITE_BUILD_ID:-}" ] && command -v buildkite-agent >/dev/null 2>&1; then
    buildkite-agent artifact upload "${ARTIFACT_DIR}/*.log" || true
  fi
}

cleanup_workspace() {
  if [ -n "${BUILDKITE_BUILD_ID:-}" ]; then
    export TARGET="$PWD"
    case "$TARGET" in
      ""|"/"|"/usr"|"/var"|"/etc"|"/bin"|"/sbin"|"/opt"|"/home"|"/tmp")
        echo "❌ Refusing to delete unsafe workspace path: ${TARGET:-<empty>}"
        return 1
        ;;
    esac
    if [ "$TARGET" = "$HOME" ]; then
      echo "❌ Refusing to delete unsafe workspace path: ${TARGET:-<empty>}"
      return 1
    fi
    if [ ! -d "$TARGET/.git" ] || [ ! -f "$TARGET/pyproject.toml" ]; then
      echo "❌ Refusing to delete unexpected workspace path: $TARGET"
      return 1
    fi
    echo "Deleting current workspace $TARGET"
    cd /
    if command -v sudo >/dev/null 2>&1; then
      sudo rm -rf "$TARGET"
    else
      rm -rf "$TARGET"
    fi
  fi
}

print_failure_logs() {
  echo "=== LMCache Server Log (${LMCACHE_LOG}) ==="
  if [ -f "${LMCACHE_LOG}" ]; then
    tail -n 200 "${LMCACHE_LOG}" || true
  else
    echo "Log not found"
  fi
  echo "=== vLLM Log (${VLLM_LOG}) ==="
  if [ -f "${VLLM_LOG}" ]; then
    tail -n 200 "${VLLM_LOG}" || true
  else
    echo "Log not found"
  fi
}

cleanup_processes() {
  set +e
  if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM (PID=${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  if [ -n "${LMCACHE_PID}" ] && kill -0 "${LMCACHE_PID}" 2>/dev/null; then
    echo "Stopping LMCache server (PID=${LMCACHE_PID})"
    kill "${LMCACHE_PID}" 2>/dev/null || true
    wait "${LMCACHE_PID}" 2>/dev/null || true
  fi
  set -e
}

wait_for_endpoint_contains() {
  local url="$1"
  local timeout="$2"
  local expected="$3"
  local label="$4"
  local response

  for _ in $(seq 1 "${timeout}"); do
    if response="$(curl -fsS "${url}" 2>/dev/null)"; then
      if [ -z "${expected}" ] || echo "${response}" | grep -q "${expected}"; then
        return 0
      fi
    fi
    sleep 1
  done

  echo "❌ ${label} did not become ready within ${timeout}s"
  return 1
}

# Scrape a Prometheus counter value from LMCache /metrics endpoint
scrape_metric() {
  local metric_name="$1"
  python3 - <<EOF
import sys, urllib.request
url = "http://localhost:${LMCACHE_HTTP_PORT}/metrics"
try:
    body = urllib.request.urlopen(url, timeout=10).read().decode()
except Exception as e:
    print(f"ERROR fetching {url}: {e}", file=sys.stderr)
    print("0")
    sys.exit(0)
total = 0.0
for line in body.splitlines():
    if line.startswith("#"):
        continue
    if not line.startswith("${metric_name}"):
        continue
    parts = line.rsplit(" ", 1)
    if len(parts) != 2:
        continue
    try:
        total += float(parts[1])
    except ValueError:
        continue
print(int(total))
EOF
}

# Poll a Prometheus counter until its value differs from `baseline`,
# or until `timeout` seconds elapse. Returns 0 on change, 1 on timeout
# (callers typically `|| true` it because not every probe expects a
# change to actually happen).
wait_for_metric_change() {
  local metric_name="$1"
  local baseline="$2"
  local timeout="${3:-10}"
  local current
  for _ in $(seq 1 "${timeout}"); do
    current="$(scrape_metric "${metric_name}")"
    if [ "${current}" != "${baseline}" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Send a completion request and print the text output
send_completion() {
  local prompt_file="$1"
  local max_tokens="${2:-50}"
  local body_file
  body_file="$(mktemp)"
  # Build the JSON body in a separate process to avoid nested-quote
  # quoting nightmares with -d "$(python3 -c "...")". Pass the prompt
  # file path and max_tokens via argv so the python snippet itself
  # does not need any shell interpolation inside its string body.
  PROMPT_FILE="${prompt_file}" MAX_TOKENS="${max_tokens}" \
  VLLM_MODEL_ID="${VLLM_MODEL_ID}" \
    python3 - >"${body_file}" <<'PYEOF'
import json
import os

prompt = open(os.environ['PROMPT_FILE']).read()
print(json.dumps({
    'model': os.environ['VLLM_MODEL_ID'],
    'prompt': prompt,
    'max_tokens': int(os.environ['MAX_TOKENS']),
    'temperature': 0,
}))
PYEOF
  local response
  response="$(curl -fsS "http://localhost:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    --data-binary "@${body_file}")"
  rm -f "${body_file}"
  echo "${response}" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])"
}

start_vllm() {
  echo "Starting vLLM server..."
  # Export so multiproc_executor worker children inherit these. Without
  # VLLM_CPU_KVCACHE_SPACE, CPU backend falls back to
  # `total_memory * gpu_memory_utilization`, which can request 100s of GiB
  # on big hosts and OOM (see vllm/v1/worker/cpu_worker.py:determine_available_memory).
  # VLLM_DEVICE is the modern env var (vLLM 0.8+); VLLM_TARGET_DEVICE is
  # kept for backwards-compatibility with older vLLM CPU wheels.
  export VLLM_DEVICE=cpu
  export VLLM_TARGET_DEVICE=cpu
  export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE}"
  export LMCACHE_MP_TRANSFER_MODE="${LMCACHE_MP_TRANSFER_MODE}"
  # Force non-MLA attention backend when the matrix profile asks for it.
  # vLLM's CPU backend still raises NotImplementedError for MLA today,
  # so DeepSeek-V2-family models (which normally route through MLA) must
  # set this on CPU to fall back to standard MHA.
  #
  # vLLM parses this env as `bool(int(os.getenv("VLLM_MLA_DISABLE","0")))`,
  # so any non-integer value (including the empty string GitHub Actions
  # injects for unset matrix fields) blows up with ValueError. We only
  # export it when the caller explicitly set "1".
  if [ "${VLLM_MLA_DISABLE:-0}" = "1" ]; then
    export VLLM_MLA_DISABLE=1
  else
    unset VLLM_MLA_DISABLE
  fi
  # Pin gloo / vLLM rendezvous to loopback. Otherwise vLLM's
  # network_utils.get_ip() picks a LAN address (e.g. 192.168.x.x on the
  # macOS GHA runner) and gloo's init_process_group sits there for ~16
  # minutes doing slow socket bind/connect retries before the engine
  # ever loads weights.
  export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
  if [ -z "${GLOO_SOCKET_IFNAME:-}" ]; then
    case "$(uname -s)" in
      Darwin) export GLOO_SOCKET_IFNAME=lo0 ;;
      Linux)  export GLOO_SOCKET_IFNAME=lo ;;
    esac
  fi
  local kv_cache_bytes
  kv_cache_bytes="$(python3 -c "print(int(${VLLM_CPU_KVCACHE_SPACE} * 1024 * 1024 * 1024))")"
  # Tell LMCacheMPConnector where the lmcache server actually listens.
  # Without this it falls back to tcp://localhost:5555 and dies with
  # "Cannot reach the LMCache MP server" whenever we run multiple e2e
  # steps in parallel/sequence on different ZMQ ports.
  local kv_transfer_config
  kv_transfer_config="$(python3 -c "
import json
print(json.dumps({
    'kv_connector': 'LMCacheMPConnector',
    'kv_role': 'kv_both',
    'kv_connector_module_path': 'lmcache.integration.vllm.lmcache_mp_connector',
    'kv_connector_extra_config': {
        'lmcache.mp.host': 'tcp://localhost',
        'lmcache.mp.port': int('${LMCACHE_ZMQ_PORT}'),
    },
}))")"
  # Optional flags — appended only when set, so unrelated callers stay
  # bit-identical to the pre-refactor invocation.
  local -a extra_serve_args=()
  if [ -n "${VLLM_LOAD_FORMAT}" ]; then
    extra_serve_args+=(--load-format "${VLLM_LOAD_FORMAT}")
  fi
  if [ -n "${VLLM_HF_OVERRIDES}" ]; then
    extra_serve_args+=(--hf-overrides "${VLLM_HF_OVERRIDES}")
  fi
  # trust-remote-code is required for models with custom python modelling
  # code (e.g. DeepSeek-V2). Harmless for models that don't need it.
  if [ "${VLLM_TRUST_REMOTE_CODE:-0}" = "1" ]; then
    extra_serve_args+=(--trust-remote-code)
  fi
  echo "vLLM model: ${VLLM_MODEL_ID}"
  echo "vLLM extra args: ${extra_serve_args[*]:-<none>}"
  vllm serve "${VLLM_MODEL_ID}" \
    --port "${VLLM_PORT}" \
    --dtype bfloat16 \
    --disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --kv-cache-memory-bytes "${kv_cache_bytes}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --kv-transfer-config "${kv_transfer_config}" \
    --enforce-eager \
    ${extra_serve_args[@]+"${extra_serve_args[@]}"} \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  echo "vLLM server started (PID=${VLLM_PID})"
  sleep 1
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "❌ vLLM server exited immediately after startup. See ${VLLM_LOG} for details"
    return 1
  fi
  echo "Waiting for vLLM readiness at http://localhost:${VLLM_PORT}/v1/models (timeout: ${VLLM_READY_TIMEOUT}s)"
  if ! wait_for_endpoint_contains "http://localhost:${VLLM_PORT}/v1/models" "${VLLM_READY_TIMEOUT}" "${VLLM_MODEL_ID}" "vLLM server"; then
    return 1
  fi
  echo "✅ vLLM server is ready"
}

stop_vllm() {
  if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM (PID=${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
  fi
}

on_error() {
  local exit_code=$?
  trap - ERR
  echo "❌ CPU install validation failed (exit code: ${exit_code})"
  set +e
  print_failure_logs
  cleanup_processes
  upload_artifacts
  cleanup_workspace || echo "❌ Workspace cleanup failed"
  set -e
  exit "$exit_code"
}

trap on_error ERR

if [ "${SKIP_INSTALL}" = "1" ]; then
  echo "=== CPU Install Validation (Phase 1) — SKIPPED (SKIP_INSTALL=1) ==="
else
  echo "=== CPU Install Validation (Phase 1) ==="
  echo "Creating virtual environment with uv at ${VENV_DIR}"
  uv venv --python 3.12 "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  uv pip install --upgrade pip setuptools wheel

  # `--index-strategy unsafe-best-match` is required because uv's
  # default `first-index` strategy locks each package to the first
  # index that lists it; the pytorch CPU index ships an older mirror
  # of `setuptools` that would block the version vllm-cpu-nightly
  # pins.
  uv pip uninstall -y vllm vllm-cpu-nightly 2>/dev/null || true
  PIP_BIN="uv pip" \
  PIP_INSTALL_EXTRA_ARGS="--index-strategy unsafe-best-match" \
    bash "${SHARED_SCRIPTS_DIR}/install_vllm_cpu.sh"

  PIP_BIN="uv pip" \
    bash "${SHARED_SCRIPTS_DIR}/install_lmcache_cpu.sh"

  echo "Freezing installed package versions"
  uv pip freeze

  echo "✅ CPU install validation passed"
fi

echo "=== CPU E2E Validation (Phase 2) ==="

if [ "${SKIP_INSTALL}" = "1" ]; then
  echo "[Phase 2 / Step 1] numpy<2 install — SKIPPED (SKIP_INSTALL=1)"
else
  echo "[Phase 2 / Step 1] Installing numpy<2 for scipy/vLLM compatibility"
  uv pip install "numpy<2"
  echo "✅ numpy<2 installed"
fi

if [ -z "${VLLM_DOWNLOAD_SCRIPT}" ]; then
  echo "[Phase 2 / Step 2] Skipping model download (VLLM_DOWNLOAD_SCRIPT is empty)."
else
  echo "[Phase 2 / Step 2] Preparing model ${VLLM_MODEL_ID} via ${VLLM_DOWNLOAD_SCRIPT}"
  # The downloader reads MODEL_ID / TARBALL_URL / CACHE_MARKER from
  # env; thread them through explicitly so the buildkite path (which
  # relies on the defaults above) and the GHA path (which pins them
  # per matrix entry) both work without depending on the surrounding
  # shell's env-var visibility rules.
  MODEL_ID="${VLLM_MODEL_ID}" \
    TARBALL_URL="${VLLM_TARBALL_URL}" \
    CACHE_MARKER="${VLLM_TARBALL_CACHE_MARKER}" \
    bash "${SHARED_SCRIPTS_DIR}/${VLLM_DOWNLOAD_SCRIPT}"
  echo "✅ Model download/check complete"
fi

echo "[Phase 2 / Step 3] Starting LMCache server"
echo "LMCache log: ${LMCACHE_LOG}"
# Build lmcache server args
LMCACHE_ARGS=(
  --port "${LMCACHE_ZMQ_PORT}"
  --http-port "${LMCACHE_HTTP_PORT}"
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}"
  --eviction-policy "${LMCACHE_EVICTION_POLICY}"
  --chunk-size "${LMCACHE_CHUNK_SIZE}"
)
# `engine_driven` / `lmcache_driven` resolve to different worker-side
# TransferContexts:
#   engine_driven -> EngineDrivenTransferContext (worker gathers/scatters data)
#     sub-mode: SHM (--shm-name __default__) or pickle (--shm-name "")
#   lmcache_driven -> LMCacheDrivenTransferContext (server-side IPC handle)
# Step 5.5 verifies which one the worker actually entered.
if [ "${LMCACHE_MP_TRANSFER_MODE}" = "engine_driven" ]; then
  if [ "${LMCACHE_SHM_NAME}" = "__default__" ]; then
    echo "Transport mode: engine-driven/shm (shared memory)"
    EXPECTED_TRANSPORT="shm"
  else
    echo "Transport mode: engine-driven/pickle (--shm-name '${LMCACHE_SHM_NAME}')"
    LMCACHE_ARGS+=(--shm-name "${LMCACHE_SHM_NAME}")
    EXPECTED_TRANSPORT="pickle"
  fi
elif [ "${LMCACHE_MP_TRANSFER_MODE}" = "lmcache_driven" ]; then
  echo "Transport mode: lmcache-driven (IPC handle path)"
  EXPECTED_TRANSPORT="lmcache_driven"
else
  echo "Transport mode: unknown '${LMCACHE_MP_TRANSFER_MODE}',"
  echo "  falling back to LMCACHE_SHM_NAME-based detection"
  if [ "${LMCACHE_SHM_NAME}" = "__default__" ]; then
    echo "Transport mode: data/shm (shared memory, fallback)"
    EXPECTED_TRANSPORT="shm"
  else
    echo "Transport mode: data/pickle (--shm-name '${LMCACHE_SHM_NAME}')"
    LMCACHE_ARGS+=(--shm-name "${LMCACHE_SHM_NAME}")
    EXPECTED_TRANSPORT="pickle"
  fi
fi

lmcache server "${LMCACHE_ARGS[@]}" \
  >"${LMCACHE_LOG}" 2>&1 &
LMCACHE_PID=$!
echo "LMCache server started (PID=${LMCACHE_PID})"
sleep 1
if ! kill -0 "${LMCACHE_PID}" 2>/dev/null; then
  echo "❌ LMCache server exited immediately after startup. See ${LMCACHE_LOG} for details"
  false
fi

echo "Waiting for LMCache healthcheck at http://localhost:${LMCACHE_HTTP_PORT}/healthcheck (timeout: ${LMCACHE_HEALTHCHECK_TIMEOUT}s)"
if ! wait_for_endpoint_contains "http://localhost:${LMCACHE_HTTP_PORT}/healthcheck" "${LMCACHE_HEALTHCHECK_TIMEOUT}" "" "LMCache server"; then
  false
fi
echo "✅ LMCache server is healthy"

echo "[Phase 2 / Step 4] Installing system libs (Linux only) and starting vLLM server"
if [ "$(uname -s)" = "Linux" ]; then
  # System libs vLLM's CPU serve path dlopens at import time:
  #   - libnuma1: needed by some vLLM CPU code paths.
  #   - ffmpeg:   provides libavutil.so.{56..60} etc. that torchcodec
  #               dlopens. Recent vLLM nightlies import torchcodec
  #               unconditionally, so `vllm serve` aborts with
  #               "libavutil.so.NN: cannot open shared object file"
  #               without FFmpeg — even for text-only models.
  # Prefer sudo if present (GitHub ubuntu runners need it). Install only
  # what's missing and never let an apt hiccup fail the step.
  MISSING_PKGS=()
  if [ ! -e /usr/lib/x86_64-linux-gnu/libnuma.so.1 ] \
     && [ ! -e /lib/x86_64-linux-gnu/libnuma.so.1 ]; then
    MISSING_PKGS+=(libnuma1)
  fi
  # torchcodec supports FFmpeg 4-8; the distro's ffmpeg package provides a
  # compatible libavutil. Detect via ldconfig so we match whichever
  # soname (56..60) the runner already ships.
  if ! ldconfig -p 2>/dev/null | grep -q 'libavutil\.so'; then
    MISSING_PKGS+=(ffmpeg)
  fi
  if [ "${#MISSING_PKGS[@]}" -gt 0 ]; then
    echo "Installing missing system packages: ${MISSING_PKGS[*]}"
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update \
        && sudo apt-get install -y --no-install-recommends "${MISSING_PKGS[@]}" \
        || echo "⚠️  apt-get install (${MISSING_PKGS[*]}) via sudo failed; continuing"
    else
      apt-get update \
        && apt-get install -y --no-install-recommends "${MISSING_PKGS[@]}" \
        || echo "⚠️  apt-get install (${MISSING_PKGS[*]}) failed; continuing"
    fi
  else
    echo "libnuma1 and FFmpeg runtime libs already present, skipping apt install"
  fi
fi
# VLLM_DEVICE is the modern env var (vLLM 0.8+)
export VLLM_DEVICE=cpu
export VLLM_TARGET_DEVICE=cpu
start_vllm

echo "[Phase 2 / Step 5] Sending E2E test request"
e2e_request_body="$(VLLM_MODEL_ID="${VLLM_MODEL_ID}" python3 -c "
import json, os
print(json.dumps({
    'model': os.environ['VLLM_MODEL_ID'],
    'prompt': 'Hello',
    'max_tokens': 5,
}))")"
completion_response="$(curl -fsS "http://localhost:${VLLM_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "${e2e_request_body}")"
echo "Completion response: ${completion_response}"
if ! echo "${completion_response}" | grep -q '"choices"'; then
  echo "❌ E2E request response failed structural validation"
  false
fi
if ! echo "${completion_response}" | grep -q "${VLLM_MODEL_ID}"; then
  echo "❌ E2E request response missing expected model"
  false
fi
echo "✅ E2E request validation passed"

# Verify transport mode (logged after vLLM connects to LMCache server)
echo "[Phase 2 / Step 5.5] Verifying transport mode: expecting '${EXPECTED_TRANSPORT}'"
# Worker logs `Creating transfer context (device_type=<dev>, mode=<m>)`
# from worker_transfer.py:create_transfer_context, where <m> is the
# resolved MPTransferMode after env-var lookup. This is the single source
# of truth for which TransferContext the worker actually entered. Note:
# the worker is a child of `vllm serve`, so its LMCache log lines land in
# VLLM_LOG (vllm's stdout), not in LMCACHE_LOG (lmcache server's stdout).
# The shm/pickle branches still grep LMCACHE_LOG because the strategy
# line is emitted by the lmcache server itself.
if [ "${EXPECTED_TRANSPORT}" = "lmcache_driven" ]; then
  if ! grep -q "Creating transfer context.*mode=lmcache_driven" "${VLLM_LOG}" 2>/dev/null; then
    echo "❌ Expected lmcache-driven worker context but 'mode=lmcache_driven' not found in vLLM log"
    tail -50 "${VLLM_LOG}"
    false
  fi
  echo "✅ Transport mode confirmed: lmcache-driven (IPC handle path)"
elif [ "${EXPECTED_TRANSPORT}" = "engine_driven" ]; then
  if ! grep -q "Creating transfer context.*mode=engine_driven" "${VLLM_LOG}" 2>/dev/null; then
    echo "❌ Expected engine-driven worker context but 'mode=engine_driven' not found in vLLM log"
    tail -50 "${VLLM_LOG}"
    false
  fi
  echo "✅ Transport mode confirmed: engine-driven"
elif [ "${EXPECTED_TRANSPORT}" = "shm" ]; then
  if ! grep -q "Using shm non-GPU transfer strategy" "${LMCACHE_LOG}" 2>/dev/null; then
    echo "❌ Expected shm transport but server strategy line not found in log"
    tail -50 "${LMCACHE_LOG}"
    false
  fi
  echo "✅ Transport mode confirmed: shm"
elif [ "${EXPECTED_TRANSPORT}" = "pickle" ]; then
  if ! grep -q "Using pickle non-GPU transfer strategy" "${LMCACHE_LOG}" 2>/dev/null; then
    echo "❌ Expected pickle transport but server strategy line not found in log"
    tail -50 "${LMCACHE_LOG}"
    false
  fi
  echo "✅ Transport mode confirmed: pickle"
fi

echo "[Phase 2 / Step 6] Cleaning up Phase 2 vLLM"
stop_vllm
echo "✅ Phase 2 cleanup completed"

echo "✅ CPU E2E validation passed"

# ═══════════════════════════════════════════════════════════════════
# Phase 3: Cache Hit Validation
# ═══════════════════════════════════════════════════════════════════
# Scenario:
#   - LMCache server stays running the entire time
#   - vLLM instance 1: request A → LMCache store; request A again → LMCache hit
#   - vLLM restart (instance 2): request A → LMCache hit (cross-instance)
#   - All three outputs must be identical (bit-exact with temperature=0)
# ═══════════════════════════════════════════════════════════════════

echo "=== Cache Hit Validation (Phase 3) ==="

# Generate a fixed ~1000 token prompt
PROMPT_FILE="/tmp/build_${BUILD_ID}_phase3_prompt.txt"
python3 -c "
# A diverse, non-repetitive story prompt (~1000 tokens for opt-125m).
# The model will continue the narrative; correctness doesn't matter,
# only that all three runs produce identical output (cache consistency).
story = '''Once upon a time in a small coastal village, there lived an old lighthouse keeper named Thomas. Every evening, he climbed the one hundred and thirty-seven steps to the top of the lighthouse to light the great lamp. The sea was unpredictable in those parts. Ships from distant lands carried spices, silk, and stories of places Thomas had never seen. One stormy night in November, a merchant vessel called the Silver Heron appeared on the horizon, listing dangerously to starboard. Thomas watched through his brass telescope as the waves crashed against its hull. He knew that if the ship did not change course within the next ten minutes, it would strike the jagged rocks known locally as the Devil's Teeth. He grabbed the emergency flare gun from the wooden cabinet and fired three red flares into the sky. The captain of the Silver Heron saw the warning and ordered hard to port. The ship groaned as it turned, barely clearing the outermost rock by twenty meters. The next morning, the captain rowed ashore to thank Thomas personally. He brought a gift: a small wooden box containing a compass that always pointed not north, but toward home. Thomas kept it on his desk for the rest of his days. Years later, when Thomas retired, he passed the compass to his granddaughter Elena, who had inherited his love of the sea. Elena became a marine biologist studying the migration patterns of humpback whales along the Pacific coast. She traveled from Alaska to Mexico following the whale pods, documenting their songs and social behaviors. Her research revealed that whale families maintained bonds across thousands of miles, communicating through low-frequency calls that could travel entire ocean basins. One afternoon while diving near a coral reef off the coast of Baja California, Elena discovered something extraordinary beneath a rocky overhang:'''
print(story, end='')
" > "${PROMPT_FILE}"
echo "Generated prompt file: ${PROMPT_FILE} ($(wc -c < "${PROMPT_FILE}") bytes)"

# Reset metrics to have a clean baseline
echo "[Phase 3 / Step 1] Resetting LMCache metrics"
curl -fsS -X POST "http://localhost:${LMCACHE_HTTP_PORT}/metrics/reset" >/dev/null
echo "✅ Metrics reset"

# Start vLLM instance 1
echo "[Phase 3 / Step 2] Starting vLLM (instance 1)"
start_vllm

# Request A (first time) → should trigger store
echo "[Phase 3 / Step 3] Request A (first) — expecting LMCache store"
L1_WRITE_BEFORE=$(scrape_metric "lmcache_mp_l1_write_chunks_total")
OUTPUT_1=$(send_completion "${PROMPT_FILE}" 50)
echo "Output 1: ${OUTPUT_1}"
sleep 2  # allow async store to complete
L1_WRITE_AFTER=$(scrape_metric "lmcache_mp_l1_write_chunks_total")
STORE_DELTA=$((L1_WRITE_AFTER - L1_WRITE_BEFORE))
echo "L1 write chunks delta: ${STORE_DELTA}"
if [ "${STORE_DELTA}" -lt 1 ]; then
  echo "❌ No L1 write activity after first request (expected store)"
  false
fi
echo "✅ LMCache store verified (${STORE_DELTA} chunks written)"

# Request A (second time, same vLLM instance) → should trigger read/hit
echo "[Phase 3 / Step 4] Request A (second) — expecting LMCache hit"
L1_READ_BEFORE=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
OUTPUT_2=$(send_completion "${PROMPT_FILE}" 50)
echo "Output 2: ${OUTPUT_2}"
sleep 2
L1_READ_AFTER=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
READ_DELTA=$((L1_READ_AFTER - L1_READ_BEFORE))
echo "L1 read chunks delta: ${READ_DELTA}"
if [ "${READ_DELTA}" -lt 1 ]; then
  echo "❌ No L1 read activity on second request (expected cache hit)"
  false
fi
echo "✅ LMCache hit verified on same instance (${READ_DELTA} chunks read)"

# Restart vLLM
echo "[Phase 3 / Step 5] Restarting vLLM (instance 2)"
stop_vllm
sleep 2
start_vllm

# Request A (third time, new vLLM instance) → should trigger read/hit from LMCache
echo "[Phase 3 / Step 6] Request A (third) — expecting LMCache hit after vLLM restart"
L1_READ_BEFORE=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
OUTPUT_3=$(send_completion "${PROMPT_FILE}" 50)
echo "Output 3: ${OUTPUT_3}"
sleep 2
L1_READ_AFTER=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
READ_DELTA=$((L1_READ_AFTER - L1_READ_BEFORE))
echo "L1 read chunks delta: ${READ_DELTA}"
if [ "${READ_DELTA}" -lt 1 ]; then
  echo "❌ No L1 read activity after vLLM restart (expected cross-instance cache hit)"
  false
fi
echo "✅ LMCache cross-instance hit verified (${READ_DELTA} chunks read)"

# Verify all three outputs are identical
echo "[Phase 3 / Step 7] Verifying output consistency"
if [ "${OUTPUT_1}" != "${OUTPUT_2}" ]; then
  echo "❌ Output mismatch between request 1 and request 2"
  echo "  Output 1: ${OUTPUT_1}"
  echo "  Output 2: ${OUTPUT_2}"
  false
fi
if [ "${OUTPUT_1}" != "${OUTPUT_3}" ]; then
  echo "❌ Output mismatch between request 1 and request 3 (after vLLM restart)"
  echo "  Output 1: ${OUTPUT_1}"
  echo "  Output 3: ${OUTPUT_3}"
  false
fi
echo "✅ All three outputs are identical — cache does not alter inference results"

# Negative test: a completely different prompt should NOT hit the cache
echo "[Phase 3 / Step 8] Request B (different prompt) — expecting cache MISS"
PROMPT_FILE_B="/tmp/build_${BUILD_ID}_phase3_prompt_b.txt"
python3 -c "
# A completely different prompt that shares no prefix with prompt A
story_b = '''In the year 2147, humanity established its first permanent colony on Mars. The settlement, named Arcadia, housed three thousand researchers and engineers working to terraform the red planet. Chief botanist Dr. Yuki Tanaka spent her days in the greenhouse domes, cultivating genetically modified crops that could thrive in Martian soil. The atmospheric processors hummed day and night, slowly converting carbon dioxide into breathable oxygen. It was tedious work measured in decades, but the colonists were patient. Every morning, Dr. Tanaka checked her instruments and recorded the oxygen levels in her logbook. Today the readings showed'''
print(story_b, end='')
" > "${PROMPT_FILE_B}"
L1_READ_BEFORE=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
OUTPUT_B=$(send_completion "${PROMPT_FILE_B}" 50)
echo "Output B: ${OUTPUT_B}"
wait_for_metric_change "lmcache_mp_l1_read_chunks_total" "${L1_READ_BEFORE}" 5 || true
L1_READ_AFTER=$(scrape_metric "lmcache_mp_l1_read_chunks_total")
READ_DELTA=$((L1_READ_AFTER - L1_READ_BEFORE))
echo "L1 read chunks delta for new prompt: ${READ_DELTA}"
if [ "${READ_DELTA}" -gt 0 ]; then
  echo "❌ Unexpected cache hit on a completely different prompt — metrics may be unreliable"
  false
fi
echo "✅ Cache miss confirmed for different prompt — metrics are trustworthy"

echo "[Phase 3 / Step 9] Cleaning up"
stop_vllm
cleanup_processes
echo "✅ Phase 3 cleanup completed"

echo ""
echo "=========================================="
echo "✅ All phases passed (Phase 1 + 2 + 3)"
echo "=========================================="

# Make sure lmcache/vllm processes started in this run are reaped so
# the next CI step does not collide on their default ZMQ/HTTP ports.
cleanup_processes

# Upload artifacts BEFORE deleting the workspace
upload_artifacts
cleanup_workspace
