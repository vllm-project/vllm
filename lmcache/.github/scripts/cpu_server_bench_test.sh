#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# CPU server bench test: starts lmcache server, runs
# ``lmcache bench server --mode cpu`` with the requested transfer mode,
# then tears down.
#
# Transfer modes (LMCACHE_BENCH_TRANSFER_MODE):
#   lmcache_driven - worker-side gather/scatter via POSIX SHM pool
#   engine_driven  - server-side copy via POSIX SHM IPC (shm_open/mmap)
#
# Environment variables (all optional, defaults shown):
#   LMCACHE_BENCH_TRANSFER_MODE  lmcache_driven|engine_driven
#                                (default: engine_driven)
#   LMCACHE_HTTP_PORT            HTTP port            (default: 18080)
#   LMCACHE_ZMQ_PORT             ZMQ RPC port         (default: 15555)
#   LMCACHE_LOG_FILE             server log path      (default: /tmp/...)
#   LMCACHE_HEALTHCHECK_TIMEOUT  seconds              (default: 60)
#   BENCH_NUM_REQUESTS           requests to run      (default: 3)
#   BENCH_NUM_TOKENS             tokens per request   (default: 512)

set -euo pipefail

OS="$(uname -s)"
echo "==> CPU server bench test (OS: ${OS})"
echo "    Python: $(python3 --version 2>&1 || true)"

TRANSFER_MODE="${LMCACHE_BENCH_TRANSFER_MODE:-engine_driven}"
HTTP_PORT="${LMCACHE_HTTP_PORT:-18080}"
ZMQ_PORT="${LMCACHE_ZMQ_PORT:-15555}"
LOG_FILE="${LMCACHE_LOG_FILE:-/tmp/cpu_server_bench_lmcache.log}"
HEALTHCHECK_TIMEOUT="${LMCACHE_HEALTHCHECK_TIMEOUT:-60}"
BENCH_NUM_REQUESTS="${BENCH_NUM_REQUESTS:-3}"
BENCH_NUM_TOKENS="${BENCH_NUM_TOKENS:-512}"

case "${TRANSFER_MODE}" in
  lmcache_driven|engine_driven) ;;
  *)
    echo "!! Unknown LMCACHE_BENCH_TRANSFER_MODE='${TRANSFER_MODE}'"
    echo "   Valid values: lmcache_driven, engine_driven"
    exit 1
    ;;
esac

echo "    TRANSFER_MODE=${TRANSFER_MODE}"
echo "    HTTP_PORT=${HTTP_PORT}  ZMQ_PORT=${ZMQ_PORT}"
echo "    BENCH_NUM_REQUESTS=${BENCH_NUM_REQUESTS}"
echo "    BENCH_NUM_TOKENS=${BENCH_NUM_TOKENS}"

# ------------------------------------------------------------------ #
# Start lmcache server
# ------------------------------------------------------------------ #
echo ""
echo "==> Starting lmcache server (log: ${LOG_FILE})"
rm -f "${LOG_FILE}"

lmcache server \
  --port "${ZMQ_PORT}" \
  --http-port "${HTTP_PORT}" \
  --l1-size-gb 1 \
  --eviction-policy LRU \
  >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "==> Cleanup: stopping lmcache server (pid=${SERVER_PID})"
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
  echo "==> Last 50 lines of server log:"
  tail -n 50 "${LOG_FILE}" 2>/dev/null || true
}
trap cleanup EXIT

# ------------------------------------------------------------------ #
# Wait for healthcheck
# ------------------------------------------------------------------ #
echo "==> Waiting for healthcheck (timeout: ${HEALTHCHECK_TIMEOUT}s)"
READY=0
for i in $(seq 1 "${HEALTHCHECK_TIMEOUT}"); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "!! lmcache server exited prematurely after ${i}s"
    break
  fi
  if curl -fsS --max-time 2 \
      "http://127.0.0.1:${HTTP_PORT}/healthcheck" >/dev/null 2>&1; then
    READY=1
    echo "    Server healthy after ${i}s"
    break
  fi
  sleep 1
done

if [ "${READY}" != "1" ]; then
  echo "!! lmcache server did not become healthy within ${HEALTHCHECK_TIMEOUT}s"
  exit 1
fi

# ------------------------------------------------------------------ #
# Run bench and validate results
# ------------------------------------------------------------------ #
echo ""
echo "==> Running: lmcache bench server" \
  "--mode cpu --transfer-mode ${TRANSFER_MODE}" \
  "--num-tokens ${BENCH_NUM_TOKENS}" \
  "--end ${BENCH_NUM_REQUESTS}"

BENCH_LOG="${BENCH_OUTPUT_LOG:-/tmp/cpu_server_bench_output.log}"
lmcache bench server \
  --rpc-url "tcp://127.0.0.1:${ZMQ_PORT}" \
  --url "http://127.0.0.1:${HTTP_PORT}" \
  --mode cpu \
  --transfer-mode "${TRANSFER_MODE}" \
  --num-tokens "${BENCH_NUM_TOKENS}" \
  --end "${BENCH_NUM_REQUESTS}" \
  2>&1 | tee "${BENCH_LOG}"

echo ""
echo "==> Validating bench results"

if grep -q "CHECKSUM MISMATCH" "${BENCH_LOG}"; then
  echo "!! CHECKSUM MISMATCH detected — store/retrieve data corruption"
  exit 1
fi

MATCH_COUNT="$(grep -c "CHECKSUM MATCH OK" "${BENCH_LOG}" || true)"
if [ "${MATCH_COUNT}" -lt "${BENCH_NUM_REQUESTS}" ]; then
  echo "!! CHECKSUM MATCH count (${MATCH_COUNT}) < expected (${BENCH_NUM_REQUESTS})"
  exit 1
fi
echo "    CHECKSUM MATCH: ${MATCH_COUNT}/${BENCH_NUM_REQUESTS} request(s) verified OK"

echo ""
echo "==> CPU server bench (${TRANSFER_MODE}) passed."
