#!/usr/bin/env bash
# Performance: query-round TTFT with LMCache must be lower than without.
# Phase A: launch SGLang + LMCache → bench → record TTFT and verify
# the daemon log shows RETRIEVE traffic. Phase B: relaunch SGLang
# without LMCache → same bench → record TTFT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
trap cleanup_all EXIT

DAEMON_LOG="perf-daemon.log"
LMC_LOG="perf-sgl-lmc.log"
NO_LOG="perf-sgl-no.log"
PORT_LMC=30200
PORT_NO=30201

echo "--- :rocket: Phase A — start daemon + LMCache-enabled SGLang"
launch_daemon "${DAEMON_LOG}"
launch_sglang "${PORT_LMC}" "${LMC_LOG}" "lmcache"

echo "--- :stopwatch: Phase A — bench, record mean TTFT"
RETRIEVALS_BEFORE=$(count_retrievals)
TTFT_LMC=$(run_long_doc_qa_ttft "${PORT_LMC}")
RETRIEVALS_AFTER=$(count_retrievals)
RETRIEVAL_DELTA=$((RETRIEVALS_AFTER - RETRIEVALS_BEFORE))
echo "  ttft_with_lmcache = ${TTFT_LMC}s, retrieval delta = ${RETRIEVAL_DELTA}"

echo "--- :recycle: Phase B — relaunch SGLang without LMCache"
kill -9 "${SGLANG_PID}" 2>/dev/null || true
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
wait "${SGLANG_PID}" 2>/dev/null || true
SGLANG_PID=""
sleep 3
launch_sglang "${PORT_NO}" "${NO_LOG}" "no-lmcache"

echo "--- :stopwatch: Phase B — bench, record mean TTFT"
TTFT_NO=$(run_long_doc_qa_ttft "${PORT_NO}")
echo "  ttft_without_lmcache = ${TTFT_NO}s"

echo "+++ :scales: Verdict"
echo "  ttft_with_lmcache    = ${TTFT_LMC}s"
echo "  ttft_without_lmcache = ${TTFT_NO}s"
echo "  retrieval delta      = ${RETRIEVAL_DELTA}"

if [[ "${RETRIEVAL_DELTA}" -lt 1 ]]; then
    echo "FAIL: LMCache was not exercised (retrieval delta ${RETRIEVAL_DELTA} < 1)" >&2
    tail -n 60 "${DAEMON_LOG}" >&2
    exit 1
fi
if ! python3 -c "import sys; sys.exit(0 if ${TTFT_LMC} < ${TTFT_NO} else 1)"; then
    echo "FAIL: LMCache did not improve query-round TTFT (${TTFT_LMC}s >= ${TTFT_NO}s)" >&2
    exit 1
fi
echo "PASS — ttft_with_lmcache (${TTFT_LMC}s) < ttft_without_lmcache (${TTFT_NO}s); ${RETRIEVAL_DELTA} retrieval(s)."
