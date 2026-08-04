#!/usr/bin/env bash
# Correctness: LMCache must not change the output, and must be exercised. A → B → flush_cache → A on the LMCache server: A populates LMCache, B is a separate request, /flush_cache clears SGLang's radix (but not LMCache), so the second A is a radix miss / LMCache hit. The cache-hit output is diffed against a no-LMCache reference run. Servers are launched sequentially (kill between phases) so two SGLang processes don't fight for GPU memory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DAEMON_LOG="correctness-daemon.log"
LMC_SGL_LOG="correctness-sgl-lmc.log"
NO_SGL_LOG="correctness-sgl-no.log"
PORT_LMC=30200
PORT_NO=30201

trap cleanup_all EXIT

echo "--- :rocket: Start daemon"
launch_daemon "${DAEMON_LOG}"

PROMPT_A="$(generate_prompt a)"
PROMPT_B="$(generate_prompt b)"

echo "--- :test_tube: Launch SGLang+LMCache; drive A → B → flush_cache → A"
launch_sglang "${PORT_LMC}" "${LMC_SGL_LOG}" "lmcache"
chat_completion "${PORT_LMC}" "${PROMPT_A}" 64 > /tmp/lmc_run1_A.txt
chat_completion "${PORT_LMC}" "${PROMPT_B}" 64 > /tmp/lmc_run2_B.txt
curl -sf -X POST "http://127.0.0.1:${PORT_LMC}/flush_cache" >/dev/null
RETRIEVALS_BEFORE=$(count_retrievals)
chat_completion "${PORT_LMC}" "${PROMPT_A}" 64 > /tmp/lmc_out.txt
RETRIEVALS_AFTER=$(count_retrievals)
RETRIEVAL_DELTA=$((RETRIEVALS_AFTER - RETRIEVALS_BEFORE))
echo "  daemon retrieval delta on the cache-hit call: ${RETRIEVAL_DELTA}"

echo "--- :recycle: Kill LMCache SGLang, relaunch without LMCache"
kill -9 "${SGLANG_PID}" 2>/dev/null || true
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
wait "${SGLANG_PID}" 2>/dev/null || true
SGLANG_PID=""
sleep 3
launch_sglang "${PORT_NO}" "${NO_SGL_LOG}" "no-lmcache"

echo "--- :mag: Reference call on the no-LMCache server"
chat_completion "${PORT_NO}" "${PROMPT_A}" 64 > /tmp/no_out.txt

echo "+++ :scales: Verdict"
if [[ "${RETRIEVAL_DELTA}" -lt 1 ]]; then
    echo "FAIL: LMCache was not exercised (retrieval delta ${RETRIEVAL_DELTA} < 1)" >&2
    tail -n 60 "${DAEMON_LOG}" >&2
    exit 1
fi
if ! diff -u /tmp/no_out.txt /tmp/lmc_out.txt; then
    echo "FAIL: LMCache changed the output" >&2
    exit 1
fi
echo "PASS — outputs match AND LMCache served ${RETRIEVAL_DELTA} retrieval(s) on the cache-hit call."
