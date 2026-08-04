#!/usr/bin/env bash
# GDS L1 smoke test. Sends a few completions (cold) to store KV to the slab,
# resets vLLM's prefix cache, then re-sends them (warm) to read the KV back from
# LMCache/GDS. Passes if every request returns HTTP 200, a real LMCache retrieve
# happened, and the warm (GDS-retrieved) outputs match the cold (recomputed)
# ones -- i.e. the GDS store/retrieve path works and is correct.
#
# Expects the GDS-enabled LMCache server + vLLM to already be running, with
# VLLM_SERVER_DEV_MODE=1 (for /reset_prefix_cache).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"
N_PROMPTS="${GDS_SMOKE_PROMPTS:-4}"
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

# A long-ish prompt so each request stores at least one LMCache chunk.
build_prompt() {  # $1 = unique id
    local filler="The key-value cache stores attention keys and values across transformer layers. "
    local body="" i
    for i in $(seq 1 80); do body="${body}${filler}"; done
    printf 'Document %s. %s' "$1" "$body"
}

# Send N_PROMPTS completions; capture each generated text to
# $OUT_DIR/<label>_<i>.txt and require every request to return HTTP 200.
send_batch() {  # $1 = phase label (cold|warm)
    local label="$1" ok=0 i prompt payload resp http body
    for i in $(seq 1 "$N_PROMPTS"); do
        prompt="$(build_prompt "$i")"
        payload=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":sys.argv[2],"max_tokens":16,"temperature":0}))' "$MODEL" "$prompt")
        resp=$(curl -s -w $'\n%{http_code}' \
            "http://127.0.0.1:${VLLM_PORT}/v1/completions" \
            -H "Content-Type: application/json" -d "$payload")
        http="${resp##*$'\n'}"
        body="${resp%$'\n'*}"
        printf '%s' "$body" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])' \
            > "${OUT_DIR}/${label}_${i}.txt" 2>/dev/null \
            || echo "<no-output>" > "${OUT_DIR}/${label}_${i}.txt"
        echo "  [$label] req $i -> HTTP $http"
        [ "$http" = "200" ] && ok=$((ok + 1))
    done
    [ "$ok" -eq "$N_PROMPTS" ] || { echo "[$label] only $ok/$N_PROMPTS returned HTTP 200"; return 1; }
}

# Count completed LMCache retrieves recorded in the server log (0 if no log yet).
count_retrieves() {
    [ -f "$LMCACHE_LOG" ] || { echo 0; return; }
    grep -c "Retrieved" "$LMCACHE_LOG" 2>/dev/null || true
}

echo "============================================"
echo "=== GDS smoke: phase 1 (cold -> store KV to the GDS slab) ==="
echo "============================================"
send_batch cold
echo "Waiting for async stores to drain to the LMCache server..."
sleep 3
retrieves_before=$(count_retrieves)

echo "============================================"
echo "=== Reset vLLM prefix cache (force warm requests through LMCache/GDS) ==="
echo "============================================"
reset_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    "http://127.0.0.1:${VLLM_PORT}/reset_prefix_cache")
if [ "$reset_code" != "200" ]; then
    echo "reset_prefix_cache failed (HTTP $reset_code); is VLLM_SERVER_DEV_MODE=1?"
    exit 1
fi
sleep 2

echo "============================================"
echo "=== GDS smoke: phase 2 (warm -> retrieve KV from the GDS slab) ==="
echo "============================================"
send_batch warm
retrieves_after=$(count_retrieves)

# 1. A real GDS retrieve must have happened (else warm recomputed / hit the APC).
echo ""
echo "LMCache retrieves logged: before=${retrieves_before} after=${retrieves_after}"
if [ "$retrieves_after" -le "$retrieves_before" ]; then
    echo "GDS smoke FAILED: no LMCache retrieve recorded -- the GDS read path was"
    echo "not exercised (warm requests recomputed or hit vLLM's prefix cache)."
    exit 1
fi

# 2. The KV retrieved from the GDS slab must produce the same output as the
#    cold recompute (deterministic decoding -> byte-identical completions).
echo "=== Verifying warm (GDS-retrieved) outputs match cold (recomputed) ==="
mismatch=0
for i in $(seq 1 "$N_PROMPTS"); do
    if diff -q "${OUT_DIR}/cold_${i}.txt" "${OUT_DIR}/warm_${i}.txt" >/dev/null 2>&1; then
        echo "  prompt $i: match"
    else
        echo "  prompt $i: MISMATCH"
        mismatch=$((mismatch + 1))
    fi
done
if [ "$mismatch" -ne 0 ]; then
    echo "GDS smoke FAILED: ${mismatch}/${N_PROMPTS} warm outputs differ from cold"
    echo "-- the KV retrieved from the GDS slab is incorrect."
    exit 1
fi

echo "=== GDS smoke test passed: GDS store + retrieve path works and is correct ==="
