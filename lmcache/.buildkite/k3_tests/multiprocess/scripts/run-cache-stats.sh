#!/usr/bin/env bash
# Test that kv_transfer_params / cached_token_stats flows end-to-end
# through the OpenAI-compatible API when LMCache MP mode is active.
#
# Flow:
#   1. Send a long prompt (cold — populates LMCache, no cache hit)
#   2. Send the same prompt again (warm — should hit LMCache)
#   3. Verify the response contains cached_token_stats with expected values
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration (inherited from run-single-test.sh)
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

STATS_DIR="$RESULTS_DIR/cache_stats"
mkdir -p "$STATS_DIR"

echo "=== Cache Stats Reporting Test ==="
echo "Model: $MODEL"
echo "vLLM Port: $VLLM_PORT"
echo "Results dir: $STATS_DIR"
echo ""

# Build a prompt long enough to span multiple LMCache chunks (default
# chunk_size=256 tokens). Repeating a sentence gives us ~600+ tokens.
LONG_CONTENT="Explain the history of computer science in great detail. $(printf 'The Turing machine is a fundamental concept in theoretical computer science that defines an abstract machine capable of manipulating symbols on a strip of tape according to a table of rules. %.0s' {1..20})"

send_request() {
    local label="$1"
    local output_file="$2"

    echo "--- Sending request: $label ---"
    local http_code
    http_code=$(curl -s -o "$output_file" -w "%{http_code}" \
        -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json; print(json.dumps('$LONG_CONTENT'))")}],
            \"max_tokens\": 1,
            \"kv_transfer_params\": {\"cached_token_stats\": true}
        }")

    if [ "$http_code" -ne 200 ]; then
        echo "FAIL: $label returned HTTP $http_code"
        cat "$output_file"
        return 1
    fi
    echo "$label: HTTP 200 OK"
}

validate_stats_present() {
    local label="$1"
    local response_file="$2"

    python3 -c "
import json, sys

with open('$response_file') as f:
    data = json.load(f)

kv_params = data.get('kv_transfer_params')
if kv_params is None:
    print('FAIL: $label — kv_transfer_params is missing from response')
    sys.exit(1)

stats = kv_params.get('cached_token_stats')
if stats is None:
    print('FAIL: $label — cached_token_stats is missing from kv_transfer_params')
    print(f'  kv_transfer_params = {kv_params}')
    sys.exit(1)

required_keys = [
    'num_vllm_cached_tokens',
    'num_lmcache_cached_tokens',
    'num_lmcache_extra_cached_tokens',
]
missing = [k for k in required_keys if k not in stats]
if missing:
    print(f'FAIL: $label — missing keys in cached_token_stats: {missing}')
    print(f'  cached_token_stats = {stats}')
    sys.exit(1)

for k in required_keys:
    v = stats[k]
    if not isinstance(v, int) or v < 0:
        print(f'FAIL: $label — {k} should be a non-negative integer, got {v!r}')
        sys.exit(1)

print(f'PASS: $label — cached_token_stats present with all required keys')
print(f'  num_vllm_cached_tokens:          {stats[\"num_vllm_cached_tokens\"]}')
print(f'  num_lmcache_cached_tokens:       {stats[\"num_lmcache_cached_tokens\"]}')
print(f'  num_lmcache_extra_cached_tokens: {stats[\"num_lmcache_extra_cached_tokens\"]}')
"
}

validate_warm_hit() {
    local cold_file="$1"
    local warm_file="$2"

    python3 -c "
import json, sys

with open('$cold_file') as f:
    cold = json.load(f)
with open('$warm_file') as f:
    warm = json.load(f)

cold_stats = cold['kv_transfer_params']['cached_token_stats']
warm_stats = warm['kv_transfer_params']['cached_token_stats']

cold_lmcache = cold_stats['num_lmcache_cached_tokens']
warm_lmcache = warm_stats['num_lmcache_cached_tokens']

print(f'Cold request — num_lmcache_cached_tokens: {cold_lmcache}')
print(f'Warm request — num_lmcache_cached_tokens: {warm_lmcache}')

if warm_lmcache <= cold_lmcache:
    print(f'FAIL: warm request should have more LMCache hits than cold request')
    print(f'  cold={cold_lmcache}, warm={warm_lmcache}')
    sys.exit(1)

if warm_lmcache == 0:
    print('FAIL: warm request has 0 LMCache cached tokens (cache not populated?)')
    sys.exit(1)

print(f'PASS: warm request has more LMCache hits ({warm_lmcache} > {cold_lmcache})')
"
}

# ── Step 1: Cold request (populates LMCache) ──────────────────
echo "============================================"
echo "=== Step 1: Cold request ==="
echo "============================================"
if ! send_request "Cold" "$STATS_DIR/cold_response.json"; then
    exit 1
fi
if ! validate_stats_present "Cold" "$STATS_DIR/cold_response.json"; then
    exit 1
fi
echo ""

# Small delay to let the store operation complete in LMCache
sleep 2

# ── Step 2: Warm request (same prompt, should hit cache) ──────
echo "============================================"
echo "=== Step 2: Warm request ==="
echo "============================================"
if ! send_request "Warm" "$STATS_DIR/warm_response.json"; then
    exit 1
fi
if ! validate_stats_present "Warm" "$STATS_DIR/warm_response.json"; then
    exit 1
fi
echo ""

# ── Step 3: Validate cache hit improvement ────────────────────
echo "============================================"
echo "=== Step 3: Validate cache hit ==="
echo "============================================"
if ! validate_warm_hit "$STATS_DIR/cold_response.json" "$STATS_DIR/warm_response.json"; then
    exit 1
fi
echo ""

# ── Step 4: Verify opt-in behavior ────────────────────────────
# Request WITHOUT kv_transfer_params should NOT have stats in response.
echo "============================================"
echo "=== Step 4: Verify opt-in (no stats without opt-in) ==="
echo "============================================"

echo "--- Sending request without kv_transfer_params ---"
http_code=$(curl -s -o "$STATS_DIR/no_opt_in_response.json" -w "%{http_code}" \
    -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}],
        \"max_tokens\": 1
    }")

if [ "$http_code" -ne 200 ]; then
    echo "FAIL: no-opt-in request returned HTTP $http_code"
    exit 1
fi

python3 -c "
import json, sys

with open('$STATS_DIR/no_opt_in_response.json') as f:
    data = json.load(f)

kv_params = data.get('kv_transfer_params')
if kv_params is not None:
    print(f'FAIL: kv_transfer_params should be absent without opt-in, got {kv_params}')
    sys.exit(1)

print('PASS: kv_transfer_params correctly absent when not opted in')
"
echo ""

# ── Summary ───────────────────────────────────────────────────
echo "============================================"
echo "=== Cache Stats Reporting Test PASSED ==="
echo "============================================"
