#!/usr/bin/env bash
# End-to-end test for LMCache HTTP API endpoints and CLI commands.
#
# Part 1 — HTTP API endpoints:
#   Systematically exercises every HTTP endpoint on the LMCache
#   multiprocess HTTP server (port 8080) with a live engine + vLLM.
#
# Part 2 — CLI commands:
#   Tests `lmcache describe` and `lmcache kvcache clear`
#   against the running server.
#
# Requires: LMCache MP server + vLLM launched by launch-processes.sh.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration (inherited from run-single-test.sh)
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

HTTP_DIR="$RESULTS_DIR/http_api"
mkdir -p "$HTTP_DIR"

BASE_URL="http://localhost:${LMCACHE_HTTP_PORT}"

echo "=== HTTP API & CLI Test ==="
echo "Model: $MODEL"
echo "vLLM port: $VLLM_PORT"
echo "LMCache HTTP: $BASE_URL"
echo "Results dir: $HTTP_DIR"
echo ""

# ── Counters ────────────────────────────────────────────────
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# ── Helpers ─────────────────────────────────────────────────

pass() {
    local label="$1"
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
    echo "PASS: $label"
}

fail() {
    local label="$1"
    shift
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
    echo "FAIL: $label — $*"
}

check_http() {
    local label="$1"
    local method="$2"
    local path="$3"
    local output="$4"
    local expected_code="${5:-200}"
    local body="${6:-}"

    local curl_args=(-s -o "$output" -w "%{http_code}")
    case "$method" in
        GET)  curl_args+=("${BASE_URL}${path}") ;;
        POST)
            curl_args+=(-X POST -H "Content-Type: application/json")
            [ -n "$body" ] && curl_args+=(-d "$body")
            curl_args+=("${BASE_URL}${path}")
            ;;
        PUT)
            curl_args+=(-X PUT -H "Content-Type: application/json")
            [ -n "$body" ] && curl_args+=(-d "$body")
            curl_args+=("${BASE_URL}${path}")
            ;;
        DELETE)
            curl_args+=(-X DELETE "${BASE_URL}${path}")
            ;;
    esac

    local http_code
    http_code=$(curl "${curl_args[@]}")

    if [ "$http_code" -ne "$expected_code" ]; then
        fail "$label" "expected HTTP $expected_code, got $http_code"
        echo "  Response body:"
        cat "$output" 2>/dev/null || true
        echo ""
        return 1
    fi
    pass "$label (HTTP $http_code)"
    return 0
}

assert_json_key() {
    local label="$1"
    local file="$2"
    local key="$3"
    local expected_value="${4:-}"

    python3 -c "
import json, sys
with open('$file') as f:
    data = json.load(f)
keys = '$key'.split('.')
cur = data
for k in keys:
    if not isinstance(cur, dict) or k not in cur:
        print('FAIL: $label — key \"$key\" not found')
        sys.exit(1)
    cur = cur[k]
if '$expected_value' and str(cur) != '$expected_value':
    print('FAIL: $label — expected \"$expected_value\", got ' + repr(cur))
    sys.exit(1)
" || {
        fail "$label" "JSON key assertion failed for key '$key'"
        return 1
    }
    pass "$label (key '$key')"
}

assert_contains() {
    local label="$1"
    local file="$2"
    local pattern="$3"

    if grep -q "$pattern" "$file" 2>/dev/null; then
        pass "$label (contains '$pattern')"
    else
        fail "$label" "output does not contain '$pattern'"
        return 1
    fi
}

# ================================================================
#  PART 1: HTTP API ENDPOINT TESTS
# ================================================================

echo ""
echo "========================================================"
echo "=== Part 1: HTTP API Endpoint Tests ==="
echo "========================================================"

# ── Step 0: Warm up cache via vLLM ─────────────────────────
echo ""
echo "============================================"
echo "=== Step 0: Warm up cache ==="
echo "============================================"

warmup_code=$(curl -s -o "$HTTP_DIR/warmup.json" -w "%{http_code}" \
    -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Explain the concept of caching in computer science.\"}],
        \"max_tokens\": 1
    }")

if [ "$warmup_code" -ne 200 ]; then
    echo "FATAL: warmup request failed with HTTP $warmup_code"
    cat "$HTTP_DIR/warmup.json" 2>/dev/null || true
    exit 1
fi
echo "Warmup request successful (HTTP 200)"
sleep 2

# ── Step 1: Root + Healthcheck ─────────────────────────────
echo ""
echo "============================================"
echo "=== Step 1: Root + Healthcheck ==="
echo "============================================"

check_http "GET /" GET "/" "$HTTP_DIR/root.json" 200
assert_json_key "GET / — status" "$HTTP_DIR/root.json" "status" "ok"
assert_json_key "GET / — service" "$HTTP_DIR/root.json" "service" "LMCache HTTP API"

check_http "GET /healthcheck" GET "/healthcheck" "$HTTP_DIR/healthcheck.json" 200
assert_json_key "GET /healthcheck — status" "$HTTP_DIR/healthcheck.json" "status" "healthy"

# ── Step 2: Configuration ──────────────────────────────────
echo ""
echo "============================================"
echo "=== Step 2: Configuration (/config) ==="
echo "============================================"

check_http "GET /config" GET "/config" "$HTTP_DIR/config.json" 200

python3 -c "
import json, sys
with open('$HTTP_DIR/config.json') as f:
    data = json.load(f)
required = ['mp', 'storage_manager', 'observability', 'http']
missing = [k for k in required if k not in data]
if missing:
    print(f'FAIL: /config missing keys: {missing}')
    sys.exit(1)
print('All config sections present: ' + ', '.join(required))
" && pass "GET /config — required keys present" \
  || fail "GET /config — required keys present" "missing config sections"

# ── Step 3: Status ─────────────────────────────────────────
echo ""
echo "============================================"
echo "=== Step 3: Status (/status) ==="
echo "============================================"

check_http "GET /status" GET "/status" "$HTTP_DIR/status.json" 200

python3 -c "
import json, sys
with open('$HTTP_DIR/status.json') as f:
    data = json.load(f)
if not isinstance(data, dict):
    print('FAIL: /status did not return a JSON dict')
    sys.exit(1)
print('Status response is a valid dict with ' + str(len(data)) + ' keys')
" && pass "GET /status — valid JSON dict" \
  || fail "GET /status — valid JSON dict" "response is not a dict"

# ── Step 4: Metrics ────────────────────────────────────────
echo ""
echo "============================================"
echo "=== Step 4: Metrics (/metrics) ==="
echo "============================================"

check_http "GET /metrics" GET "/metrics" "$HTTP_DIR/metrics.txt" 200
assert_contains "GET /metrics — has lmcache metrics" "$HTTP_DIR/metrics.txt" "lmcache_"
assert_contains "GET /metrics — Prometheus format" "$HTTP_DIR/metrics.txt" "# "

# ── Step 5: Environment ────────────────────────────────────
echo ""
echo "============================================"
echo "=== Step 5: Environment (/env) ==="
echo "============================================"

check_http "GET /env" GET "/env" "$HTTP_DIR/env.json" 200

python3 -c "
import json, sys
with open('$HTTP_DIR/env.json') as f:
    data = json.load(f)
if not isinstance(data, dict):
    print('FAIL: /env did not return a JSON dict')
    sys.exit(1)
if 'PATH' not in data:
    print('FAIL: /env missing PATH key')
    sys.exit(1)
print('/env returned ' + str(len(data)) + ' environment variables')
" && pass "GET /env — valid JSON with PATH" \
  || fail "GET /env — valid JSON with PATH" "validation failed"

# ── Step 6: Log Level ──────────────────────────────────────
echo ""
echo "============================================"
echo "=== Step 6: Log Level (/loglevel) ==="
echo "============================================"

check_http "GET /loglevel (list)" GET "/loglevel" "$HTTP_DIR/loglevel_list.txt" 200
assert_contains "GET /loglevel — header" "$HTTP_DIR/loglevel_list.txt" "Loggers and Levels"

check_http "GET /loglevel (set DEBUG)" GET "/loglevel?logger_name=lmcache&level=DEBUG" "$HTTP_DIR/loglevel_set.txt" 200
assert_contains "GET /loglevel (set DEBUG) — confirmation" "$HTTP_DIR/loglevel_set.txt" "Set lmcache level to DEBUG"

check_http "GET /loglevel (invalid level)" GET "/loglevel?logger_name=lmcache&level=INVALID_LEVEL" "$HTTP_DIR/loglevel_bad.txt" 400

# ── Step 7: Threads ────────────────────────────────────────
echo ""
echo "============================================"
echo "=== Step 7: Threads ==="
echo "============================================"

check_http "GET /threads" GET "/threads" "$HTTP_DIR/threads.txt" 200
assert_contains "GET /threads — header" "$HTTP_DIR/threads.txt" "Thread Summary"

check_http "GET /periodic-threads" GET "/periodic-threads" "$HTTP_DIR/periodic_threads.json" 200

python3 -c "
import json, sys
with open('$HTTP_DIR/periodic_threads.json') as f:
    data = json.load(f)
if 'summary' not in data or 'threads' not in data:
    print('FAIL: /periodic-threads missing summary or threads key')
    sys.exit(1)
if 'total_count' not in data['summary']:
    print('FAIL: /periodic-threads summary missing total_count')
    sys.exit(1)
print('/periodic-threads: ' + str(data['summary']['total_count']) + ' threads registered')
" && pass "GET /periodic-threads — structure" \
  || fail "GET /periodic-threads — structure" "validation failed"

check_http "GET /periodic-threads-health" GET "/periodic-threads-health" "$HTTP_DIR/periodic_health.json" 200
assert_json_key "GET /periodic-threads-health — healthy key" "$HTTP_DIR/periodic_health.json" "healthy"

# ── Step 8: Clear Cache ────────────────────────────────────
echo ""
echo "============================================"
echo "=== Step 8: Clear Cache (/cache/clear) ==="
echo "============================================"

check_http "POST /cache/clear" POST "/cache/clear" "$HTTP_DIR/clear_cache.json" 200
assert_json_key "POST /cache/clear — status" "$HTTP_DIR/clear_cache.json" "status" "ok"

# ── Step 9: Quota CRUD ─────────────────────────────────────
echo ""
echo "============================================"
echo "=== Step 9: Quota CRUD ==="
echo "============================================"

QUOTA_SALT="test_salt_${BUILD_ID}"

# 9a: Create quota
check_http "PUT /quota/${QUOTA_SALT}" PUT "/quota/${QUOTA_SALT}" "$HTTP_DIR/quota_put.json" 200 '{"limit_gb": 1.0}'
assert_json_key "PUT quota — status" "$HTTP_DIR/quota_put.json" "status" "ok"

# 9b: Read quota
check_http "GET /quota/${QUOTA_SALT}" GET "/quota/${QUOTA_SALT}" "$HTTP_DIR/quota_get.json" 200

python3 -c "
import json, sys
with open('$HTTP_DIR/quota_get.json') as f:
    data = json.load(f)
if not data.get('exists'):
    print('FAIL: quota should exist after PUT')
    sys.exit(1)
limit = data.get('limit_gb', 0)
if abs(limit - 1.0) > 0.01:
    print(f'FAIL: expected limit_gb ~1.0, got {limit}')
    sys.exit(1)
print(f'Quota exists with limit_gb={limit}')
" && pass "GET /quota — exists with correct limit" \
  || fail "GET /quota — exists with correct limit" "validation failed"

# 9c: List quotas
check_http "GET /quota (list)" GET "/quota" "$HTTP_DIR/quota_list.json" 200
assert_json_key "GET /quota (list) — users key" "$HTTP_DIR/quota_list.json" "users"

# 9d: Delete quota
check_http "DELETE /quota/${QUOTA_SALT}" DELETE "/quota/${QUOTA_SALT}" "$HTTP_DIR/quota_delete.json" 200
assert_json_key "DELETE quota — status" "$HTTP_DIR/quota_delete.json" "status" "removed"

# 9e: Verify deletion
check_http "GET /quota/${QUOTA_SALT} (after delete)" GET "/quota/${QUOTA_SALT}" "$HTTP_DIR/quota_gone.json" 200

python3 -c "
import json, sys
with open('$HTTP_DIR/quota_gone.json') as f:
    data = json.load(f)
if data.get('exists'):
    print('FAIL: quota should not exist after DELETE')
    sys.exit(1)
print('Quota correctly gone after DELETE')
" && pass "GET /quota after DELETE — not exists" \
  || fail "GET /quota after DELETE — not exists" "quota still exists"

# 9f: Delete non-existent
check_http "DELETE /quota/nonexistent_salt" DELETE "/quota/nonexistent_salt" "$HTTP_DIR/quota_notfound.json" 200
assert_json_key "DELETE nonexistent — status" "$HTTP_DIR/quota_notfound.json" "status" "not_found"

# 9g: Error — negative limit
check_http "PUT /quota (negative limit)" PUT "/quota/err_salt" "$HTTP_DIR/quota_neg.json" 400 '{"limit_gb": -1.0}'

# 9h: Error — missing key
check_http "PUT /quota (missing limit_gb)" PUT "/quota/err_salt" "$HTTP_DIR/quota_badkey.json" 400 '{"bad_key": 1}'

# ── Step 10: Version ───────────────────────────────────────
echo ""
echo "============================================"
echo "=== Step 10: Version ==="
echo "============================================"

check_http "GET /version" GET "/version" "$HTTP_DIR/version.txt" 200
check_http "GET /lmc_version" GET "/lmc_version" "$HTTP_DIR/lmc_version.txt" 200
check_http "GET /commit_id" GET "/commit_id" "$HTTP_DIR/commit_id.txt" 200

# ================================================================
#  PART 2: CLI COMMAND TESTS
# ================================================================

echo ""
echo "========================================================"
echo "=== Part 2: CLI Command Tests ==="
echo "========================================================"

# ── Step 11: lmcache describe kvcache ──────────────────────
echo ""
echo "============================================"
echo "=== Step 11: lmcache describe kvcache ==="
echo "============================================"

if lmcache describe kvcache --url "${BASE_URL}" > "$HTTP_DIR/describe.txt" 2>&1; then
    pass "lmcache describe kvcache — exit code 0"
else
    fail "lmcache describe kvcache — exit code 0" "command returned non-zero"
    echo "  Output:"
    cat "$HTTP_DIR/describe.txt" 2>/dev/null || true
fi

assert_contains "describe — Health section" "$HTTP_DIR/describe.txt" "Health"
assert_contains "describe — L1 capacity section" "$HTTP_DIR/describe.txt" "L1 capacity"

# ── Step 12: lmcache kvcache clear ─────────────────────────
echo ""
echo "============================================"
echo "=== Step 12: lmcache kvcache clear ==="
echo "============================================"

if lmcache kvcache clear --url "${BASE_URL}" > "$HTTP_DIR/kvcache_clear.txt" 2>&1; then
    pass "lmcache kvcache clear — exit code 0"
else
    fail "lmcache kvcache clear — exit code 0" "command returned non-zero"
    echo "  Output:"
    cat "$HTTP_DIR/kvcache_clear.txt" 2>/dev/null || true
fi

assert_contains "kvcache clear — OK" "$HTTP_DIR/kvcache_clear.txt" "OK"

# ================================================================
#  SUMMARY
# ================================================================

echo ""
echo "========================================================"
echo "=== Summary ==="
echo "========================================================"
echo "Tests run:    $TESTS_RUN"
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $TESTS_FAILED"
echo ""

if [ "$TESTS_FAILED" -gt 0 ]; then
    echo "============================================"
    echo "=== HTTP API & CLI Test FAILED ==="
    echo "============================================"
    exit 1
fi

echo "============================================"
echo "=== HTTP API & CLI Test PASSED ==="
echo "============================================"
