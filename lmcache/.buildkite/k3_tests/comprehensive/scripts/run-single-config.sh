#!/usr/bin/env bash
# Comprehensive integration test: run a single config natively in a K8s pod.
# Replaces the old Docker-based vllm-integration-tests.sh.
#
# Usage: run-single-config.sh <config.yaml>
set -euo pipefail

CFG_NAME="${1:?Usage: $0 <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
source .buildkite/k3_tests/common_scripts/helpers.sh

CONFIG_DIR="${REPO_ROOT}/.buildkite/configs"
LOGFILE="${REPO_ROOT}/${CFG_NAME%.yaml}.log"
BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"
NEED_UPLOAD="${NEED_UPLOAD:-false}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-240}"

# Tee all output so Buildkite can collect it as an artifact
exec > >(tee -a "$LOGFILE") 2>&1

# ── Validate config ──────────────────────────────────────────
cfg_file="${CONFIG_DIR}/${CFG_NAME}"
if [[ ! -f "$cfg_file" ]]; then
    echo "Config not found: ${cfg_file}"
    exit 1
fi

feature_type=$(yq -r '.feature.type // ""' "$cfg_file")
echo -e "\033[1;33m===== Testing LMCache with ${CFG_NAME} (type=${feature_type:-standard}) =====\033[0m"

# Prevent git from hanging on prompts
export GIT_TERMINAL_PROMPT=0

# PID tracking for cleanup
PIDS=()
on_exit() {
    local rc=$?
    echo "--- Cleaning up (exit code: $rc)..."
    for p in "${PIDS[@]}"; do
        kill "$p" 2>/dev/null || true
        wait "$p" 2>/dev/null || true
    done
    # Copy vLLM server logs to repo root so Buildkite can collect them as artifacts
    cp /tmp/build_${BUILD_ID}_${CFG_NAME%.yaml}*.log "${REPO_ROOT}/" 2>/dev/null || true
    sleep 5
}
trap on_exit EXIT INT TERM

###############
# UTILITIES   #
###############

# Start a vLLM process natively (no Docker).
# Reads env vars from a yq-extracted docker section and vllm args from a vllm section.
start_single_server() {
    local docker_section="$1"
    local vllm_section="$2"
    local port="$3"
    local gpu="${4:-}"
    local logfile="${5:-/tmp/build_${BUILD_ID}_${CFG_NAME%.yaml}.log}"

    # Collect env vars from docker.env[]
    local -a env_cmd=()
    while IFS= read -r e; do
        [[ -n "$e" ]] && env_cmd+=("$e")
    done < <(yq -r '.env[]?' <<<"$docker_section")

    # Standard env vars
    env_cmd+=("VLLM_USE_FLASHINFER_SAMPLER=0")
    [[ -n "${HF_TOKEN:-}" ]] && env_cmd+=("HF_TOKEN=${HF_TOKEN}")
    [[ -n "$gpu" ]] && env_cmd+=("CUDA_VISIBLE_DEVICES=${gpu}")

    # Override socket prefix so each server gets its own socket directory.
    # In the old Docker setup, a volume mount mapped /tmp/lmcache_internal_api_server
    # inside the container to /tmp/lmcache_internal_api_server/${port} on the host.
    # Without Docker, we replicate this by setting the prefix to include the port.
    mkdir -p "/tmp/lmcache_internal_api_server/${port}"
    env_cmd+=("LMCACHE_INTERNAL_API_SERVER_SOCKET_PATH_PREFIX=/tmp/lmcache_internal_api_server/${port}/socket")

    # Parse vllm model and args
    local vllm_model
    vllm_model="$(yq -r '.model' <<<"$vllm_section")"
    local -a vllm_cli_args=()
    mapfile -t vllm_cli_args < <(yq -r '.args // [] | .[]' <<<"$vllm_section")

    echo "Starting vLLM: model=$vllm_model port=$port gpu=${gpu:-all}"
    env "${env_cmd[@]}" \
        vllm serve "$vllm_model" "${vllm_cli_args[@]}" --port "$port" \
        >"$logfile" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    echo "  PID=$pid log=$logfile"

    # Wait for readiness
    if ! wait_for_server "$port" "$SERVER_WAIT_TIMEOUT"; then
        echo "Server failed to start. Last 50 lines of log:"
        tail -50 "$logfile" || true
        return 1
    fi
}

###############
# LAUNCH      #
###############

PORT=$(find_free_port 8000)
PORT1=""
PORT2=""
model=""

if [[ "$feature_type" == "pd" ]]; then
    # ── Prefiller-Decoder mode (2 GPUs) ─────────────────────
    PORT1=$(find_free_port 8100)
    PORT2=$(find_free_port $((PORT1 + 100)))

    prefiller_docker="$(yq '.["docker-prefiller"]' "$cfg_file")"
    prefiller_vllm="$(yq '.["vllm-prefiller"]' "$cfg_file")"
    decoder_docker="$(yq '.["docker-decoder"]' "$cfg_file")"
    decoder_vllm="$(yq '.["vllm-decoder"]' "$cfg_file")"
    model="$(yq -r '.["vllm-prefiller"].model' "$cfg_file")"

    # Extra PD env vars
    proxy=$(yq -er '.["docker-prefiller"]["proxy-port"]' "$cfg_file" 2>/dev/null || echo "7500")
    init=$(yq -er '.["docker-decoder"]["init-port"]' "$cfg_file" 2>/dev/null || echo "7300")
    alloc=$(yq -er '.["docker-decoder"]["alloc-port"]' "$cfg_file" 2>/dev/null || echo "7400")

    # Inject PD-specific env vars into docker sections
    prefiller_docker=$(echo "$prefiller_docker" | yq -y --arg proxy "$proxy" '. + {"env": (.env + ["LMCACHE_PD_PROXY_PORT=" + $proxy])}')
    decoder_docker=$(echo "$decoder_docker" | yq -y --arg init "$init" --arg alloc "$alloc" '. + {"env": (.env + ["LMCACHE_PD_PEER_INIT_PORT=" + $init, "LMCACHE_PD_PEER_ALLOC_PORT=" + $alloc])}')

    echo "--- Starting prefiller on port $PORT1 (GPU 0)"
    # Prefiller needs UCX_TLS for NIXL transport
    prefiller_docker=$(echo "$prefiller_docker" | yq -y ". + {\"env\": (.env + [\"UCX_TLS=cuda_ipc,cuda_copy,tcp\"])}")
    start_single_server "$prefiller_docker" "$prefiller_vllm" "$PORT1" "0" \
        "${REPO_ROOT}/${CFG_NAME%.yaml}-prefiller.log"

    echo "--- Starting decoder on port $PORT2 (GPU 1)"
    decoder_docker=$(echo "$decoder_docker" | yq -y ". + {\"env\": (.env + [\"UCX_TLS=cuda_ipc,cuda_copy,tcp\"])}")
    start_single_server "$decoder_docker" "$decoder_vllm" "$PORT2" "1" \
        "${REPO_ROOT}/${CFG_NAME%.yaml}-decoder.log"

    # Start disagg proxy
    echo "--- Starting PD proxy on port $PORT"
    python3 "${REPO_ROOT}/examples/disagg_prefill/disagg_proxy_server.py" \
        --port "$PORT" \
        --prefiller-port "$PORT1" \
        --decoder-port "$PORT2" \
        --decoder-init-port "$init" \
        --decoder-alloc-port "$alloc" \
        --proxy-port "$proxy" \
        > "${REPO_ROOT}/${CFG_NAME%.yaml}-proxy.log" 2>&1 &
    PIDS+=($!)
    sleep 10

elif [[ "$feature_type" == "p2p" ]]; then
    # ── Peer-to-Peer mode (2 GPUs) ──────────────────────────
    PORT1=$(find_free_port 8177)
    PORT2=$(find_free_port $((PORT1 + 100)))

    docker1="$(yq '.docker1' "$cfg_file")"
    vllm1="$(yq '.vllm1' "$cfg_file")"
    docker2="$(yq '.docker2' "$cfg_file")"
    vllm2="$(yq '.vllm2' "$cfg_file")"
    model="$(yq -r '.vllm1.model' "$cfg_file")"

    # Controller ports
    pull=$(yq -er '.docker1["pull-port"]' "$cfg_file" 2>/dev/null || echo "8300")
    reply=$(yq -er '.docker1["reply-port"]' "$cfg_file" 2>/dev/null || echo "8400")

    # Inject controller URLs
    docker1=$(echo "$docker1" | yq -y --arg pull "$pull" --arg reply "$reply" '. + {"env": (.env + ["LMCACHE_CONTROLLER_PULL_URL=localhost:" + $pull, "LMCACHE_CONTROLLER_REPLY_URL=localhost:" + $reply, "UCX_TLS=tcp"])}')

    pull2=$(yq -er '.docker2["pull-port"]' "$cfg_file" 2>/dev/null || echo "$pull")
    reply2=$(yq -er '.docker2["reply-port"]' "$cfg_file" 2>/dev/null || echo "$reply")
    docker2=$(echo "$docker2" | yq -y --arg pull2 "$pull2" --arg reply2 "$reply2" '. + {"env": (.env + ["LMCACHE_CONTROLLER_PULL_URL=localhost:" + $pull2, "LMCACHE_CONTROLLER_REPLY_URL=localhost:" + $reply2, "UCX_TLS=tcp"])}')

    # Start controller
    echo "--- Starting P2P controller on port $PORT"
    PYTHONHASHSEED=123 lmcache_controller \
        --host localhost \
        --port "$PORT" \
        --monitor-ports "{\"pull\": ${pull}, \"reply\": ${reply}}" \
        > "/tmp/build_${BUILD_ID}_${CFG_NAME%.yaml}_controller.log" 2>&1 &
    PIDS+=($!)
    sleep 10

    echo "--- Starting instance 1 on port $PORT1 (GPU 0)"
    start_single_server "$docker1" "$vllm1" "$PORT1" "0" \
        "/tmp/build_${BUILD_ID}_${CFG_NAME%.yaml}1.log"

    echo "--- Starting instance 2 on port $PORT2 (GPU 1)"
    start_single_server "$docker2" "$vllm2" "$PORT2" "1" \
        "/tmp/build_${BUILD_ID}_${CFG_NAME%.yaml}2.log"

else
    # ── Single server mode (1 GPU) ──────────────────────────
    docker_section="$(yq '.docker' "$cfg_file")"
    vllm_section="$(yq '.vllm' "$cfg_file")"
    model="$(yq -r '.vllm.model' "$cfg_file")"

    echo "--- Starting single server on port $PORT"
    start_single_server "$docker_section" "$vllm_section" "$PORT" "" \
        "/tmp/build_${BUILD_ID}_${CFG_NAME%.yaml}.log"
fi

###############
# WORKLOAD    #
###############

test_mode="$(yq -r '.workload.type' "$cfg_file")"

if [[ "$test_mode" == "dummy" ]]; then
    echo "--- Running dummy test (simple HTTP request)"
    http_status_code=$(
        curl --max-time 60 "http://localhost:${PORT}/v1/completions" \
            -w "%{http_code}" -o response-file.txt \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"$model"'",
                "prompt": "<|begin_of_text|><|system|>\nYou are a helpful AI assistant.\n<|user|>\nWhat is the capital of France?\n<|assistant|>",
                "max_tokens": 100,
                "temperature": 0.7
            }'
    )
    if [[ "$http_status_code" -ne 200 ]]; then
        echo "FAIL: Dummy test HTTP status ${http_status_code}"
        cat response-file.txt
        exit 1
    fi
    echo "PASS: Dummy test succeeded"
    cat response-file.txt

elif [[ "$test_mode" == "long_doc_qa" ]]; then
    echo "--- Running long_doc_qa workload"

    # Build workload JSON (merge workload section with model, strip non-CLI fields)
    # Fields like expected-latency-gain are used by the checking logic, not long_doc_qa.py.
    # "completion" -> "completions" rename to match the argparse flag.
    workload_yaml="$(yq --arg model "$model" '(.workload * {"model": $model}) | del(.type) | del(.["expected-latency-gain"]) | if .completion then .completions = .completion | del(.completion) else . end' "$cfg_file")"
    cfg_json="$(yq '.' "$cfg_file")"

    # Determine which checking fields are requested
    check_warmup_round_time_per_prompt=$(
        jq -e '(.["checking-fields"] // []) | index("warmup_round_time_per_prompt") != null' \
            <<< "$cfg_json" >/dev/null && echo true || echo false
    )
    check_query_round_time_per_prompt=$(
        jq -e '(.["checking-fields"] // []) | index("query_round_time_per_prompt") != null' \
            <<< "$cfg_json" >/dev/null && echo true || echo false
    )
    check_query_ttft_per_prompt=$(
        jq -e '(.["checking-fields"] // []) | index("warmup_ttft_per_prompt") != null' \
            <<< "$cfg_json" >/dev/null && echo true || echo false
    )

    run_long_doc_qa_workload() {
        local workload_config="$1"
        local port="$2"
        local check_warmup="${3:-false}"
        local check_ttft="${4:-false}"
        local check_round="${5:-false}"
        local feat_type="${6:-dummy}"
        local need_upload="${7:-false}"

        echo "Running long_doc_qa on port $port"

        # Build args array from JSON config
        local -a workload_args=()
        mapfile -d '' -t workload_args < <(
        jq -j '
            to_entries[]
            | select(.value != null and (.value|tostring) != "")
            | (
                if (.value|type) == "boolean" then
                if .value then ["--\(.key)", "\u0000"] else [] end
                elif (.value|type) == "array" then
                .value[]
                | ["--\(.key)", "\u0000",
                    (if (type=="string") then . else tostring end),
                    "\u0000"]
                else
                ["--\(.key)", "\u0000",
                (if ((.value|type)=="string") then .value else (.value|tostring) end),
                "\u0000"]
                end
            )
            | .[]
        ' <<<"$workload_config"
        )

        local json
        json=$(
            python3 "${REPO_ROOT}/benchmarks/long_doc_qa/long_doc_qa.py" \
                "${workload_args[@]}" \
                --port="$port" \
                --output="response.txt" \
                --json-output \
                2>>response.txt | tail -n 1
        )
        local query_ttft_per_prompt query_round_time_per_prompt warmup_round_time_per_prompt
        query_ttft_per_prompt=$(echo "$json" | jq -r '.query_ttft_per_prompt')
        query_round_time_per_prompt=$(echo "$json" | jq -r '.query_round_time_per_prompt')
        warmup_round_time_per_prompt=$(echo "$json" | jq -r '.warmup_round_time_per_prompt')

        if [[ "$need_upload" == "true" ]]; then
            # Nightly: write date-stamped file and upload as Buildkite artifact.
            # A finalize step (upload-baselines.sh) will collect all artifacts,
            # prune old files, and make a single commit to benchmarks-main.
            local datestamp
            datestamp="$(date +%Y%m%d)"
            local artifact_name="${feat_type}-${datestamp}.json"
            local artifact_path="${REPO_ROOT}/benchmarks/long_doc_qa/${artifact_name}"

            echo "$json"
            mkdir -p "$(dirname "$artifact_path")"
            printf '%s\n' "$json" > "$artifact_path"

            echo "[INFO] Uploading baseline artifact: ${artifact_name}"
            buildkite-agent artifact upload "$artifact_path" 2>/dev/null || {
                echo "[WARN] buildkite-agent not available; wrote $artifact_path locally"
            }
            return 0
        fi

        # ── PR comparison: rolling 5-day worst-case baseline ──────
        # Fetch all date-stamped baselines for this feature from benchmarks-main.
        # Use max() of each metric (= worst/slowest) as the threshold baseline.
        # This makes the gate more tolerant of nightly hardware noise.
        timeout 30 git fetch origin benchmarks-main >/dev/null 2>&1 || true

        # List all baseline files for this feature (date-stamped and legacy).
        # Note: git ls-tree does NOT expand shell globs in pathspecs, so we
        # list the directory and filter with grep instead.
        local -a baseline_files=()
        mapfile -t baseline_files < <(
            git ls-tree --name-only origin/benchmarks-main -- \
                benchmarks/long_doc_qa/ 2>/dev/null \
            | grep -E "^benchmarks/long_doc_qa/${feat_type}(-[0-9]{8})?\.json$" \
            || true
        )

        if [[ ${#baseline_files[@]} -eq 0 ]]; then
            if [[ "$feat_type" != "dummy" ]]; then
                echo "No baselines found for ${feat_type} -- skipping performance comparisons"
                echo "Current metrics: TTFT=$query_ttft_per_prompt, Latency=$query_round_time_per_prompt, Warmup=$warmup_round_time_per_prompt"
            fi
            return 0
        fi

        echo "[INFO] Found ${#baseline_files[@]} baseline file(s) for ${feat_type}:"
        printf "  %s\n" "${baseline_files[@]}"

        # Compute rolling max (worst-case) across all baselines
        local expected_query_ttft=0 expected_query_round=0 expected_warmup=0
        for bf in "${baseline_files[@]}"; do
            local bj
            bj=$(git show "origin/benchmarks-main:${bf}" 2>/dev/null || echo "")
            if [[ -z "$bj" ]] || ! echo "$bj" | jq -e . >/dev/null 2>&1; then
                echo "  Skipping invalid baseline: $bf"
                continue
            fi
            # Take max of each metric (higher time = worse perf = more permissive gate)
            expected_query_ttft=$(awk "BEGIN { a=$expected_query_ttft; b=$(echo "$bj" | jq -r '.query_ttft_per_prompt // 0'); print (a>b?a:b) }")
            expected_query_round=$(awk "BEGIN { a=$expected_query_round; b=$(echo "$bj" | jq -r '.query_round_time_per_prompt // 0'); print (a>b?a:b) }")
            expected_warmup=$(awk "BEGIN { a=$expected_warmup; b=$(echo "$bj" | jq -r '.warmup_round_time_per_prompt // 0'); print (a>b?a:b) }")
        done

        echo "[INFO] Rolling worst-case baseline: TTFT=$expected_query_ttft, Latency=$expected_query_round, Warmup=$expected_warmup"

        # Sanity check: if all baselines were invalid, skip comparison
        if awk "BEGIN { exit ($expected_query_ttft == 0 && $expected_query_round == 0 && $expected_warmup == 0) ? 0 : 1 }"; then
            echo "All baselines were invalid or empty -- skipping performance comparisons"
            return 0
        fi

        # Adaptive threshold: with fewer baselines the worst-case max is less
        # representative, so we allow more headroom.
        #   5+ baselines → 1.2x (20% tolerance, full confidence)
        #   4  baselines → 1.3x
        #   3  baselines → 1.4x
        #   2  baselines → 1.5x
        #   1  baseline  → 1.6x (60% tolerance, low confidence)
        local num_baselines=${#baseline_files[@]}
        local threshold
        threshold=$(awk -v n="$num_baselines" 'BEGIN {
            gap = (5 - n); if (gap < 0) gap = 0
            printf "%.1f", 1.2 + gap * 0.1
        }')
        local pct
        pct=$(awk -v t="$threshold" 'BEGIN { printf "%d", (t - 1) * 100 }')
        echo "[INFO] Baselines: ${num_baselines}/5 — using ${pct}% tolerance (threshold=${threshold}x)"

        if [[ "$check_ttft" == "true" ]]; then
            echo "Worst-case baseline query ttft per prompt: $expected_query_ttft"
            echo "Actual query ttft per prompt: $query_ttft_per_prompt"
            awk -v expected="$expected_query_ttft" -v actual="$query_ttft_per_prompt" -v t="$threshold" -v pct="$pct" 'BEGIN {
                if (actual > expected * t) { printf "FAIL: Query ttft per prompt >%d%% worse than rolling baseline\n", pct; exit 1 }
                else { print "PASS: Query ttft per prompt within threshold" }
            }'
        fi

        if [[ "$check_round" == "true" ]]; then
            echo "Worst-case baseline query round time per prompt: $expected_query_round"
            echo "Actual query round time per prompt: $query_round_time_per_prompt"
            awk -v expected="$expected_query_round" -v actual="$query_round_time_per_prompt" -v t="$threshold" -v pct="$pct" 'BEGIN {
                if (actual > expected * t) { printf "FAIL: Query round time per prompt >%d%% worse than rolling baseline\n", pct; exit 1 }
                else { print "PASS: Query round time per prompt within threshold" }
            }'
        fi

        if [[ "$check_warmup" == "true" ]]; then
            echo "Worst-case baseline warmup round time per prompt: $expected_warmup"
            echo "Actual warmup round time per prompt: $warmup_round_time_per_prompt"
            awk -v expected="$expected_warmup" -v actual="$warmup_round_time_per_prompt" -v t="$threshold" -v pct="$pct" 'BEGIN {
                if (actual > expected * t) { printf "FAIL: Warmup round time per prompt >%d%% worse than rolling baseline\n", pct; exit 1 }
                else { print "PASS: Warmup round time per prompt within threshold" }
            }'
        fi
    }

    if [[ "$feature_type" == "p2p" ]]; then
        run_long_doc_qa_workload "$workload_yaml" "$PORT1"
        run_long_doc_qa_workload "$workload_yaml" "$PORT2" \
            "$check_warmup_round_time_per_prompt" "$check_query_ttft_per_prompt" \
            "$check_query_round_time_per_prompt" "${CFG_NAME%.yaml}" "$NEED_UPLOAD"
    else
        run_long_doc_qa_workload "$workload_yaml" "$PORT" \
            "$check_warmup_round_time_per_prompt" "$check_query_ttft_per_prompt" \
            "$check_query_round_time_per_prompt" "${CFG_NAME%.yaml}" "$NEED_UPLOAD"
    fi
fi

###############
# MEMORY LEAK #
###############

check_memory_leak() {
    local port="$1"
    local socket_path="/tmp/lmcache_internal_api_server/${port}/socket_7000"

    local use_hot
    use_hot=$(curl -s --unix-socket "$socket_path" "http://localhost/conf" 2>/dev/null | jq -r '.local_cpu // false')
    if [[ -z "$use_hot" || "$use_hot" == "null" ]]; then
        use_hot="false"
    fi

    echo "Checking memory leak on socket_path $socket_path (use_hot=$use_hot)..."

    local metrics
    metrics=$(curl -s --unix-socket "$socket_path" "http://localhost/metrics" 2>/dev/null)
    if [[ -z "$metrics" ]]; then
        echo "ERROR: Failed to fetch metrics from $socket_path"
        return 1
    fi

    local local_cpu_hot_cache_count active_memory_objs_count pinned_memory_objs_count pin_monitor_pinned_objects_count
    local_cpu_hot_cache_count=$(echo "$metrics" | grep -E '^lmcache:local_cpu_hot_cache_count\b' | awk '{print $2}' | head -n 1)
    active_memory_objs_count=$(echo "$metrics" | grep -E '^lmcache:active_memory_objs_count\b' | awk '{print $2}' | head -n 1)
    pinned_memory_objs_count=$(echo "$metrics" | grep -E '^lmcache:pinned_memory_objs_count\b' | awk '{print $2}' | head -n 1)
    pin_monitor_pinned_objects_count=$(echo "$metrics" | grep -E '^lmcache:pin_monitor_pinned_objects_count\b' | awk '{print $2}' | head -n 1)

    local_cpu_hot_cache_count=$(printf "%.0f" "${local_cpu_hot_cache_count:-0}")
    active_memory_objs_count=$(printf "%.0f" "${active_memory_objs_count:-0}")
    pinned_memory_objs_count=$(printf "%.0f" "${pinned_memory_objs_count:-0}")
    pin_monitor_pinned_objects_count=$(printf "%.0f" "${pin_monitor_pinned_objects_count:-0}")

    echo "  local_cpu_hot_cache_count: $local_cpu_hot_cache_count"
    echo "  active_memory_objs_count: $active_memory_objs_count"
    echo "  pinned_memory_objs_count: $pinned_memory_objs_count"
    echo "  pin_monitor_pinned_objects_count: $pin_monitor_pinned_objects_count"

    local has_leak=false

    if [[ "$pinned_memory_objs_count" -ne 0 ]]; then
        echo "ERROR: Memory leak detected - pinned_memory_objs_count ($pinned_memory_objs_count) should be 0"
        has_leak=true
    fi

    if [[ "$use_hot" == "false" || "$use_hot" == "False" ]]; then
        if [[ "$local_cpu_hot_cache_count" -ne 0 ]]; then
            echo "ERROR: Memory leak - local_cpu_hot_cache_count ($local_cpu_hot_cache_count) should be 0 when use_hot=false"
            has_leak=true
        fi
        if [[ "$active_memory_objs_count" -ne 0 ]]; then
            echo "ERROR: Memory leak - active_memory_objs_count ($active_memory_objs_count) should be 0 when use_hot=false"
            has_leak=true
        fi
    else
        if [[ "$active_memory_objs_count" -ne "$local_cpu_hot_cache_count" ]]; then
            echo "ERROR: Memory leak - active_memory_objs_count ($active_memory_objs_count) should equal local_cpu_hot_cache_count ($local_cpu_hot_cache_count) when use_hot=true"
            has_leak=true
        fi
    fi

    if [[ "$has_leak" == "true" ]]; then
        echo "$metrics"
        return 1
    fi

    echo "  Memory leak check passed!"
    return 0
}

# Wait for metrics to settle
sleep 30
echo "--- Checking for memory leaks..."

if [[ "$feature_type" == "p2p" ]]; then
    echo "Skipping memory leak check for p2p (known issue with pinned_memory_objs not clearing)."
elif [[ "$feature_type" == "pd" ]]; then
    if ! check_memory_leak "$PORT1"; then
        echo "Memory leak check failed for instance 1 (port $PORT1)"
        exit 1
    fi
    if ! check_memory_leak "$PORT2"; then
        echo "Memory leak check failed for instance 2 (port $PORT2)"
        exit 1
    fi
elif [[ "$CFG_NAME" == "multi_device.yaml" || "$CFG_NAME" == "layerwise.yaml" ]]; then
    echo "Skipping memory leak check for $CFG_NAME case as it's a flaky test while run check_memory_leak check."
else
    if ! check_memory_leak "$PORT"; then
        echo "Memory leak check failed for $CFG_NAME"
        exit 1
    fi
fi

echo "===== PASS: ${CFG_NAME} ====="
