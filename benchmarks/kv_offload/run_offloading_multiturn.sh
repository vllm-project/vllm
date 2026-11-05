#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Multi-turn KV offloading benchmark.
#
# Serves a model once per KV offloading configuration and drives it with a
# guidellm multi-turn workload: conversations of TURNS turns that open with one
# of PREFIX_COUNT shared prefixes of PREFIX_TOKENS tokens and add PROMPT_TOKENS
# new tokens per turn. When the prefixes and conversations outgrow the GPU KV
# cache, prefix hits come from the offloading tiers, so throughput and TTFT
# depend on how well the tiers keep and reload them.
#
# Configurations (CONFIGS, space separated):
#   gpu     - no offloading, GPU prefix caching only (baseline)
#   cpu     - OffloadingConnector with CPUOffloadingSpec
#   cpu_fs  - TieringOffloadingSpec: CPU primary tier and an fs secondary tier
#
# Example (the setup from #60128: gpt-oss-120b, TP=2 on 2xH100):
#   MODEL=openai/gpt-oss-120b TP_SIZE=2 CPU_BYTES=25769803776 \
#   FS_DIR=/mnt/nvme/kv CONFIGS="cpu cpu_fs" N_READ_THREADS=4 N_WRITE_THREADS=4 \
#   KV_EXTRA='"enable_cross_layers_blocks": "True", "block_size": 1024' \
#   SERVE_ARGS="--gpu-memory-utilization 0.7 --no-disable-hybrid-kv-cache-manager" \
#   bash benchmarks/kv_offload/run_offloading_multiturn.sh
#
# With TARGET=http://host:port the script benchmarks an already running server
# instead of starting one, under the configuration name "external".
#
# CONCURRENCY may list several values (e.g. "16 64"); without TARGET each value
# gets a fresh server. Results from earlier runs in the same RESULTS_DIR are
# included in the summary table.
#
# Run inside the vLLM virtualenv (so `vllm` and `python` resolve to it).
# Requires guidellm (tested with 0.8.1; uv pip install guidellm).
#
# Note: until #57422 lands, the cpu_fs configuration leaves its CPU tier
# (CPU_BYTES) in /dev/shm/vllm_offload_<engine_id>.mmap after the server stops.

set -euo pipefail

TARGET=${TARGET:-}
TARGET=${TARGET%/}
[ -n "$TARGET" ] || : "${MODEL:?set MODEL (or TARGET)}"
MODEL=${MODEL:-}
TP_SIZE=${TP_SIZE:-1}
PORT=${PORT:-8000}
SERVE_ARGS=${SERVE_ARGS:-}
CONFIGS=${CONFIGS:-"gpu cpu cpu_fs"}
CPU_BYTES=${CPU_BYTES:-}
FS_DIR=${FS_DIR:-}
N_READ_THREADS=${N_READ_THREADS:-16}
N_WRITE_THREADS=${N_WRITE_THREADS:-16}
KV_EXTRA=${KV_EXTRA:-}
TOKENIZER=${TOKENIZER:-}   # defaults to the served model name

CONCURRENCY=${CONCURRENCY:-64}
TURNS=${TURNS:-5}
PREFIX_TOKENS=${PREFIX_TOKENS:-20000}
PREFIX_COUNT=${PREFIX_COUNT:-}
PROMPT_TOKENS=${PROMPT_TOKENS:-4096}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-1}
DURATION=${DURATION:-700}
SEED=${SEED:-889}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
STOP_TIMEOUT=${STOP_TIMEOUT:-180}
RESULTS_DIR=${RESULTS_DIR:-"offloading_multiturn_$(date +%Y%m%d_%H%M%S)"}

command -v guidellm >/dev/null || { echo "guidellm not found (uv pip install guidellm)" >&2; exit 1; }
command -v python >/dev/null || { echo "python not found; run inside the vLLM virtualenv" >&2; exit 1; }
mkdir -p "$RESULTS_DIR"

# Check every configuration before the first run starts.
if [ -z "$TARGET" ]; then
    for cfg in $CONFIGS; do
        case "$cfg" in
            gpu) ;;
            cpu) : "${CPU_BYTES:?set CPU_BYTES for the cpu configuration}" ;;
            cpu_fs) : "${CPU_BYTES:?set CPU_BYTES for cpu_fs}" "${FS_DIR:?set FS_DIR for cpu_fs}" ;;
            *) echo "unknown configuration: $cfg" >&2; exit 1 ;;
        esac
    done
fi

SERVER_PID=""
FS_RUN_DIR=""

stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        # vLLM shuts down on SIGTERM as on SIGINT, and SIGTERM is not ignored
        # by background jobs.
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq "$STOP_TIMEOUT"); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server $SERVER_PID did not stop on SIGTERM, sending SIGKILL" >&2
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    # The next configuration must not find this server still answering.
    for _ in $(seq "$STOP_TIMEOUT"); do
        port_in_use || return 0
        sleep 1
    done
    echo "Port $PORT still answers after stopping the server" >&2
    exit 1
}

cleanup() {
    stop_server
    # Only the per-run directory this script created is removed.
    if [ -n "$FS_RUN_DIR" ] && [ -d "$FS_RUN_DIR" ]; then
        rm -rf -- "$FS_RUN_DIR" || echo "Could not remove $FS_RUN_DIR" >&2
    fi
    FS_RUN_DIR=""
}
trap cleanup EXIT
trap 'exit 130' INT TERM

kv_transfer_config() {
    local cfg=$1 extra=""
    [ -n "$KV_EXTRA" ] && extra=", $KV_EXTRA"
    case "$cfg" in
        gpu)
            echo "" ;;
        cpu)
            cat <<EOF
{"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
 "kv_connector_extra_config": {"cpu_bytes_to_use": $CPU_BYTES$extra}}
EOF
            ;;
        cpu_fs)
            cat <<EOF
{"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
 "kv_connector_extra_config": {"spec_name": "TieringOffloadingSpec",
  "cpu_bytes_to_use": $CPU_BYTES$extra,
  "secondary_tiers": [{"type": "fs", "root_dir": "$FS_RUN_DIR",
   "n_read_threads": $N_READ_THREADS, "n_write_threads": $N_WRITE_THREADS}]}}
EOF
            ;;
    esac
}

port_in_use() {
    curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models"
}

start_server() {
    local cfg=$1 log=$2 kv
    if [ "$cfg" = cpu_fs ]; then
        # A fresh directory per run, so every run starts with an empty FS tier.
        FS_RUN_DIR=$(mktemp -d "${FS_DIR%/}/vllm-offload-bench.XXXXXX")
    fi
    if port_in_use; then
        echo "Port $PORT is already in use; stop that server or set PORT" >&2
        exit 1
    fi
    kv=$(kv_transfer_config "$cfg")
    local args=(serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP_SIZE"
                --enable-prefix-caching)
    [ -n "$kv" ] && args+=(--kv-transfer-config "$kv")
    # shellcheck disable=SC2206
    [ -n "$SERVE_ARGS" ] && args+=($SERVE_ARGS)
    echo "$kv" > "${log%.log}_kv_transfer_config.json"
    vllm "${args[@]}" > "$log" 2>&1 &
    SERVER_PID=$!
    local start=$SECONDS
    until curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server exited during startup, last lines of $log:" >&2
            tail -n 30 "$log" >&2
            exit 1
        fi
        if (( SECONDS - start > READY_TIMEOUT )); then
            echo "Server not ready after ${READY_TIMEOUT}s" >&2
            exit 1
        fi
        sleep 5
    done
}

# A failed scrape leaves no file, so the summary reports N/A instead of 0.
scrape_metrics() {
    curl -sf --max-time 30 "$1/metrics" > "$2" && return 0
    rm -f "$2"
    echo "Could not read $1/metrics; offloading counters will be N/A" >&2
}

run_guidellm() {
    local target=$1 streams=$2 out=$3 prefix_count
    prefix_count=${PREFIX_COUNT:-$(( 4 * streams > 256 ? 4 * streams : 256 ))}
    local data
    data=$(printf '{"kind": "synthetic_text", "prompt_tokens": %d, "output_tokens": %d, "turns": %d, "prefix_buckets": [{"bucket_weight": 100, "prefix_count": %d, "prefix_tokens": %d}]}' \
        "$PROMPT_TOKENS" "$OUTPUT_TOKENS" "$TURNS" "$prefix_count" "$PREFIX_TOKENS")
    local tok=()
    [ -n "$TOKENIZER" ] && tok=(--tokenizer "kind=huggingface_auto,model=$TOKENIZER")
    guidellm run ${tok[@]+"${tok[@]}"} \
        --backend "kind=openai_http,target=$target,request_format=/v1/completions" \
        --profile "kind=concurrent,streams=$streams" \
        --constraint "kind=max_duration,seconds=$DURATION" \
        --seed "kind=static,value=$SEED" \
        --data "$data" \
        --output "kind=json,path=$out" \
        --disable-progress > "${out%.json}_guidellm.log" 2>&1
}

[ -n "$TARGET" ] && CONFIGS=external
for cfg in $CONFIGS; do
    for streams in $CONCURRENCY; do
        run="$RESULTS_DIR/${cfg}_c${streams}"
        echo "=== $cfg, concurrency $streams"
        if [ -n "$TARGET" ]; then
            target=$TARGET
        else
            start_server "$cfg" "${run}_server.log"
            target="http://127.0.0.1:$PORT"
        fi
        scrape_metrics "$target" "${run}_metrics_before.txt"
        run_guidellm "$target" "$streams" "${run}.json"
        scrape_metrics "$target" "${run}_metrics_after.txt"
        [ -n "$TARGET" ] || cleanup
    done
done

python - "$RESULTS_DIR" <<'EOF'
import glob, json, os, sys

def counters(path):
    """Sum of the offloading byte counters in a /metrics snapshot, or None."""
    if not os.path.exists(path):
        return None
    out = {"cpu_to_gpu": 0.0, "secondary_read": 0.0}
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("vllm:kv_offload_load_bytes_total{"):
            out["cpu_to_gpu"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:kv_offload_tiering_read_bytes_total{"):
            out["secondary_read"] += float(line.rsplit(" ", 1)[1])
    return out

rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*_c*.json")), key=os.path.getmtime):
    if path.endswith("_kv_transfer_config.json"):
        continue
    with open(path) as f:
        report = json.load(f)
    name = os.path.basename(path)[: -len(".json")]
    before = counters(path[: -len(".json")] + "_metrics_before.txt")
    after = counters(path[: -len(".json")] + "_metrics_after.txt")
    keys = ("cpu_to_gpu", "secondary_read")
    if before is None or after is None:
        delta = {k: None for k in keys}
    else:
        delta = {k: (after[k] - before[k]) / 1e9 for k in keys}
    for bench in report.get("benchmarks", []):
        m = bench["metrics"]
        ttft = m["time_to_first_token_ms"]["successful"]
        totals = m["request_totals"] if "request_totals" in m else {}
        rows.append((
            name,
            m["tokens_per_second"]["successful"]["mean"],
            ttft["mean"] / 1000,
            ttft["median"] / 1000,
            ttft["percentiles"]["p95"] / 1000,
            totals.get("successful", ""),
            totals.get("errored", ""),
            delta["cpu_to_gpu"],
            delta["secondary_read"],
        ))
header = ("run", "total tok/s", "TTFT mean s", "TTFT p50 s", "TTFT p95 s", "ok", "errors",
          "CPU->GPU GB", "secondary read GB")
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))
def gb(value):
    return "N/A" if value is None else f"{value:.1f}"

for r in rows:
    print(f"| {r[0]} | {r[1]:,.0f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]} | {r[6]} "
          f"| {gb(r[7])} | {gb(r[8])} |")
EOF
