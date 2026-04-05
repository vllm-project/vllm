#!/usr/bin/env bash
# Start a local Mooncake master server for benchmarking.
#
# The master provides two services:
#   - RPC (port 50051): object metadata, allocation, eviction
#   - HTTP metadata (port 8080): transfer engine peer discovery
#
# Usage:
#   bash start_mooncake_master.sh                          # foreground
#   bash start_mooncake_master.sh --bg                     # background (PID → mooncake_master.pid)
#   bash start_mooncake_master.sh --enable-offload --bg    # background with disk offloading
#   bash start_mooncake_master.sh --stop                   # kill backgrounded master
#
# Flags can appear in any order; --stop is processed first and ignores others.
#
# Environment variables (all optional):
#   MC_RPC_PORT       - Master RPC port           (default: 50051)
#   MC_HTTP_PORT      - HTTP metadata port         (default: 8080)
#   MC_METRICS_PORT   - Prometheus metrics port    (default: 9003)
#   MC_RPC_THREADS    - RPC worker threads         (default: 4)
#   MC_LEASE_TTL      - KV lease TTL in ms         (default: 30000)
#   MC_EVICT_HI       - Eviction high watermark    (default: 0.95)
#   MC_EVICT_RATIO    - Fraction to evict per pass (default: 0.1)


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/mooncake_master.pid"
LOG_FILE="$SCRIPT_DIR/mooncake_master.log"

# --- Parse flags -----------------------------------------------------------
OPT_BG=false
OPT_STOP=false
OPT_OFFLOAD=false

for arg in "$@"; do
    case "$arg" in
        --bg)              OPT_BG=true ;;
        --stop)            OPT_STOP=true ;;
        --enable-offload)  OPT_OFFLOAD=true ;;
        *)
            echo "Unknown flag: $arg" >&2
            echo "Usage: $0 [--bg] [--enable-offload] [--stop]" >&2
            exit 1
            ;;
    esac
done

# --- Master settings -------------------------------------------------------
MC_RPC_PORT="${MC_RPC_PORT:-50051}"
MC_HTTP_PORT="${MC_HTTP_PORT:-8080}"
MC_METRICS_PORT="${MC_METRICS_PORT:-9003}"
MC_RPC_THREADS="${MC_RPC_THREADS:-4}"
MC_LEASE_TTL="${MC_LEASE_TTL:-30000}"
MC_EVICT_HI="${MC_EVICT_HI:-0.95}"
MC_EVICT_RATIO="${MC_EVICT_RATIO:-0.1}"


# --- Helper: stop a previously backgrounded master -------------------------
is_our_master() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null || return 1
    [[ "$(ps -o comm= -p "$pid" 2>/dev/null)" == mooncake_maste* ]] || return 1
    [[ "$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$(id -u)" ]]
}

stop_master() {
    [[ -f "$PID_FILE" ]] || return 0
    local pid=$(<"$PID_FILE")

    if ! is_our_master "$pid"; then
        rm -f "$PID_FILE"
        return 0
    fi

    echo "Stopping mooncake_master (PID $pid) ..."
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 5); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null; sleep 1; fi
    rm -f "$PID_FILE"
}

if $OPT_STOP; then
    stop_master
    exit 0
fi

# Ensure no stale instance
stop_master 2>/dev/null || true

# --- Build command ----------------------------------------------------------
CMD=(
    mooncake_master
    -rpc_port="$MC_RPC_PORT"
    -rpc_thread_num="$MC_RPC_THREADS"
    -rpc_address=0.0.0.0
    -rpc_enable_tcp_no_delay=true
    -enable_http_metadata_server=true
    -http_metadata_server_host=0.0.0.0
    -http_metadata_server_port="$MC_HTTP_PORT"
    -enable_metric_reporting=true
    -metrics_port="$MC_METRICS_PORT"
    -default_kv_lease_ttl="$MC_LEASE_TTL"
    -eviction_high_watermark_ratio="$MC_EVICT_HI"
    -eviction_ratio="$MC_EVICT_RATIO"
    -logtostderr
)

if $OPT_OFFLOAD; then
    CMD+=( -enable_offload=true )
    # Note: do NOT pass -root_fs_dir here. That enables direct disk writes
    # during put, which segfaults on GPU-resident data. Disk offload for
    # vLLM works via the client-side FileStorage heartbeat path instead:
    # CPU memory -> SSD (background), not GPU -> SSD (inline).
fi

# --- Start ------------------------------------------------------------------
echo "Starting mooncake_master"
echo "  RPC:      0.0.0.0:${MC_RPC_PORT}"
echo "  HTTP:     0.0.0.0:${MC_HTTP_PORT}/metadata"
echo "  Metrics:  0.0.0.0:${MC_METRICS_PORT}"
if $OPT_OFFLOAD; then
    echo "  Offload:  ON"
else
    echo "  Offload:  OFF (pass --enable-offload to enable)"
fi

if $OPT_BG; then
    : > "$LOG_FILE"
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    sleep 1
    if ! is_our_master "$(cat "$PID_FILE")"; then
        echo "  ERROR: mooncake_master failed to stay running" >&2
        tail -n 50 "$LOG_FILE" >&2 || true
        rm -f "$PID_FILE"
        exit 1
    fi
    echo "  PID:      $(<"$PID_FILE") (written to $PID_FILE)"
    echo "  Log:      $LOG_FILE"
    echo ""
    echo "Stop with:  bash $0 --stop"
else
    echo "  (foreground — Ctrl-C to stop)"
    exec "${CMD[@]}"
fi
