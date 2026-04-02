#!/usr/bin/env bash
# Start a local Mooncake master server for benchmarking.
#
# The master provides two services:
#   - RPC (port 50051): object metadata, allocation, eviction
#   - HTTP metadata (port 8080): transfer engine peer discovery
#
# Usage:
#   bash start_mooncake_master.sh          # foreground
#   bash start_mooncake_master.sh --bg     # background (PID written to mooncake_master.pid)
#   bash start_mooncake_master.sh --stop   # kill backgrounded master
#
# Environment variables (all optional):
#   MC_RPC_PORT       - Master RPC port          (default: 50051)
#   MC_HTTP_PORT      - HTTP metadata port        (default: 8080)
#   MC_METRICS_PORT   - Prometheus metrics port   (default: 9003)
#   MC_RPC_THREADS    - RPC worker threads        (default: 4)
#   MC_LEASE_TTL      - KV lease TTL in ms        (default: 30000)
#   MC_EVICT_HI       - Eviction high watermark   (default: 0.95)
#   MC_EVICT_RATIO    - Fraction to evict per pass (default: 0.1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/mooncake_master.pid"
LOG_FILE="$SCRIPT_DIR/mooncake_master.log"

MC_RPC_PORT="${MC_RPC_PORT:-50051}"
MC_HTTP_PORT="${MC_HTTP_PORT:-8080}"
MC_METRICS_PORT="${MC_METRICS_PORT:-9003}"
MC_RPC_THREADS="${MC_RPC_THREADS:-4}"
MC_LEASE_TTL="${MC_LEASE_TTL:-30000}"
MC_EVICT_HI="${MC_EVICT_HI:-0.95}"
MC_EVICT_RATIO="${MC_EVICT_RATIO:-0.1}"

stop_master() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(<"$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping mooncake_master (PID $pid)"
            kill "$pid"
            wait "$pid" 2>/dev/null || true
        else
            echo "mooncake_master (PID $pid) is not running"
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file found at $PID_FILE"
    fi
}

if [[ "${1:-}" == "--stop" ]]; then
    stop_master
    exit 0
fi

# Ensure no stale instance
stop_master 2>/dev/null || true

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

echo "Starting mooncake_master"
echo "  RPC:      0.0.0.0:${MC_RPC_PORT}"
echo "  HTTP:     0.0.0.0:${MC_HTTP_PORT}/metadata"
echo "  Metrics:  0.0.0.0:${MC_METRICS_PORT}"

if [[ "${1:-}" == "--bg" ]]; then
    "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "  PID:      $(<"$PID_FILE") (written to $PID_FILE)"
    echo "  Log:      $LOG_FILE"
    echo ""
    echo "Stop with:  bash $0 --stop"
else
    echo "  (foreground — Ctrl-C to stop)"
    exec "${CMD[@]}"
fi
