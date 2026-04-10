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
#   bash start_mooncake_master.sh --bg --env-file /path/to/mooncake_master.env
#                                                      # background + connection env file
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
#   MC_LEASE_TTL      - KV lease TTL in ms         (default: 1800000)
#   MC_EVICT_HI       - Eviction high watermark    (default: 0.95)
#   MC_EVICT_RATIO    - Fraction to evict per pass (default: 0.1)
#   MC_MASTER_HOST    - Advertised host/IP for workers (default: first hostname -I entry)


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/mooncake_master.pid"
LOG_FILE="$SCRIPT_DIR/mooncake_master.log"

# --- Parse flags -----------------------------------------------------------
OPT_BG=false
OPT_STOP=false
OPT_OFFLOAD=false
ENV_FILE=""
MASTER_HOST="${MC_MASTER_HOST:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bg)
            OPT_BG=true
            shift
            ;;
        --stop)
            OPT_STOP=true
            shift
            ;;
        --enable-offload)
            OPT_OFFLOAD=true
            shift
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --host)
            MASTER_HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--bg] [--enable-offload] [--stop] [--env-file <path>] [--host <ip-or-host>]" >&2
            exit 1
            ;;
    esac
done

# --- Master settings -------------------------------------------------------
MC_RPC_PORT="${MC_RPC_PORT:-50051}"
MC_HTTP_PORT="${MC_HTTP_PORT:-8080}"
MC_METRICS_PORT="${MC_METRICS_PORT:-9003}"
MC_RPC_THREADS="${MC_RPC_THREADS:-4}"
MC_LEASE_TTL="${MC_LEASE_TTL:-1800000}"
MC_EVICT_HI="${MC_EVICT_HI:-0.95}"
MC_EVICT_RATIO="${MC_EVICT_RATIO:-0.1}"

resolve_master_bin() {
    local wrapper_bin
    local wrapper_dir
    local candidate

    wrapper_bin="$(command -v mooncake_master)"
    wrapper_dir="$(cd "$(dirname "$wrapper_bin")" && pwd)"

    shopt -s nullglob
    for candidate in "$wrapper_dir"/../lib/python*/site-packages/mooncake/mooncake_master; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            shopt -u nullglob
            return 0
        fi
    done
    shopt -u nullglob

    printf '%s\n' "$wrapper_bin"
}

MASTER_BIN="$(resolve_master_bin)"


detect_master_host() {
    if [[ -n "${MASTER_HOST//[[:space:]]/}" ]]; then
        printf '%s\n' "$MASTER_HOST"
        return 0
    fi
    if command -v hostname >/dev/null 2>&1; then
        local host_ip
        host_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
        if [[ -n "${host_ip//[[:space:]]/}" ]]; then
            printf '%s\n' "$host_ip"
            return 0
        fi
        local host_name
        host_name="$(hostname -f 2>/dev/null || hostname 2>/dev/null || true)"
        if [[ -n "${host_name//[[:space:]]/}" ]]; then
            printf '%s\n' "$host_name"
            return 0
        fi
    fi
    return 1
}


write_master_env_file() {
    local env_file=$1
    [[ -n "${env_file//[[:space:]]/}" ]] || return 0

    mkdir -p "$(dirname "$env_file")"
    cat >"$env_file" <<EOF
export MOONCAKE_MASTER=${MASTER_HOST}:${MC_RPC_PORT}
export MOONCAKE_TE_META_DATA_SERVER=http://${MASTER_HOST}:${MC_HTTP_PORT}/metadata
EOF
}


# --- Helper: stop a previously backgrounded master -------------------------
is_our_master() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null || return 1
    [[ "$(ps -o comm= -p "$pid" 2>/dev/null)" == mooncake_maste* ]] || return 1
    [[ "$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$(id -u)" ]]
}

master_ports_ready() {
    ss -ltnH "( sport = :${MC_RPC_PORT} or sport = :${MC_HTTP_PORT} or sport = :${MC_METRICS_PORT} )" \
        2>/dev/null | grep -q LISTEN
}

any_master_port_in_use() {
    ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq \
        "(^|[:.])(${MC_RPC_PORT}|${MC_HTTP_PORT}|${MC_METRICS_PORT})$"
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
    for _ in $(seq 5); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PID_FILE"
}

if $OPT_STOP; then
    stop_master
    exit 0
fi

if [[ -f "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
    if is_our_master "$pid"; then
        MASTER_HOST="$(detect_master_host)" || {
            echo "Error: failed to determine Mooncake master host automatically; pass --host <ip-or-host> or set MC_MASTER_HOST." >&2
            exit 1
        }
        write_master_env_file "$ENV_FILE"
        if master_ports_ready; then
            echo "Reusing existing mooncake_master (PID $pid)."
            if [[ -n "${ENV_FILE//[[:space:]]/}" ]]; then
                echo "  Env file: $ENV_FILE"
            fi
            exit 0
        fi
        echo "Error: mooncake_master PID file points to a running process (PID $pid), but the expected ports are not all listening." >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

if any_master_port_in_use; then
    echo "Error: one of the Mooncake master ports (${MC_RPC_PORT}, ${MC_HTTP_PORT}, ${MC_METRICS_PORT}) is already in use." >&2
    exit 1
fi

MASTER_HOST="$(detect_master_host)" || {
    echo "Error: failed to determine Mooncake master host automatically; pass --host <ip-or-host> or set MC_MASTER_HOST." >&2
    exit 1
}

# --- Build command ----------------------------------------------------------
CMD=(
    "$MASTER_BIN"
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
echo "  Advertise: ${MASTER_HOST}"
if $OPT_OFFLOAD; then
    echo "  Offload:  ON"
else
    echo "  Offload:  OFF (pass --enable-offload to enable)"
fi

if $OPT_BG; then
    : > "$LOG_FILE"
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    for _ in $(seq 30); do
        if is_our_master "$(cat "$PID_FILE")" \
            && ss -ltn "( sport = :${MC_RPC_PORT} or sport = :${MC_HTTP_PORT} or sport = :${MC_METRICS_PORT} )" \
                2>/dev/null | grep -q LISTEN; then
            break
        fi
        sleep 1
    done
    if ! is_our_master "$(cat "$PID_FILE")"; then
        echo "  ERROR: mooncake_master failed to stay running" >&2
        tail -n 50 "$LOG_FILE" >&2 || true
        rm -f "$PID_FILE"
        exit 1
    fi
    write_master_env_file "$ENV_FILE"
    echo "  PID:      $(<"$PID_FILE") (written to $PID_FILE)"
    echo "  Log:      $LOG_FILE"
    if [[ -n "${ENV_FILE//[[:space:]]/}" ]]; then
        echo "  Env file: $ENV_FILE"
    fi
    echo ""
    echo "Stop with:  bash $0 --stop"
else
    write_master_env_file "$ENV_FILE"
    echo "  (foreground — Ctrl-C to stop)"
    exec "${CMD[@]}"
fi
