#!/usr/bin/env bash
# Start a node-local Mooncake owner (real client) for vLLM requester ranks.
#
# Usage:
#   MC_OWNER_DEVICE=rocep139s0,rocep140s0 \
#   bash start_mooncake_owner.sh --cpu-mem-size 80
#
#   MC_OWNER_DEVICE=rocep139s0,rocep140s0 \
#   bash start_mooncake_owner.sh --cpu-mem-size 80 --disk-size 400 --bg
#
#   bash start_mooncake_owner.sh --stop
#
# Flags:
#   --cpu-mem-size <GB>   Owner CPU memory size in GB (required unless --stop)
#   --disk-size <GB>      Enable disk offload with the given quota in GB
#   --disk-path <dir>     Disk offload directory (default: /mnt/data/mooncake_offload)
#   --host <ip-or-host>   Owner host/IP to advertise (default: first hostname -I entry)
#   --segment-port <p>    Advertised owner segment port (default: 50053)
#   --rpc-port <p>        Owner RPC port (default: 50052)
#   --master <addr>       Mooncake master address (default: MOONCAKE_MASTER or 127.0.0.1:50051)
#   --metadata-server <u> Metadata server URL (default: MOONCAKE_TE_META_DATA_SERVER or http://127.0.0.1:8080/metadata)
#   --protocol <name>     Transport protocol (default: MOONCAKE_PROTOCOL or rdma)
#   --device <names>      RDMA device name(s) (default: MC_OWNER_DEVICE)
#   --threads <n>         Owner RPC worker threads (default: 1)
#   --bg                  Run in background
#   --stop                Stop the backgrounded owner launched by this script
#
# The owner auto-selects active RDMA devices when MC_OWNER_DEVICE/--device is
# unset. For diagnostics, use:
#   bash scripts/mooncake/recommend_mooncake_rnic_config.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/network_utils.sh"
source "${SCRIPT_DIR}/rdma_config_utils.sh"

HOST_TAG="$(hostname -s 2>/dev/null || echo local)"
PID_FILE="${SCRIPT_DIR}/mooncake_owner.${HOST_TAG}.pid"
LOG_FILE="${SCRIPT_DIR}/mooncake_owner.${HOST_TAG}.log"

OPT_BG=false
OPT_STOP=false
CPU_MEM_GB=""
DISK_GB=""
DISK_PATH="/mnt/data/mooncake_offload"
OWNER_HOST=""
SEGMENT_PORT="${MC_OWNER_SEGMENT_PORT:-50053}"
RPC_PORT="${MC_OWNER_RPC_PORT:-50052}"
MASTER_SERVER="${MOONCAKE_MASTER:-127.0.0.1:50051}"
METADATA_SERVER="${MOONCAKE_TE_META_DATA_SERVER:-http://127.0.0.1:8080/metadata}"
PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
DEVICE_NAME="${MC_OWNER_DEVICE:-}"
THREADS="${MC_OWNER_THREADS:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-mem-size)
            CPU_MEM_GB="$2"; shift 2 ;;
        --disk-size)
            DISK_GB="$2"; shift 2 ;;
        --disk-path)
            DISK_PATH="$2"; shift 2 ;;
        --host)
            OWNER_HOST="$2"; shift 2 ;;
        --segment-port)
            SEGMENT_PORT="$2"; shift 2 ;;
        --rpc-port)
            RPC_PORT="$2"; shift 2 ;;
        --master)
            MASTER_SERVER="$2"; shift 2 ;;
        --metadata-server)
            METADATA_SERVER="$2"; shift 2 ;;
        --protocol)
            PROTOCOL="$2"; shift 2 ;;
        --device)
            DEVICE_NAME="$2"; shift 2 ;;
        --threads)
            THREADS="$2"; shift 2 ;;
        --bg)
            OPT_BG=true; shift ;;
        --stop)
            OPT_STOP=true; shift ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--bg] [--stop] --cpu-mem-size <GB> [--disk-size <GB>]" >&2
            exit 1
            ;;
    esac
done

is_our_owner() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null || return 1
    [[ "$(ps -o comm= -p "$pid" 2>/dev/null)" == mooncake_clie* ]] || return 1
    [[ "$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$(id -u)" ]]
}

print_rnic_hint() {
    echo "Hint: use scripts/mooncake/recommend_mooncake_rnic_config.sh to inspect local Mooncake RDMA detection." >&2
}

detect_owner_devices() {
    local detected_gid_index=""
    local rdma_entries=()
    local rdma_entry=""
    local device_name=""
    local devices=()

    [[ "$PROTOCOL" == "rdma" || "$PROTOCOL" == "efa" ]] || return 0
    [[ -z "${DEVICE_NAME//[[:space:]]/}" ]] || return 0

    if [[ -z "${MC_GID_INDEX:-}" ]]; then
        detected_gid_index="$(detect_preferred_gid_index)" || {
            echo "Error: failed to auto-detect a usable RDMA GID index." >&2
            return 1
        }
        export MC_GID_INDEX="$detected_gid_index"
    fi

    mapfile -t rdma_entries < <(list_active_rdma_devices "${MC_GID_INDEX:-}")
    if [[ "${#rdma_entries[@]}" -eq 0 ]]; then
        if [[ -n "${MC_GID_INDEX:-}" ]]; then
            echo "Error: no active RDMA devices matched MC_GID_INDEX=${MC_GID_INDEX}." >&2
        else
            echo "Error: no active RDMA devices were discovered." >&2
        fi
        return 1
    fi

    for rdma_entry in "${rdma_entries[@]}"; do
        IFS='|' read -r device_name _rest <<< "$rdma_entry"
        unset IFS
        devices+=("$device_name")
    done

    DEVICE_NAME="$(IFS=,; printf '%s' "${devices[*]}")"
}

wait_for_owner_ready() {
    local pid=$1
    local timeout_s=${2:-30}
    wait_for_tcp_port "$RPC_PORT" "$timeout_s" "$pid"
}

stop_owner() {
    [[ -f "$PID_FILE" ]] || return 0
    local pid
    pid=$(<"$PID_FILE")

    if ! is_our_owner "$pid"; then
        rm -f "$PID_FILE"
        return 0
    fi

    echo "Stopping mooncake_client owner (PID $pid) ..."
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
    stop_owner
    exit 0
fi

if [[ -z "$CPU_MEM_GB" ]]; then
    echo "Error: --cpu-mem-size is required unless --stop is used." >&2
    exit 1
fi

if [[ -z "$OWNER_HOST" ]]; then
    OWNER_HOST="$(detect_local_advertise_host "$OWNER_HOST" || true)"
fi
[[ -n "${OWNER_HOST//[[:space:]]/}" ]] || {
    echo "Error: failed to determine owner host automatically; pass --host <ip-or-host>." >&2
    exit 1
}

if [[ -f "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
    if is_our_owner "$pid"; then
        echo "Error: mooncake_client owner is already running with PID $pid. Use '$0 --stop' first." >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

if tcp_port_is_listening "$RPC_PORT"; then
    echo "Error: owner RPC port ${RPC_PORT} is already in use." >&2
    exit 1
fi

if [[ "$PROTOCOL" == "rdma" || "$PROTOCOL" == "efa" ]]; then
    if ! detect_owner_devices; then
        print_rnic_hint
        exit 1
    fi
    if ! validate_rdma_device_list "$DEVICE_NAME"; then
        print_rnic_hint
        exit 1
    fi
fi

if [[ -n "$DISK_GB" ]]; then
    DISK_BYTES=$(( DISK_GB * 1073741824 ))
    HEADROOM_BYTES=$(( 40 * 1073741824 ))

    export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$DISK_PATH"
    export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="${MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR:-bucket_storage_backend}"
    export MOONCAKE_BUCKET_EVICTION_POLICY="${MOONCAKE_BUCKET_EVICTION_POLICY:-lru}"
    export MOONCAKE_USE_URING="${MOONCAKE_USE_URING:-true}"
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES="${MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES:-$((32*1024*1024*1024))}"
    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="$DISK_BYTES"
    export MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS="${MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS:-1}"
    if (( DISK_BYTES > HEADROOM_BYTES )); then
        export MOONCAKE_BUCKET_MAX_TOTAL_SIZE=$(( DISK_BYTES - HEADROOM_BYTES ))
    else
        export MOONCAKE_BUCKET_MAX_TOTAL_SIZE=$(( DISK_BYTES * 9 / 10 ))
    fi

    mkdir -p "$MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"
fi

CMD=(
    mooncake_client
    --master_server_address="$MASTER_SERVER"
    --metadata_server="$METADATA_SERVER"
    --host="${OWNER_HOST}:${SEGMENT_PORT}"
    --port="$RPC_PORT"
    --protocol="$PROTOCOL"
    --device_names="$DEVICE_NAME"
    --global_segment_size="${CPU_MEM_GB}GB"
    --threads="$THREADS"
)

if [[ -n "$DISK_GB" ]]; then
    CMD+=( --enable_offload=true )
fi

echo "Starting Mooncake owner"
echo "  Master:        $MASTER_SERVER"
echo "  Metadata:      $METADATA_SERVER"
echo "  Host:          ${OWNER_HOST}:${SEGMENT_PORT}"
echo "  RPC:           127.0.0.1:${RPC_PORT}"
echo "  Protocol:      $PROTOCOL"
echo "  Device:        ${DEVICE_NAME:-<none>}"
echo "  CPU memory:    ${CPU_MEM_GB} GB"
if [[ -n "$DISK_GB" ]]; then
    echo "  Disk offload:  ON (${DISK_GB} GB at ${MOONCAKE_OFFLOAD_FILE_STORAGE_PATH})"
else
    echo "  Disk offload:  OFF"
fi

if $OPT_BG; then
    : > "$LOG_FILE"
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    if ! wait_for_owner_ready "$(cat "$PID_FILE")" 30; then
        echo "  ERROR: mooncake_client failed to become ready" >&2
        tail -n 50 "$LOG_FILE" >&2 || true
        rm -f "$PID_FILE"
        exit 1
    fi
    echo "  PID:           $(<"$PID_FILE") (written to $PID_FILE)"
    echo "  Log:           $LOG_FILE"
    echo
    echo "Stop with:       bash $0 --stop"
else
    echo "  (foreground — Ctrl-C to stop)"
    exec "${CMD[@]}"
fi
