#!/usr/bin/env bash
# Launch a node-local Mooncake owner and a vLLM server in one managed process
# tree so the owner is cleaned up automatically when serving exits.
#
# Required env:
#   MC_OWNER_CPU_MEM_GIB   Owner CPU memory size in GiB
#
# Optional env:
#   MC_GID_INDEX                    RDMA GID index filter for RNIC detection
#   MC_MASTER_ENV_FILE              Shared env file written by start_mooncake_master.sh
#   MC_OWNER_DISK_GIB               Owner disk offload quota in GiB
#   MC_OWNER_DISK_PATH              Base directory for managed owner disk offload data
#   MC_OWNER_HOST                   Advertised owner host/IP
#   MC_OWNER_RPC_PORT               Owner RPC port (default: 50052)
#   MC_OWNER_SEGMENT_PORT           Owner segment port (default: 50053)
#   MC_OWNER_READY_TIMEOUT_S        Readiness timeout in seconds (default: 60)
#   MC_OWNER_LOG_DIR                Directory for owner log file
#   MC_MOONCAKE_LOAD_ASYNC          Connector load_async flag (default: 1)
#   MC_MOONCAKE_ENABLE_CROSS_LAYERS_BLOCKS  Connector enable_cross_layers_blocks
#                                        flag (default: 1)
#
# The wrapper owns Mooncake-specific config:
#   - reads MOONCAKE_MASTER / metadata server from the master env file
#   - auto-detects gpu_bdf_rnic_map and owner RNIC union
#   - injects --kv-transfer-config for MooncakeStoreConnector

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rdma_config_utils.sh"

HOST_TAG="$(hostname -s 2>/dev/null || echo local)"

OWNER_CPU_MEM_GIB="${MC_OWNER_CPU_MEM_GIB:-}"
OWNER_DISK_GIB="${MC_OWNER_DISK_GIB:-}"
OWNER_DISK_PATH="${MC_OWNER_DISK_PATH:-}"
OWNER_DEVICE="${MC_OWNER_DEVICE:-}"
OWNER_HOST="${MC_OWNER_HOST:-}"
OWNER_RPC_PORT="${MC_OWNER_RPC_PORT:-50052}"
OWNER_SEGMENT_PORT="${MC_OWNER_SEGMENT_PORT:-50053}"
OWNER_READY_TIMEOUT_S="${MC_OWNER_READY_TIMEOUT_S:-60}"
OWNER_LOG_DIR="${MC_OWNER_LOG_DIR:-$SCRIPT_DIR}"
OWNER_LOG_FILE="${OWNER_LOG_DIR}/mooncake_owner.${HOST_TAG}.managed.log"
MASTER_ENV_FILE="${MC_MASTER_ENV_FILE:-}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
MOONCAKE_LOAD_ASYNC="${MC_MOONCAKE_LOAD_ASYNC:-1}"
MOONCAKE_ENABLE_CROSS_LAYERS_BLOCKS="${MC_MOONCAKE_ENABLE_CROSS_LAYERS_BLOCKS:-1}"

OWNER_PID=""
SERVER_PID=""
OWNER_DISK_RUN_PATH=""
GPU_BDF_RNIC_MAP_JSON=""

is_port_listening() {
    local port="$1"
    ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "(^|[:.])${port}$"
}

stop_pid() {
    local pid="$1"
    [[ -n "$pid" ]] || return 0
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

local_owner_segment_process_exists() {
    ps -eo user=,args= 2>/dev/null | awk \
        -v user_name="$(id -un)" \
        -v owner_host="$OWNER_HOST" \
        -v segment_port="$OWNER_SEGMENT_PORT" '
            $1 == user_name && $0 ~ /mooncake_client/ &&
            $0 ~ ("--host=" owner_host ":" segment_port) {
                found = 1
            }
            END { exit(found ? 0 : 1) }
        '
}

metadata_key_http_code() {
    local key="$1"
    curl -sS -o /dev/null -w '%{http_code}' \
        --get \
        --data-urlencode "key=${key}" \
        "$MOONCAKE_TE_META_DATA_SERVER" \
        || true
}

delete_metadata_key() {
    local key="$1"
    local http_code=""
    http_code="$(
        curl -sS -o /dev/null -w '%{http_code}' \
            -X DELETE \
            --get \
            --data-urlencode "key=${key}" \
            "$MOONCAKE_TE_META_DATA_SERVER" \
            || true
    )"

    case "$http_code" in
        200|404)
            return 0
            ;;
        *)
            echo "Failed to delete Mooncake metadata key ${key} via ${MOONCAKE_TE_META_DATA_SERVER} (http ${http_code})." >&2
            return 1
            ;;
    esac
}

wait_for_metadata_absent() {
    local key="$1"
    local timeout_s="${2:-5}"
    local http_code=""

    for _ in $(seq "$timeout_s"); do
        http_code="$(metadata_key_http_code "$key")"
        case "$http_code" in
            404)
                return 0
                ;;
            200)
                sleep 1
                ;;
            *)
                echo "Unexpected response while checking Mooncake metadata key ${key}: http ${http_code}." >&2
                return 1
                ;;
        esac
    done

    echo "Timed out waiting for Mooncake metadata key ${key} to disappear." >&2
    return 1
}

owner_failed_with_duplicate_rpc_meta() {
    [[ -f "$OWNER_LOG_FILE" ]] && \
        grep -q "Duplicate rpc_meta key not allowed" "$OWNER_LOG_FILE"
}

reclaim_stale_owner_metadata() {
    local segment_name="${OWNER_HOST}:${OWNER_SEGMENT_PORT}"
    local rpc_meta_key="mooncake/rpc_meta/${segment_name}"
    local ram_key="mooncake/ram/${segment_name}"

    if local_owner_segment_process_exists; then
        echo "Refusing to reclaim Mooncake metadata for ${segment_name}: a local mooncake_client is still advertising that segment." >&2
        return 1
    fi

    echo "Reclaiming stale Mooncake metadata for ${segment_name}" >&2
    delete_metadata_key "$rpc_meta_key"
    delete_metadata_key "$ram_key"
    wait_for_metadata_absent "$rpc_meta_key"
    wait_for_metadata_absent "$ram_key"
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    stop_pid "${SERVER_PID:-}"
    stop_pid "${OWNER_PID:-}"
    if [[ -n "${OWNER_DISK_RUN_PATH:-}" ]]; then
        rm -rf -- "${OWNER_DISK_RUN_PATH}"
    fi
    exit "$exit_code"
}

prepare_owner_disk_path() {
    [[ -n "$OWNER_DISK_GIB" ]] || return 0
    [[ -n "$OWNER_DISK_PATH" ]] || return 0

    local base_path="${OWNER_DISK_PATH%/}"
    [[ -n "$base_path" ]] || base_path="$OWNER_DISK_PATH"
    [[ -n "$base_path" ]] || return 0

    mkdir -p "$base_path"
    OWNER_DISK_RUN_PATH="${base_path}/${HOST_TAG}.rpc${OWNER_RPC_PORT}.$$"
    rm -rf -- "$OWNER_DISK_RUN_PATH"
    mkdir -p "$OWNER_DISK_RUN_PATH"
}

print_owner_failure_hint() {
    [[ -f "$OWNER_LOG_FILE" ]] || return 0
    if grep -q "Duplicate rpc_meta key not allowed" "$OWNER_LOG_FILE"; then
        echo "Hint: stale Mooncake owner metadata is still registered. Stop the old owner or clean the stale metadata before retrying." >&2
    fi
}

wait_for_owner_ready() {
    for _ in $(seq "$OWNER_READY_TIMEOUT_S"); do
        if [[ -n "${OWNER_PID:-}" ]] && ! kill -0 "$OWNER_PID" 2>/dev/null; then
            echo "Mooncake owner exited before becoming ready." >&2
            tail -n 50 "$OWNER_LOG_FILE" >&2 || true
            print_owner_failure_hint
            return 1
        fi
        if is_port_listening "$OWNER_RPC_PORT"; then
            return 0
        fi
        sleep 1
    done

    echo "Timed out waiting for Mooncake owner RPC port ${OWNER_RPC_PORT}." >&2
    tail -n 50 "$OWNER_LOG_FILE" >&2 || true
    print_owner_failure_hint
    return 1
}

start_owner_once() {
    "${OWNER_CMD[@]}" >>"$OWNER_LOG_FILE" 2>&1 &
    OWNER_PID=$!
    if wait_for_owner_ready; then
        return 0
    fi
    wait "$OWNER_PID" 2>/dev/null || true
    OWNER_PID=""
    return 1
}

start_owner() {
    echo "Starting managed Mooncake owner; log: $OWNER_LOG_FILE"
    : > "$OWNER_LOG_FILE"
    if start_owner_once; then
        return 0
    fi

    if owner_failed_with_duplicate_rpc_meta; then
        if reclaim_stale_owner_metadata; then
            printf '\nRetrying managed Mooncake owner after stale-metadata cleanup.\n' >>"$OWNER_LOG_FILE"
            start_owner_once
            return $?
        fi
    fi

    return 1
}

default_mooncake_config_path() {
    printf '%s\n' "${SCRIPT_DIR}/mooncake_config.json"
}

detect_owner_host() {
    local host=""
    host="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [[ -n "${host//[[:space:]]/}" ]]; then
        printf '%s\n' "$host"
        return 0
    fi
    return 1
}

load_master_connection_env() {
    local candidate=""
    if [[ -n "${MOONCAKE_MASTER:-}" && -n "${MOONCAKE_TE_META_DATA_SERVER:-}" ]]; then
        return 0
    fi

    if [[ -z "${MASTER_ENV_FILE//[[:space:]]/}" ]]; then
        for candidate in \
            "${OWNER_LOG_DIR}/mooncake_master.env" \
            "$(dirname "${OWNER_LOG_DIR}")/mooncake_master.env"; do
            if [[ -r "$candidate" ]]; then
                MASTER_ENV_FILE="$candidate"
                break
            fi
        done
    fi

    if [[ ! -r "$MASTER_ENV_FILE" ]]; then
        echo "Error: Mooncake master env file not found: $MASTER_ENV_FILE" >&2
        echo "Hint: start the master via pre_serve with:" >&2
        echo "  bash scripts/mooncake/start_mooncake_master.sh --bg --env-file {log_dir}/mooncake_master.env" >&2
        return 1
    fi

    set -a
    # shellcheck disable=SC1090
    source "$MASTER_ENV_FILE"
    set +a

    if [[ -z "${MOONCAKE_MASTER:-}" || -z "${MOONCAKE_TE_META_DATA_SERVER:-}" ]]; then
        echo "Error: master env file is missing MOONCAKE_MASTER or MOONCAKE_TE_META_DATA_SERVER: $MASTER_ENV_FILE" >&2
        return 1
    fi
}

json_bool() {
    local raw="${1,,}"
    local name="$2"
    case "$raw" in
        1|true|yes|on) printf 'true\n' ;;
        0|false|no|off) printf 'false\n' ;;
        *)
            echo "Error: ${name} must be one of 1/0/true/false/yes/no/on/off, got: $1" >&2
            return 1
            ;;
    esac
}

detect_rnic_config() {
    local output=""
    local line=""
    local key=""
    local value=""
    local detected_gid_index=""
    local detected_owner_device=""
    local detected_gpu_map=""

    if ! output="$("${SCRIPT_DIR}/recommend_mooncake_rnic_config.sh" --machine-readable 2>&1)"; then
        echo "Error: failed to auto-detect Mooncake RNIC configuration." >&2
        echo "$output" >&2
        return 1
    fi

    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            MC_GID_INDEX) detected_gid_index="$value" ;;
            MC_OWNER_DEVICE) detected_owner_device="$value" ;;
            GPU_BDF_RNIC_MAP_JSON) detected_gpu_map="$value" ;;
        esac
    done <<< "$output"

    if [[ -z "${detected_gid_index//[[:space:]]/}" || \
          -z "${detected_owner_device//[[:space:]]/}" || \
          -z "${detected_gpu_map//[[:space:]]/}" ]]; then
        echo "Error: helper did not return MC_GID_INDEX, MC_OWNER_DEVICE, and GPU_BDF_RNIC_MAP_JSON." >&2
        echo "$output" >&2
        return 1
    fi

    export MC_GID_INDEX="$detected_gid_index"
    if [[ -z "${OWNER_DEVICE//[[:space:]]/}" ]]; then
        OWNER_DEVICE="$detected_owner_device"
    fi
    GPU_BDF_RNIC_MAP_JSON="$detected_gpu_map"
}

command_has_kv_transfer_config() {
    local arg=""
    for arg in "$@"; do
        if [[ "$arg" == "--kv-transfer-config" || "$arg" == --kv-transfer-config=* ]]; then
            return 0
        fi
    done
    return 1
}

build_kv_transfer_config_json() {
    local load_async_json=""
    local cross_layers_json=""
    local extra_config_json=""

    load_async_json="$(json_bool "$MOONCAKE_LOAD_ASYNC" "MC_MOONCAKE_LOAD_ASYNC")"
    cross_layers_json="$(json_bool "$MOONCAKE_ENABLE_CROSS_LAYERS_BLOCKS" "MC_MOONCAKE_ENABLE_CROSS_LAYERS_BLOCKS")"

    extra_config_json=$(printf '{"load_async":%s,"enable_cross_layers_blocks":%s' \
        "$load_async_json" \
        "$cross_layers_json")
    if [[ -n "${GPU_BDF_RNIC_MAP_JSON:-}" ]]; then
        extra_config_json+=$(printf ',"gpu_bdf_rnic_map":%s' "$GPU_BDF_RNIC_MAP_JSON")
    fi
    extra_config_json+='}'

    printf '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":%s}\n' \
        "$extra_config_json"
}

if [[ -z "$OWNER_CPU_MEM_GIB" ]]; then
    echo "Error: MC_OWNER_CPU_MEM_GIB must be set." >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 1
fi

mkdir -p "$OWNER_LOG_DIR"

if [[ -z "${MOONCAKE_CONFIG_PATH:-}" ]]; then
    MOONCAKE_CONFIG_PATH="$(default_mooncake_config_path)"
    export MOONCAKE_CONFIG_PATH
fi
if [[ ! -r "$MOONCAKE_CONFIG_PATH" ]]; then
    echo "Error: MOONCAKE_CONFIG_PATH does not exist: $MOONCAKE_CONFIG_PATH" >&2
    exit 1
fi

load_master_connection_env

if [[ "$MOONCAKE_PROTOCOL" == "rdma" || "$MOONCAKE_PROTOCOL" == "efa" ]]; then
    export MC_ENABLE_DEST_DEVICE_AFFINITY="${MC_ENABLE_DEST_DEVICE_AFFINITY:-1}"
    detect_rnic_config
    if ! validate_rdma_device_list "$OWNER_DEVICE"; then
        exit 1
    fi
fi

if command_has_kv_transfer_config "$@"; then
    echo "Error: run_vllm_with_mooncake_owner.sh injects Mooncake --kv-transfer-config automatically. Remove the explicit --kv-transfer-config from the wrapped command." >&2
    exit 1
fi

prepare_owner_disk_path

if [[ -z "${OWNER_HOST//[[:space:]]/}" ]]; then
    OWNER_HOST="$(detect_owner_host || true)"
fi
if [[ -z "${OWNER_HOST//[[:space:]]/}" ]]; then
    echo "Error: failed to determine owner host automatically; set MC_OWNER_HOST." >&2
    exit 1
fi

OWNER_CMD=(
    bash "${SCRIPT_DIR}/start_mooncake_owner.sh"
    --cpu-mem-size "$OWNER_CPU_MEM_GIB"
    --rpc-port "$OWNER_RPC_PORT"
    --segment-port "$OWNER_SEGMENT_PORT"
)

if [[ -n "$OWNER_DISK_GIB" ]]; then
    OWNER_CMD+=(--disk-size "$OWNER_DISK_GIB")
fi
if [[ -n "$OWNER_DISK_RUN_PATH" ]]; then
    OWNER_CMD+=(--disk-path "$OWNER_DISK_RUN_PATH")
elif [[ -n "$OWNER_DISK_PATH" ]]; then
    OWNER_CMD+=(--disk-path "$OWNER_DISK_PATH")
fi
if [[ -n "$OWNER_DEVICE" ]]; then
    export MC_OWNER_DEVICE="$OWNER_DEVICE"
    OWNER_CMD+=(--device "$OWNER_DEVICE")
fi
if [[ -n "$OWNER_HOST" ]]; then
    OWNER_CMD+=(--host "$OWNER_HOST")
fi

KV_TRANSFER_CONFIG_JSON="$(build_kv_transfer_config_json)"

echo "Mooncake master: $MOONCAKE_MASTER"
echo "Mooncake metadata server: $MOONCAKE_TE_META_DATA_SERVER"
if [[ -n "${MC_GID_INDEX:-}" ]]; then
    echo "Detected GID index: $MC_GID_INDEX"
fi
if [[ -n "${OWNER_DEVICE:-}" ]]; then
    echo "Detected owner RNICs: $OWNER_DEVICE"
fi
if [[ -n "${GPU_BDF_RNIC_MAP_JSON:-}" ]]; then
    echo "Detected GPU/RNIC map: $GPU_BDF_RNIC_MAP_JSON"
fi

SERVER_CMD=("$@" "--kv-transfer-config" "$KV_TRANSFER_CONFIG_JSON")

trap cleanup EXIT INT TERM

start_owner

echo "Starting managed server: ${SERVER_CMD[*]}"
"${SERVER_CMD[@]}" &
SERVER_PID=$!

wait "$SERVER_PID"
