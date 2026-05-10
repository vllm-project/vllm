#!/usr/bin/env bash
# Set up environment variables for a vLLM worker with Mooncake KV offloading.
#
# Usage:
#   source setup_vllm_env.sh --cpu-mem-size 80                       # 80 GB CPU offload, no disk
#   source setup_vllm_env.sh --cpu-mem-size 80 --disk-size 400      # 80 GB CPU + 400 GB disk offload
#   source setup_vllm_env.sh --cpu-mem-size 80 --disk-size 400 --disk-path /nvme/offload
#
# Options:
#   --cpu-mem-size <GB>  CPU memory size in GB for Mooncake global segment (required)
#   --disk-size    <GB>  Enable disk offloading with the given quota in GB (optional)
#   --disk-path    <dir> SSD directory for disk offloading (default: /mnt/data/mooncake_offload)
#
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Parse arguments --------------------------------------------------------
CPU_MEM_GB=""
DISK_GB=""
DISK_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-mem-size)
            CPU_MEM_GB="$2"; shift 2 ;;
        --disk-size)
            DISK_GB="$2"; shift 2 ;;
        --disk-path)
            DISK_PATH="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: source $0 --cpu-mem-size <GB> [--disk-size <GB>] [--disk-path <dir>]" >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

if [[ -z "$CPU_MEM_GB" ]]; then
    echo "Error: --cpu-mem-size is required" >&2
    echo "Usage: source $0 --cpu-mem-size <GB> [--disk-size <GB>] [--disk-path <dir>]" >&2
    return 1 2>/dev/null || exit 1
fi

# --- Mooncake connection pool -----------------------------------------------
export MC_TCP_ENABLE_CONNECTION_POOL=1

# --- Update mooncake config with CPU memory size ----------------------------
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-${SCRIPTS_DIR}/mooncake_config.json}"

python3 -c "
import json, sys

cfg_path, global_size_gb = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['global_segment_size'] = global_size_gb + 'GB'
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
    f.write('\n')
" "$MOONCAKE_CONFIG_PATH" "$CPU_MEM_GB"

echo "Mooncake env configured:"
echo "  Config:   $MOONCAKE_CONFIG_PATH"
echo "  CPU mem:  ${CPU_MEM_GB} GB (global segment)"

# --- Disk offloading (client-side env vars) ---------------------------------
if [[ -n "$DISK_GB" ]]; then
    DISK_PATH="${DISK_PATH:-/mnt/data/mooncake_offload}"
    DISK_BYTES=$(( DISK_GB * 1073741824 ))
    if [[ "$DISK_GB" -lt 10 ]]; then
        echo "Error: Disk size must be at least 10 GB" >&2
        return 1 2>/dev/null || exit 1
    fi

    export MOONCAKE_ENABLE_OFFLOAD=1
    export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$DISK_PATH"
    export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="${MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR:-bucket_storage_backend}"
    export MOONCAKE_BUCKET_EVICTION_POLICY="${MOONCAKE_BUCKET_EVICTION_POLICY:-lru}"
    export MOONCAKE_USE_URING="${MOONCAKE_USE_URING:-true}"
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES="${MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES:-1073741824}"  # 1 GB staging buffer

    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="$DISK_BYTES"
    export MOONCAKE_BUCKET_MAX_TOTAL_SIZE=$(( DISK_BYTES - 42949672960 ))  # 40 GB headroom

    export MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS="${MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS:-3}"

    mkdir -p "$MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"

    echo "  Disk:     ${DISK_GB} GB (path=${DISK_PATH}, eviction=${MOONCAKE_BUCKET_EVICTION_POLICY})"
else
    echo "  Disk:     OFF (pass --disk-size <GB> to enable)"
fi
