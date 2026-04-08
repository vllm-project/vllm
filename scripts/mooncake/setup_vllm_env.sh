#!/usr/bin/env bash
# Set up requester-side environment variables for a vLLM worker with Mooncake KV offloading.
#
# Usage:
#   source setup_vllm_env.sh --cpu-mem-size 80
#   source setup_vllm_env.sh --cpu-mem-size 80 --disk-size 400
#
# Options:
#   --cpu-mem-size <GB>  Requester local-buffer hint in GB (required)
#   --disk-size    <GB>  Accepted for compatibility; owner disk offload is started separately
#
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Parse arguments --------------------------------------------------------
CPU_MEM_GB=""
DISK_GB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-mem-size)
            CPU_MEM_GB="$2"; shift 2 ;;
        --disk-size)
            DISK_GB="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: source $0 --cpu-mem-size <GB> [--disk-size <GB>]" >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

if [[ -z "$CPU_MEM_GB" ]]; then
    echo "Error: --cpu-mem-size is required" >&2
    echo "Usage: source $0 --cpu-mem-size <GB> [--disk-size <GB>]" >&2
    return 1 2>/dev/null || exit 1
fi

# --- Mooncake connection pool -----------------------------------------------
export MC_TCP_ENABLE_CONNECTION_POOL=1
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-${SCRIPTS_DIR}/mooncake_config.json}"
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES="$(( CPU_MEM_GB * 1073741824 ))"

echo "Mooncake env configured:"
echo "  Config:   $MOONCAKE_CONFIG_PATH"
echo "  Local buffer hint: ${CPU_MEM_GB} GB"
echo "  Owner:    external Mooncake service must be started separately"
if [[ -n "$DISK_GB" ]]; then
    echo "  Disk:     requester-only setup ignores --disk-size; start the owner with disk offload separately"
fi
