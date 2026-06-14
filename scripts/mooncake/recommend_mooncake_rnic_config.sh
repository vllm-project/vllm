#!/usr/bin/env bash
# Print the local Mooncake RDMA defaults used by the managed launcher and
# worker auto-detection.
#
# The script does not modify any config. It exits non-zero if the topology is
# ambiguous or incomplete, but still prints the best-effort recommendation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rdma_config_utils.sh"

OUTPUT_MODE="human"
SELECTED_GID_INDEX="${MC_GID_INDEX:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine-readable)
            OUTPUT_MODE="machine"
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Usage: recommend_mooncake_rnic_config.sh [--machine-readable]

Default output:
  Prints the per-GPU worker RNIC recommendation, MOONCAKE_DEVICE, and
  MC_OWNER_DEVICE.

--machine-readable:
  Prints exactly:
    MC_GID_INDEX=<index>
    MOONCAKE_DEVICE=<csv>
    MC_OWNER_DEVICE=<csv>
EOF
            exit 0
            ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${SELECTED_GID_INDEX//[[:space:]]/}" ]]; then
    SELECTED_GID_INDEX="$(detect_preferred_gid_index)" || {
        echo "Error: failed to auto-detect a usable RDMA GID index." >&2
        exit 1
    }
fi

if PLAN_OUTPUT="$(plan_worker_rdma_assignments "$SELECTED_GID_INDEX")"; then
    STATUS=0
else
    STATUS=$?
fi
WORKER_DEVICE_CSV="$(get_worker_rdma_devices_csv "$SELECTED_GID_INDEX" 2>/dev/null || true)"
OWNER_DEVICE_CSV="$(get_active_rdma_device_names_csv "$SELECTED_GID_INDEX" 2>/dev/null || true)"

if [[ -z "${OWNER_DEVICE_CSV//[[:space:]]/}" ]]; then
    echo "Error: no active RDMA devices matched MC_GID_INDEX=${SELECTED_GID_INDEX}." >&2
    exit 1
fi

WORKER_PLAN=()
if [[ -n "${PLAN_OUTPUT//[[:space:]]/}" ]]; then
    mapfile -t WORKER_PLAN <<< "$PLAN_OUTPUT"
fi

if [[ "$OUTPUT_MODE" == "machine" ]]; then
    printf 'MC_GID_INDEX=%s\n' "$SELECTED_GID_INDEX"
    printf 'MOONCAKE_DEVICE=%s\n' "$WORKER_DEVICE_CSV"
    printf 'MC_OWNER_DEVICE=%s\n' "$OWNER_DEVICE_CSV"
    if [[ "$STATUS" -ne 0 ]]; then
        echo "Unsafe best-effort recommendation: at least one GPU->RNIC choice was ambiguous or missing." >&2
        echo "Review the human-readable diagnostics from recommend_mooncake_rnic_config.sh." >&2
    fi
    exit "$STATUS"
fi

if [[ "$STATUS" -ne 0 ]]; then
    echo "Unsafe best-effort recommendation: at least one GPU->RNIC choice was ambiguous or missing." >&2
    echo "Review the diagnostics below and set MOONCAKE_DEVICE explicitly if needed." >&2
fi

echo "# Recommendations filtered by MC_GID_INDEX=${SELECTED_GID_INDEX}"
echo "# Local GPU to RNIC diagnostics"
for plan_entry in "${WORKER_PLAN[@]}"; do
    IFS='|' read -r gpu_index gpu_bus gpu_numa device_name reason <<< "$plan_entry"
    unset IFS
    printf '# GPU %s (%s, NUMA %s) -> %s [%s]\n' \
        "$gpu_index" \
        "$gpu_bus" \
        "$gpu_numa" \
        "${device_name:-<missing>}" \
        "${reason:-missing}"
done
echo

echo "# worker env"
printf 'MOONCAKE_DEVICE=%s\n' "$WORKER_DEVICE_CSV"
echo

echo "# owner env"
printf 'MC_OWNER_DEVICE=%s\n' "$OWNER_DEVICE_CSV"
echo
echo "# note"
echo "# MOONCAKE_DEVICE is a comma-separated per-GPU worker list. If unset, requesters fall back to Mooncake auto-selection."

exit "$STATUS"
