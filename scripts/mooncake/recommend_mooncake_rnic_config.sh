#!/usr/bin/env bash
# Print a ready-to-paste GPU-BDF -> RNIC map for Mooncake workers and the
# corresponding owner RNIC union for the local node.
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
  Prints ready-to-paste kv_connector_extra_config JSON and MC_OWNER_DEVICE.

--machine-readable:
  Prints exactly:
    MC_GID_INDEX=<index>
    MC_OWNER_DEVICE=<csv>
    GPU_BDF_RNIC_MAP_JSON=<compact-json>
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

mapfile -t GPU_ENTRIES < <(get_local_gpu_entries)
mapfile -t RDMA_ENTRIES < <(list_active_rdma_devices "$SELECTED_GID_INDEX")

if [[ "${#GPU_ENTRIES[@]}" -eq 0 ]]; then
    echo "Error: no local GPUs were discovered with nvidia-smi." >&2
    exit 1
fi

if [[ "${#RDMA_ENTRIES[@]}" -eq 0 ]]; then
    echo "Error: no active RDMA devices matched MC_GID_INDEX=${SELECTED_GID_INDEX}." >&2
    exit 1
fi

declare -A RDMA_BUS_BY_NAME=()
declare -A RDMA_NUMA_BY_NAME=()
declare -A RDMA_NETDEVS_BY_NAME=()
declare -A MAP_BY_GPU_BDF=()
declare -A OWNER_DEVICE_SEEN=()
declare -A GPU_REASON_BY_BDF=()
OWNER_DEVICES=()
STATUS=0

for rdma_entry in "${RDMA_ENTRIES[@]}"; do
    IFS='|' read -r rdma_name rdma_bus rdma_numa rdma_netdevs <<< "$rdma_entry"
    unset IFS
    RDMA_BUS_BY_NAME["$rdma_name"]="$rdma_bus"
    RDMA_NUMA_BY_NAME["$rdma_name"]="$rdma_numa"
    RDMA_NETDEVS_BY_NAME["$rdma_name"]="$rdma_netdevs"
done

choose_device_for_gpu() {
    local gpu_index=$1
    local gpu_bus=$2
    local gpu_numa=$3
    local rdma_name=""
    local best_name=""
    local best_distance=1073741824
    local best_count=0
    local best_same_numa_name=""
    local best_same_numa_distance=1073741824
    local best_same_numa_count=0
    local -a exact_matches=()
    local distance=0

    for rdma_name in "${!RDMA_BUS_BY_NAME[@]}"; do
        if gpu_matches_rnic_netdev "$gpu_index" "${RDMA_NETDEVS_BY_NAME[$rdma_name]}"; then
            exact_matches+=("$rdma_name")
        fi
    done

    if [[ "${#exact_matches[@]}" -eq 1 ]]; then
        printf '%s|exact-netdev|0\n' "${exact_matches[0]}"
        return 0
    fi
    if [[ "${#exact_matches[@]}" -gt 1 ]]; then
        IFS=$'\n' exact_matches=($(printf '%s\n' "${exact_matches[@]}" | sort))
        unset IFS
        printf '%s|ambiguous-exact-netdev|1\n' "${exact_matches[0]}"
        return 0
    fi

    for rdma_name in "${!RDMA_BUS_BY_NAME[@]}"; do
        distance="$(pci_bus_distance "$gpu_bus" "${RDMA_BUS_BY_NAME[$rdma_name]}")"
        if [[ "$distance" -lt "$best_distance" ]]; then
            best_name="$rdma_name"
            best_distance="$distance"
            best_count=1
        elif [[ "$distance" -eq "$best_distance" ]]; then
            if [[ "$rdma_name" < "$best_name" ]]; then
                best_name="$rdma_name"
            fi
            best_count=$((best_count + 1))
        fi

        if [[ "$gpu_numa" != "-1" && "$gpu_numa" == "${RDMA_NUMA_BY_NAME[$rdma_name]}" ]]; then
            if [[ "$distance" -lt "$best_same_numa_distance" ]]; then
                best_same_numa_name="$rdma_name"
                best_same_numa_distance="$distance"
                best_same_numa_count=1
            elif [[ "$distance" -eq "$best_same_numa_distance" ]]; then
                if [[ "$rdma_name" < "$best_same_numa_name" ]]; then
                    best_same_numa_name="$rdma_name"
                fi
                best_same_numa_count=$((best_same_numa_count + 1))
            fi
        fi
    done

    if [[ -n "$best_same_numa_name" ]]; then
        printf '%s|same-numa-pci-distance|%s\n' \
            "$best_same_numa_name" \
            "$(( best_same_numa_count > 1 ? 1 : 0 ))"
        return 0
    fi

    if [[ -n "$best_name" ]]; then
        printf '%s|global-pci-distance|%s\n' \
            "$best_name" \
            "$(( best_count > 1 ? 1 : 0 ))"
        return 0
    fi

    return 1
}

for gpu_entry in "${GPU_ENTRIES[@]}"; do
    IFS='|' read -r gpu_index gpu_bus gpu_numa <<< "$gpu_entry"
    unset IFS

    choice="$(choose_device_for_gpu "$gpu_index" "$gpu_bus" "$gpu_numa" || true)"
    if [[ -z "$choice" ]]; then
        STATUS=1
        GPU_REASON_BY_BDF["$gpu_bus"]="no-active-rnic-match"
        continue
    fi

    IFS='|' read -r chosen_device reason ambiguous <<< "$choice"
    unset IFS
    MAP_BY_GPU_BDF["$gpu_bus"]="$chosen_device"
    GPU_REASON_BY_BDF["$gpu_bus"]="$reason"
    if [[ "$ambiguous" == "1" ]]; then
        STATUS=1
    fi
    if [[ -z "${OWNER_DEVICE_SEEN[$chosen_device]+x}" ]]; then
        OWNER_DEVICE_SEEN["$chosen_device"]=1
        OWNER_DEVICES+=("$chosen_device")
    fi
done

if [[ "${#MAP_BY_GPU_BDF[@]}" -ne "${#GPU_ENTRIES[@]}" ]]; then
    STATUS=1
fi

owner_device_csv="$(IFS=,; printf '%s' "${OWNER_DEVICES[*]}")"

emit_gpu_bdf_rnic_map_json() {
    local pretty=${1:-0}
    local first=1
    if [[ "$pretty" == "1" ]]; then
        echo '{'
        echo '  "gpu_bdf_rnic_map": {'
    else
        printf '{'
    fi
    for gpu_entry in "${GPU_ENTRIES[@]}"; do
        IFS='|' read -r _gpu_index gpu_bus _gpu_numa <<< "$gpu_entry"
        unset IFS
        [[ -n "${MAP_BY_GPU_BDF[$gpu_bus]:-}" ]] || continue
        if [[ "$first" -eq 0 ]]; then
            if [[ "$pretty" == "1" ]]; then
                echo ','
            else
                printf ','
            fi
        fi
        if [[ "$pretty" == "1" ]]; then
            printf '    "%s": "%s"' "$gpu_bus" "${MAP_BY_GPU_BDF[$gpu_bus]}"
        else
            printf '"%s":"%s"' "$gpu_bus" "${MAP_BY_GPU_BDF[$gpu_bus]}"
        fi
        first=0
    done
    if [[ "$pretty" == "1" ]]; then
        echo
        echo '  }'
        echo '}'
    else
        printf '}'
    fi
}

if [[ "$OUTPUT_MODE" == "machine" ]]; then
    printf 'MC_GID_INDEX=%s\n' "$SELECTED_GID_INDEX"
    printf 'MC_OWNER_DEVICE=%s\n' "$owner_device_csv"
    printf 'GPU_BDF_RNIC_MAP_JSON='
    emit_gpu_bdf_rnic_map_json 0
    printf '\n'
    if [[ "$STATUS" -ne 0 ]]; then
        echo "Unsafe best-effort recommendation: at least one GPU->RNIC choice was ambiguous or missing." >&2
        echo "Review the human-readable diagnostics from recommend_mooncake_rnic_config.sh." >&2
    fi
    exit "$STATUS"
fi

if [[ "$STATUS" -ne 0 ]]; then
    echo "Unsafe best-effort recommendation: at least one GPU->RNIC choice was ambiguous or missing." >&2
    echo "Review the diagnostics below and set the mapping explicitly." >&2
fi

echo "# Recommendations filtered by MC_GID_INDEX=${SELECTED_GID_INDEX}"
echo "# Local GPU to RNIC diagnostics"
for gpu_entry in "${GPU_ENTRIES[@]}"; do
    IFS='|' read -r gpu_index gpu_bus gpu_numa <<< "$gpu_entry"
    unset IFS
    printf '# GPU %s (%s, NUMA %s) -> %s [%s]\n' \
        "$gpu_index" \
        "$gpu_bus" \
        "$gpu_numa" \
        "${MAP_BY_GPU_BDF[$gpu_bus]:-<missing>}" \
        "${GPU_REASON_BY_BDF[$gpu_bus]:-missing}"
done
echo

echo "# kv_connector_extra_config JSON"
emit_gpu_bdf_rnic_map_json 1
echo

echo "# owner env"
printf 'MC_OWNER_DEVICE=%s\n' "$owner_device_csv"

exit "$STATUS"
