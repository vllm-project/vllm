#!/usr/bin/env bash

normalize_pci_bus_id() {
    local raw="${1,,}"
    local domain=""
    local bus=""
    local device=""
    local function=""

    if [[ "$raw" =~ ^(([0-9a-f]{4,8}):)?([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])$ ]]; then
        domain="${BASH_REMATCH[2]:-0}"
        bus="${BASH_REMATCH[3]}"
        device="${BASH_REMATCH[4]}"
        function="${BASH_REMATCH[5]}"
        printf '%08x:%02x:%02x.%x\n' \
            "$((16#$domain))" \
            "$((16#$bus))" \
            "$((16#$device))" \
            "$((16#$function))"
        return 0
    fi
    return 1
}

port_is_active() {
    local device_name=$1
    local port=$2
    local state_file="/sys/class/infiniband/${device_name}/ports/${port}/state"
    [[ -r "$state_file" ]] || return 1
    grep -q "ACTIVE" "$state_file"
}

device_matches_gid_index() {
    local device_name=$1
    local port=$2
    local gid_index=$3
    local ndev_file="/sys/class/infiniband/${device_name}/ports/${port}/gid_attrs/ndevs/${gid_index}"
    local gid_file="/sys/class/infiniband/${device_name}/ports/${port}/gids/${gid_index}"
    local ndev=""
    local gid=""
    local compact_gid=""

    [[ -r "$ndev_file" && -r "$gid_file" ]] || return 1
    ndev="$(cat "$ndev_file" 2>/dev/null)" || return 1
    [[ -n "${ndev//[[:space:]]/}" ]] || return 1

    gid="$(cat "$gid_file" 2>/dev/null)" || return 1
    compact_gid="${gid//[:[:space:]]/}"
    [[ -n "$compact_gid" ]] || return 1
    [[ "$compact_gid" != "00000000000000000000000000000000" ]]
}

gid_entry_is_non_link_local() {
    local device_name=$1
    local port=$2
    local gid_index=$3
    local gid_file="/sys/class/infiniband/${device_name}/ports/${port}/gids/${gid_index}"
    local gid=""
    local compact_gid=""

    [[ -r "$gid_file" ]] || return 1
    gid="$(cat "$gid_file" 2>/dev/null)" || return 1
    compact_gid="${gid//[:[:space:]]/}"
    [[ -n "$compact_gid" ]] || return 1
    [[ "$compact_gid" != "00000000000000000000000000000000" ]] || return 1
    [[ "$compact_gid" != fe80* ]]
}

get_gid_entry_type() {
    local device_name=$1
    local port=$2
    local gid_index=$3
    local type_file="/sys/class/infiniband/${device_name}/ports/${port}/gid_attrs/types/${gid_index}"

    [[ -r "$type_file" ]] || return 1
    cat "$type_file" 2>/dev/null
}

gid_entry_is_roce_v2() {
    local gid_type=""
    gid_type="$(get_gid_entry_type "$1" "$2" "$3" 2>/dev/null || true)"
    [[ "$gid_type" == "RoCE v2" ]]
}

device_is_active_for_gid_index() {
    local device_name=$1
    local gid_index=${2:-}
    local port_path=""
    local port=""

    for port_path in "/sys/class/infiniband/${device_name}"/ports/*; do
        [[ -d "$port_path" ]] || continue
        port="$(basename "$port_path")"
        if ! port_is_active "$device_name" "$port"; then
            continue
        fi
        if [[ -z "$gid_index" ]]; then
            return 0
        fi
        if device_matches_gid_index "$device_name" "$port" "$gid_index"; then
            return 0
        fi
    done
    return 1
}

get_rdma_device_bus_id() {
    local device_name=$1
    normalize_pci_bus_id "$(basename "$(readlink -f "/sys/class/infiniband/${device_name}/device")")"
}

get_pci_numa_node() {
    local pci_bus_id=$1
    local normalized=""
    local numa_path=""

    normalized="$(normalize_pci_bus_id "$pci_bus_id" 2>/dev/null || true)"
    [[ -n "$normalized" ]] || {
        printf '%s\n' "-1"
        return 0
    }

    for numa_path in \
        "/sys/bus/pci/devices/${normalized}/numa_node" \
        "/sys/bus/pci/devices/0000:${normalized#*:}/numa_node"; do
        if [[ -r "$numa_path" ]]; then
            cat "$numa_path"
            return 0
        fi
    done

    printf '%s\n' "-1"
}

get_rdma_device_numa_node() {
    local device_name=$1
    get_pci_numa_node "$(get_rdma_device_bus_id "$device_name")"
}

get_rdma_device_netdevs_csv() {
    local device_name=$1
    local net_dir="/sys/class/infiniband/${device_name}/device/net"
    local netdev_path=""
    local netdevs=()

    [[ -d "$net_dir" ]] || return 0
    for netdev_path in "$net_dir"/*; do
        [[ -e "$netdev_path" ]] || continue
        netdevs+=("$(basename "$netdev_path")")
    done
    if [[ "${#netdevs[@]}" -gt 0 ]]; then
        IFS=,
        printf '%s\n' "${netdevs[*]}"
        unset IFS
    fi
}

list_active_rdma_devices() {
    local gid_index=${1:-${MC_GID_INDEX:-}}
    local device_path=""
    local device_name=""
    local bus_id=""
    local numa_node=""
    local netdevs=""

    for device_path in /sys/class/infiniband/*; do
        [[ -e "$device_path" ]] || continue
        device_name="$(basename "$device_path")"
        if ! device_is_active_for_gid_index "$device_name" "$gid_index"; then
            continue
        fi
        bus_id="$(get_rdma_device_bus_id "$device_name" 2>/dev/null || true)"
        [[ -n "$bus_id" ]] || continue
        numa_node="$(get_rdma_device_numa_node "$device_name")"
        netdevs="$(get_rdma_device_netdevs_csv "$device_name")"
        printf '%s|%s|%s|%s\n' \
            "$device_name" \
            "$bus_id" \
            "$numa_node" \
            "$netdevs"
    done | sort
}

detect_preferred_gid_index() {
    local max_index=${1:-8}
    local device_path=""
    local device_name=""
    local port_path=""
    local port=""
    local index=""
    local best_index=""
    local best_non_link_local_v2_count=-1
    local best_non_link_local_count=-1
    local best_total_v2_count=-1
    local best_total_count=-1
    local non_link_local_v2_count=0
    local non_link_local_count=0
    local total_v2_count=0
    local total_count=0

    for index in $(seq 0 "$max_index"); do
        non_link_local_v2_count=0
        non_link_local_count=0
        total_v2_count=0
        total_count=0
        for device_path in /sys/class/infiniband/*; do
            [[ -e "$device_path" ]] || continue
            device_name="$(basename "$device_path")"
            for port_path in "$device_path"/ports/*; do
                [[ -d "$port_path" ]] || continue
                port="$(basename "$port_path")"
                if ! port_is_active "$device_name" "$port"; then
                    continue
                fi
                if ! device_matches_gid_index "$device_name" "$port" "$index"; then
                    continue
                fi
                total_count=$((total_count + 1))
                if gid_entry_is_roce_v2 "$device_name" "$port" "$index"; then
                    total_v2_count=$((total_v2_count + 1))
                fi
                if gid_entry_is_non_link_local "$device_name" "$port" "$index"; then
                    non_link_local_count=$((non_link_local_count + 1))
                    if gid_entry_is_roce_v2 "$device_name" "$port" "$index"; then
                        non_link_local_v2_count=$((non_link_local_v2_count + 1))
                    fi
                fi
            done
        done

        if (( non_link_local_v2_count > best_non_link_local_v2_count )); then
            best_index="$index"
            best_non_link_local_v2_count=$non_link_local_v2_count
            best_non_link_local_count=$non_link_local_count
            best_total_v2_count=$total_v2_count
            best_total_count=$total_count
        elif (( non_link_local_v2_count == best_non_link_local_v2_count )) && \
             (( non_link_local_count > best_non_link_local_count )); then
            best_index="$index"
            best_non_link_local_count=$non_link_local_count
            best_total_v2_count=$total_v2_count
            best_total_count=$total_count
        elif (( non_link_local_v2_count == best_non_link_local_v2_count )) && \
             (( non_link_local_count == best_non_link_local_count )) && \
             (( total_v2_count > best_total_v2_count )); then
            best_index="$index"
            best_total_v2_count=$total_v2_count
            best_total_count=$total_count
        elif (( non_link_local_v2_count == best_non_link_local_v2_count )) && \
             (( non_link_local_count == best_non_link_local_count )) && \
             (( total_v2_count == best_total_v2_count )) && \
             (( total_count > best_total_count )); then
            best_index="$index"
            best_total_count=$total_count
        fi
    done

    if [[ -n "$best_index" ]] && (( best_total_count > 0 )); then
        printf '%s\n' "$best_index"
        return 0
    fi
    return 1
}

validate_rdma_device_list() {
    local device_csv=$1
    local gid_index=${2:-${MC_GID_INDEX:-}}
    local entry=""
    local device_name=""

    [[ -n "${device_csv//[[:space:]]/}" ]] || {
        echo "RDMA device list is empty." >&2
        return 1
    }

    IFS=',' read -r -a _rdma_device_entries <<< "$device_csv"
    unset IFS
    for entry in "${_rdma_device_entries[@]}"; do
        device_name="${entry//[[:space:]]/}"
        [[ -n "$device_name" ]] || {
            echo "RDMA device list contains an empty entry." >&2
            return 1
        }
        [[ -d "/sys/class/infiniband/${device_name}" ]] || {
            echo "RDMA device ${device_name} does not exist under /sys/class/infiniband." >&2
            return 1
        }
        if ! device_is_active_for_gid_index "$device_name" "$gid_index"; then
            if [[ -n "$gid_index" ]]; then
                echo "RDMA device ${device_name} is not active for MC_GID_INDEX=${gid_index}." >&2
            else
                echo "RDMA device ${device_name} is not active." >&2
            fi
            return 1
        fi
    done
}

get_local_gpu_entries() {
    local query_output=""
    local line=""
    local gpu_index=""
    local bus_id=""
    local normalized_bus=""
    local numa_node=""

    command -v nvidia-smi >/dev/null 2>&1 || return 1
    query_output="$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null)" \
        || return 1

    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        gpu_index="${line%%,*}"
        bus_id="${line#*,}"
        gpu_index="${gpu_index//[[:space:]]/}"
        bus_id="${bus_id//[[:space:]]/}"
        normalized_bus="$(normalize_pci_bus_id "$bus_id" 2>/dev/null || true)"
        [[ -n "$normalized_bus" ]] || continue
        numa_node="$(get_pci_numa_node "$normalized_bus")"
        printf '%s|%s|%s\n' "$gpu_index" "$normalized_bus" "$numa_node"
    done <<< "$query_output"
}

gpu_matches_rnic_netdev() {
    local gpu_index=$1
    local netdev_csv=$2
    local netdev=""

    IFS=',' read -r -a _rdma_netdev_entries <<< "$netdev_csv"
    unset IFS
    for netdev in "${_rdma_netdev_entries[@]}"; do
        if [[ "$netdev" =~ ^gpu${gpu_index}rdma[0-9]+$ ]]; then
            return 0
        fi
    done
    return 1
}

pci_bus_distance() {
    local lhs
    local rhs
    local lhs_domain=""
    local lhs_bus=""
    local lhs_device=""
    local lhs_function=""
    local rhs_domain=""
    local rhs_bus=""
    local rhs_device=""
    local rhs_function=""
    local lhs_domain_dec=0
    local lhs_bus_dec=0
    local lhs_device_dec=0
    local lhs_function_dec=0
    local rhs_domain_dec=0
    local rhs_bus_dec=0
    local rhs_device_dec=0
    local rhs_function_dec=0
    local distance=0

    lhs="$(normalize_pci_bus_id "$1" 2>/dev/null || true)"
    rhs="$(normalize_pci_bus_id "$2" 2>/dev/null || true)"
    [[ -n "$lhs" && -n "$rhs" ]] || {
        printf '%s\n' "1073741824"
        return 0
    }

    lhs_domain="${lhs%%:*}"
    lhs_bus="${lhs#*:}"
    lhs_bus="${lhs_bus%%:*}"
    lhs_device="${lhs##*:}"
    lhs_function="${lhs_device#*.}"
    lhs_device="${lhs_device%%.*}"

    rhs_domain="${rhs%%:*}"
    rhs_bus="${rhs#*:}"
    rhs_bus="${rhs_bus%%:*}"
    rhs_device="${rhs##*:}"
    rhs_function="${rhs_device#*.}"
    rhs_device="${rhs_device%%.*}"

    lhs_domain_dec=$((16#$lhs_domain))
    lhs_bus_dec=$((16#$lhs_bus))
    lhs_device_dec=$((16#$lhs_device))
    lhs_function_dec=$((16#$lhs_function))
    rhs_domain_dec=$((16#$rhs_domain))
    rhs_bus_dec=$((16#$rhs_bus))
    rhs_device_dec=$((16#$rhs_device))
    rhs_function_dec=$((16#$rhs_function))

    distance=$(( \
        (lhs_domain_dec > rhs_domain_dec ? lhs_domain_dec - rhs_domain_dec : rhs_domain_dec - lhs_domain_dec) * 1000000 \
        + (lhs_bus_dec > rhs_bus_dec ? lhs_bus_dec - rhs_bus_dec : rhs_bus_dec - lhs_bus_dec) * 1000 \
        + (lhs_device_dec > rhs_device_dec ? lhs_device_dec - rhs_device_dec : rhs_device_dec - lhs_device_dec) * 10 \
        + (lhs_function_dec > rhs_function_dec ? lhs_function_dec - rhs_function_dec : rhs_function_dec - lhs_function_dec) \
    ))
    printf '%s\n' "$distance"
}

get_active_rdma_device_names_csv() {
    local gid_index=${1:-${MC_GID_INDEX:-}}
    local rdma_output=""
    local device_name=""
    local -a devices=()

    rdma_output="$(list_active_rdma_devices "$gid_index")" || return 1
    [[ -n "${rdma_output//[[:space:]]/}" ]] || return 1

    while IFS='|' read -r device_name _rest; do
        [[ -n "$device_name" ]] || continue
        devices+=("$device_name")
    done <<< "$rdma_output"

    IFS=,
    printf '%s\n' "${devices[*]}"
    unset IFS
}

plan_worker_rdma_assignments() {
    local gid_index=${1:-${MC_GID_INDEX:-}}
    local gpu_output=""
    local rdma_output=""
    local gpu_entry=""
    local rdma_entry=""
    local gpu_index=""
    local gpu_bus=""
    local gpu_numa=""
    local rdma_name=""
    local rdma_bus=""
    local rdma_numa=""
    local rdma_netdevs=""
    local chosen_device=""
    local chosen_reason=""
    local best_distance=1073741824
    local distance=0
    local status=0
    local -a gpu_entries=()
    local -a rdma_entries=()
    local -a exact_matches=()
    local -A rdma_bus_by_name=()
    local -A rdma_numa_by_name=()
    local -A rdma_netdevs_by_name=()
    local -A map_by_gpu_bdf=()
    local -A gpu_reason_by_bdf=()
    local -A used_device=()

    gpu_output="$(get_local_gpu_entries)" || gpu_output=""
    rdma_output="$(list_active_rdma_devices "$gid_index")" || rdma_output=""

    if [[ -n "${gpu_output//[[:space:]]/}" ]]; then
        mapfile -t gpu_entries <<< "$gpu_output"
    fi
    if [[ -n "${rdma_output//[[:space:]]/}" ]]; then
        mapfile -t rdma_entries <<< "$rdma_output"
    fi

    if [[ "${#gpu_entries[@]}" -eq 0 ]]; then
        echo "Error: no local GPUs were discovered with nvidia-smi." >&2
        return 1
    fi
    if [[ "${#rdma_entries[@]}" -eq 0 ]]; then
        echo "Error: no active RDMA devices matched MC_GID_INDEX=${gid_index}." >&2
        return 1
    fi

    for rdma_entry in "${rdma_entries[@]}"; do
        IFS='|' read -r rdma_name rdma_bus rdma_numa rdma_netdevs <<< "$rdma_entry"
        unset IFS
        rdma_bus_by_name["$rdma_name"]="$rdma_bus"
        rdma_numa_by_name["$rdma_name"]="$rdma_numa"
        rdma_netdevs_by_name["$rdma_name"]="$rdma_netdevs"
    done

    for gpu_entry in "${gpu_entries[@]}"; do
        IFS='|' read -r gpu_index gpu_bus _gpu_numa <<< "$gpu_entry"
        unset IFS

        exact_matches=()
        for rdma_name in "${!rdma_bus_by_name[@]}"; do
            if gpu_matches_rnic_netdev "$gpu_index" \
                "${rdma_netdevs_by_name[$rdma_name]}"; then
                exact_matches+=("$rdma_name")
            fi
        done

        if [[ "${#exact_matches[@]}" -gt 1 ]]; then
            IFS=$'\n' exact_matches=($(printf '%s\n' "${exact_matches[@]}" | sort))
            unset IFS
            status=1
        fi
        if [[ "${#exact_matches[@]}" -eq 0 ]]; then
            continue
        fi

        map_by_gpu_bdf["$gpu_bus"]="${exact_matches[0]}"
        if [[ "${#exact_matches[@]}" -eq 1 ]]; then
            gpu_reason_by_bdf["$gpu_bus"]="exact-netdev"
        else
            gpu_reason_by_bdf["$gpu_bus"]="ambiguous-exact-netdev"
        fi
        used_device["${exact_matches[0]}"]=1
    done

    for gpu_entry in "${gpu_entries[@]}"; do
        IFS='|' read -r _gpu_index gpu_bus gpu_numa <<< "$gpu_entry"
        unset IFS

        if [[ -n "${map_by_gpu_bdf[$gpu_bus]+x}" ]]; then
            continue
        fi

        chosen_device=""
        chosen_reason=""
        best_distance=1073741824

        for rdma_name in "${!rdma_bus_by_name[@]}"; do
            if [[ -n "${used_device[$rdma_name]+x}" ]]; then
                continue
            fi
            if [[ "$gpu_numa" == "-1" || \
                  "$gpu_numa" != "${rdma_numa_by_name[$rdma_name]}" ]]; then
                continue
            fi
            distance="$(pci_bus_distance "$gpu_bus" "${rdma_bus_by_name[$rdma_name]}")"
            if [[ -z "$chosen_device" || "$distance" -lt "$best_distance" || \
                  ( "$distance" -eq "$best_distance" && \
                    "$rdma_name" < "$chosen_device" ) ]]; then
                chosen_device="$rdma_name"
                chosen_reason="same-numa-pci-distance"
                best_distance="$distance"
            fi
        done

        if [[ -z "$chosen_device" ]]; then
            for rdma_name in "${!rdma_bus_by_name[@]}"; do
                if [[ -n "${used_device[$rdma_name]+x}" ]]; then
                    continue
                fi
                distance="$(pci_bus_distance "$gpu_bus" \
                    "${rdma_bus_by_name[$rdma_name]}")"
                if [[ -z "$chosen_device" || "$distance" -lt "$best_distance" || \
                      ( "$distance" -eq "$best_distance" && \
                        "$rdma_name" < "$chosen_device" ) ]]; then
                    chosen_device="$rdma_name"
                    chosen_reason="global-pci-distance"
                    best_distance="$distance"
                fi
            done
        fi

        if [[ -z "$chosen_device" ]]; then
            status=1
            gpu_reason_by_bdf["$gpu_bus"]="no-unused-rnic-match"
            continue
        fi

        map_by_gpu_bdf["$gpu_bus"]="$chosen_device"
        gpu_reason_by_bdf["$gpu_bus"]="$chosen_reason"
        used_device["$chosen_device"]=1
    done

    if [[ "${#map_by_gpu_bdf[@]}" -ne "${#gpu_entries[@]}" ]]; then
        status=1
    fi

    for gpu_entry in "${gpu_entries[@]}"; do
        IFS='|' read -r gpu_index gpu_bus gpu_numa <<< "$gpu_entry"
        unset IFS
        printf '%s|%s|%s|%s|%s\n' \
            "$gpu_index" \
            "$gpu_bus" \
            "$gpu_numa" \
            "${map_by_gpu_bdf[$gpu_bus]:-}" \
            "${gpu_reason_by_bdf[$gpu_bus]:-missing}"
    done

    return "$status"
}

get_worker_rdma_devices_csv() {
    local gid_index=${1:-${MC_GID_INDEX:-}}
    local plan_output=""
    local status=0
    local device_name=""
    local -a devices=()

    if plan_output="$(plan_worker_rdma_assignments "$gid_index")"; then
        status=0
    else
        status=$?
    fi
    [[ -n "${plan_output//[[:space:]]/}" ]] || return "$status"

    while IFS='|' read -r _gpu_index _gpu_bus _gpu_numa device_name _reason; do
        [[ -n "$device_name" ]] || continue
        devices+=("$device_name")
    done <<< "$plan_output"

    if [[ "${#devices[@]}" -gt 0 ]]; then
        IFS=,
        printf '%s\n' "${devices[*]}"
        unset IFS
    fi
    return "$status"
}
