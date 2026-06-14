#!/usr/bin/env bash

set -u
set -o pipefail

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

usage() {
    cat <<'EOF'
Usage: report_network_gpu_topology.sh

Generate a human-readable report about:
- physical NICs and their drivers / PCI placement
- RDMA devices and whether they are Ethernet RDMA or InfiniBand RDMA
- GPUs, NUMA affinity, and local CPU affinity
- NVIDIA GPU/NIC topology and NVLink visibility
- a concise summary of likely intra-node and inter-node fabrics

Notes:
- virtual interfaces such as loopback / veth are omitted by default
- the report is best-effort and degrades cleanly when rdma, ethtool, lspci,
  or nvidia-smi are unavailable
EOF
}

trim() {
    local s="$*"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

join_by() {
    local sep="$1"
    shift || true
    local first=1
    local item
    for item in "$@"; do
        if (( first )); then
            printf '%s' "$item"
            first=0
        else
            printf '%s%s' "$sep" "$item"
        fi
    done
}

section() {
    printf '\n== %s ==\n' "$1"
}

subsection() {
    printf '\n[%s]\n' "$1"
}

read_first_line() {
    local path="$1"
    local line=""
    if [[ -r "$path" ]]; then
        IFS= read -r line < "$path" || true
    fi
    if [[ -z "$line" ]]; then
        printf 'N/A'
    else
        printf '%s' "$line"
    fi
}

normalize_bdf() {
    local raw
    raw=$(trim "$*")
    if [[ -z "$raw" || "$raw" == "N/A" ]]; then
        printf 'N/A'
        return
    fi
    raw="${raw#00000000:}"
    if [[ "$raw" =~ ^[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$ ]]; then
        raw="0000:$raw"
    fi
    printf '%s' "$(printf '%s' "$raw" | tr 'A-F' 'a-f')"
}

pci_bdf_from_path() {
    local path="$1"
    local base=""
    if [[ -n "$path" ]]; then
        base=$(basename "$path")
    fi
    if [[ "$base" =~ ^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$ ]]; then
        normalize_bdf "$base"
    else
        printf 'N/A'
    fi
}

device_numa() {
    local bdf="$1"
    local path="/sys/bus/pci/devices/$bdf/numa_node"
    if [[ "$bdf" == "N/A" || ! -r "$path" ]]; then
        printf 'N/A'
        return
    fi
    read_first_line "$path"
}

device_cpulist() {
    local bdf="$1"
    local path="/sys/bus/pci/devices/$bdf/local_cpulist"
    if [[ "$bdf" == "N/A" || ! -r "$path" ]]; then
        printf 'N/A'
        return
    fi
    read_first_line "$path"
}

pci_desc() {
    local bdf="$1"
    if ! have_cmd lspci || [[ "$bdf" == "N/A" ]]; then
        printf 'N/A'
        return
    fi
    local line
    line=$(lspci -nn -s "$bdf" 2>/dev/null | sed 's/^[^ ]* //')
    if [[ -z "$line" ]]; then
        printf 'N/A'
    else
        printf '%s' "$line"
    fi
}

netdev_driver() {
    local dev="$1"
    local driver=""
    if have_cmd ethtool; then
        driver=$(ethtool -i "$dev" 2>/dev/null | awk -F': ' '/^driver:/{print $2; exit}')
    fi
    if [[ -z "$driver" && -L "/sys/class/net/$dev/device/driver" ]]; then
        driver=$(basename "$(readlink -f "/sys/class/net/$dev/device/driver")")
    fi
    if [[ -z "$driver" ]]; then
        printf 'N/A'
    else
        printf '%s' "$driver"
    fi
}

netdev_businfo() {
    local dev="$1"
    local bus=""
    if have_cmd ethtool; then
        bus=$(ethtool -i "$dev" 2>/dev/null | awk -F': ' '/^bus-info:/{print $2; exit}')
    fi
    if [[ -z "$bus" || "$bus" == "N/A" ]]; then
        bus=$(pci_bdf_from_path "$(readlink -f "/sys/class/net/$dev/device" 2>/dev/null || true)")
    fi
    normalize_bdf "$bus"
}

netdev_addrs() {
    local dev="$1"
    local -a addrs=()
    if have_cmd ip; then
        mapfile -t addrs < <(ip -o addr show dev "$dev" 2>/dev/null | awk '$3 ~ /^(inet|inet6)$/ {print $4}')
    fi
    if (( ${#addrs[@]} == 0 )); then
        printf '-'
    else
        join_by ', ' "${addrs[@]}"
    fi
}

strip_numeric_prefix() {
    local value="$1"
    value=$(trim "$value")
    printf '%s' "${value#*: }"
}

relation_rank() {
    case "$1" in
        X) printf '0' ;;
        NV*) printf '0' ;;
        PIX) printf '1' ;;
        PXB) printf '2' ;;
        PHB) printf '3' ;;
        NODE) printf '4' ;;
        SYS) printf '5' ;;
        *) printf '99' ;;
    esac
}

rdma_fabric_type() {
    local rdma_dev="$1"
    local link_layer="${RDMA_LINK_LAYER[$rdma_dev]:-N/A}"
    if [[ "$link_layer" == "Ethernet" ]]; then
        printf 'Ethernet RDMA (RoCE-family)'
    elif [[ "$link_layer" == "InfiniBand" ]]; then
        printf 'InfiniBand RDMA'
    elif [[ "$link_layer" == "N/A" ]]; then
        printf 'RDMA (link layer unavailable)'
    else
        printf 'RDMA over %s' "$link_layer"
    fi
}

print_kv() {
    printf '  %-15s %s\n' "$1" "$2"
}

declare -a NETDEVS_PHYS=()
declare -i VIRTUAL_OMITTED=0
declare -A NETDEV_STATE=()
declare -A NETDEV_MTU=()
declare -A NETDEV_MAC=()
declare -A NETDEV_DRIVER=()
declare -A NETDEV_BDF=()
declare -A NETDEV_NUMA=()
declare -A NETDEV_CPUS=()
declare -A NETDEV_ADDRS=()
declare -A NETDEV_DESC=()
declare -A NETDEV_RDMA=()

declare -a RDMA_DEVS=()
declare -A RDMA_NETDEV=()
declare -A RDMA_BDF=()
declare -A RDMA_NUMA=()
declare -A RDMA_CPUS=()
declare -A RDMA_FW=()
declare -A RDMA_HCA=()
declare -A RDMA_LINK_LAYER=()
declare -A RDMA_STATE=()
declare -A RDMA_PHYS_STATE=()

declare -a GPU_INDEXES=()
declare -A GPU_BDF=()
declare -A GPU_NAME=()
declare -A GPU_UUID=()
declare -A GPU_NUMA=()
declare -A GPU_CPUS=()

declare -a TOPO_COLS=()
declare -a TOPO_GPU_KEYS=()
declare -a TOPO_NIC_KEYS=()
declare -A TOPO_ALIAS_TO_RDMA=()
declare -A TOPO_RDMA_TO_ALIAS=()
declare -A TOPO_REL=()
TOPO_RAW=""

collect_netdevs() {
    local net_path dev resolved bdf
    for net_path in /sys/class/net/*; do
        dev=$(basename "$net_path")
        resolved=$(readlink -f "$net_path" 2>/dev/null || true)
        if [[ "$dev" == "lo" || "$resolved" == /sys/devices/virtual/* ]]; then
            ((VIRTUAL_OMITTED += 1))
            continue
        fi

        NETDEVS_PHYS+=("$dev")
        NETDEV_STATE["$dev"]=$(read_first_line "/sys/class/net/$dev/operstate")
        NETDEV_MTU["$dev"]=$(read_first_line "/sys/class/net/$dev/mtu")
        NETDEV_MAC["$dev"]=$(read_first_line "/sys/class/net/$dev/address")
        NETDEV_DRIVER["$dev"]=$(netdev_driver "$dev")
        bdf=$(netdev_businfo "$dev")
        NETDEV_BDF["$dev"]="$bdf"
        NETDEV_NUMA["$dev"]=$(device_numa "$bdf")
        NETDEV_CPUS["$dev"]=$(device_cpulist "$bdf")
        NETDEV_ADDRS["$dev"]=$(netdev_addrs "$dev")
        NETDEV_DESC["$dev"]=$(pci_desc "$bdf")
    done

    if (( ${#NETDEVS_PHYS[@]} > 0 )); then
        mapfile -t NETDEVS_PHYS < <(printf '%s\n' "${NETDEVS_PHYS[@]}" | sort)
    fi
}

collect_rdma() {
    local rdma_out line dev netdev state phys_state rdma_path bdf

    if have_cmd rdma; then
        rdma_out=$(rdma link show 2>/dev/null || true)
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            if [[ "$line" =~ ^link[[:space:]]+([^/]+)/[0-9]+[[:space:]]+state[[:space:]]+([^[:space:]]+)[[:space:]]+physical_state[[:space:]]+([^[:space:]]+).*[[:space:]]netdev[[:space:]]+([^[:space:]]+) ]]; then
                dev="${BASH_REMATCH[1]}"
                state="${BASH_REMATCH[2]}"
                phys_state="${BASH_REMATCH[3]}"
                netdev="${BASH_REMATCH[4]}"
                RDMA_NETDEV["$dev"]="$netdev"
                RDMA_STATE["$dev"]="$state"
                RDMA_PHYS_STATE["$dev"]="$phys_state"
            fi
        done <<< "$rdma_out"
    fi

    for rdma_path in /sys/class/infiniband/*; do
        [[ -e "$rdma_path" ]] || continue
        dev=$(basename "$rdma_path")
        RDMA_DEVS+=("$dev")
        bdf=$(pci_bdf_from_path "$(readlink -f "$rdma_path/device" 2>/dev/null || true)")
        RDMA_BDF["$dev"]="$bdf"
        RDMA_NUMA["$dev"]=$(device_numa "$bdf")
        RDMA_CPUS["$dev"]=$(device_cpulist "$bdf")
        RDMA_FW["$dev"]=$(read_first_line "$rdma_path/fw_ver")
        RDMA_HCA["$dev"]=$(read_first_line "$rdma_path/hca_type")
        RDMA_LINK_LAYER["$dev"]=$(read_first_line "$rdma_path/ports/1/link_layer")

        if [[ -z "${RDMA_STATE[$dev]:-}" ]]; then
            RDMA_STATE["$dev"]=$(strip_numeric_prefix "$(read_first_line "$rdma_path/ports/1/state")")
        fi
        if [[ -z "${RDMA_PHYS_STATE[$dev]:-}" ]]; then
            RDMA_PHYS_STATE["$dev"]=$(strip_numeric_prefix "$(read_first_line "$rdma_path/ports/1/phys_state")")
        fi

        if [[ -z "${RDMA_NETDEV[$dev]:-}" ]]; then
            local candidate
            for candidate in "${NETDEVS_PHYS[@]}"; do
                if [[ -e "/sys/class/net/$candidate/device/infiniband/$dev" ]]; then
                    RDMA_NETDEV["$dev"]="$candidate"
                    break
                fi
            done
        fi
    done

    if (( ${#RDMA_DEVS[@]} > 0 )); then
        mapfile -t RDMA_DEVS < <(printf '%s\n' "${RDMA_DEVS[@]}" | sort)
    fi

    local rdma_dev
    for rdma_dev in "${RDMA_DEVS[@]}"; do
        netdev="${RDMA_NETDEV[$rdma_dev]:-}"
        if [[ -n "$netdev" ]]; then
            if [[ -n "${NETDEV_RDMA[$netdev]:-}" ]]; then
                NETDEV_RDMA["$netdev"]+=",${rdma_dev}"
            else
                NETDEV_RDMA["$netdev"]="$rdma_dev"
            fi
        fi
    done
}

collect_gpus() {
    local line idx bdf name uuid
    if ! have_cmd nvidia-smi; then
        return
    fi

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        IFS=',' read -r idx bdf name uuid <<< "$line"
        idx=$(trim "$idx")
        bdf=$(normalize_bdf "$bdf")
        name=$(trim "$name")
        uuid=$(trim "$uuid")
        GPU_INDEXES+=("$idx")
        GPU_BDF["$idx"]="$bdf"
        GPU_NAME["$idx"]="$name"
        GPU_UUID["$idx"]="$uuid"
        GPU_NUMA["$idx"]=$(device_numa "$bdf")
        GPU_CPUS["$idx"]=$(device_cpulist "$bdf")
    done < <(nvidia-smi --query-gpu=index,pci.bus_id,name,uuid --format=csv,noheader 2>/dev/null || true)
}

parse_topology() {
    local topo_clean parsed line idx col alias rdma_dev row col_idx rel
    if ! have_cmd nvidia-smi; then
        return
    fi

    topo_clean=$(nvidia-smi topo -m 2>/dev/null | sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' || true)
    if [[ -z "$topo_clean" ]]; then
        return
    fi
    TOPO_RAW="$topo_clean"

    parsed=$(printf '%s\n' "$topo_clean" | awk -F'\t' '
        function trim(s) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
            return s
        }
        NR == 1 {
            n = 0
            for (i = 1; i <= NF; i++) {
                f = trim($i)
                if (f ~ /^(GPU[0-9]+|NIC[0-9]+)$/) {
                    cols[++n] = f
                }
            }
            print "COUNT", n
            for (i = 1; i <= n; i++) {
                print "COL", i, cols[i]
            }
            next
        }
        /^NIC Legend:/ {
            legend = 1
            next
        }
        legend {
            line = $0
            if (line ~ /^[[:space:]]*NIC[0-9]+:[[:space:]]*/) {
                sub(/^[[:space:]]*/, "", line)
                split(line, a, /:[[:space:]]*/)
                print "ALIAS", a[1], a[2]
            }
        }
        {
            row = trim($1)
            if (row ~ /^(GPU[0-9]+|NIC[0-9]+)$/) {
                printf "ROW %s", row
                out = 0
                for (i = 2; i <= NF && out < n; i++) {
                    f = trim($i)
                    if (f == "") {
                        continue
                    }
                    printf " %s", f
                    out++
                }
                print ""
            }
        }')

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        set -- $line
        case "$1" in
            COUNT)
                ;;
            COL)
                idx="$2"
                col="$3"
                TOPO_COLS[$((idx - 1))]="$col"
                if [[ "$col" == GPU* ]]; then
                    TOPO_GPU_KEYS+=("$col")
                elif [[ "$col" == NIC* ]]; then
                    TOPO_NIC_KEYS+=("$col")
                fi
                ;;
            ALIAS)
                alias="$2"
                rdma_dev="$3"
                TOPO_ALIAS_TO_RDMA["$alias"]="$rdma_dev"
                TOPO_RDMA_TO_ALIAS["$rdma_dev"]="$alias"
                ;;
            ROW)
                row="$2"
                shift 2
                col_idx=0
                for rel in "$@"; do
                    if (( col_idx < ${#TOPO_COLS[@]} )); then
                        col="${TOPO_COLS[$col_idx]}"
                        TOPO_REL["$row|$col"]="$rel"
                    fi
                    ((col_idx += 1))
                done
                ;;
        esac
    done <<< "$parsed"
}

best_nics_for_gpu() {
    local gpu_key="$1"
    local best_rank=999
    local -a matches=()
    local alias rdma_dev netdev rel rank

    for alias in "${TOPO_NIC_KEYS[@]}"; do
        rel="${TOPO_REL[$gpu_key|$alias]:-}"
        [[ -z "$rel" || "$rel" == "X" ]] && continue
        rank=$(relation_rank "$rel")
        if (( rank < best_rank )); then
            best_rank=$rank
            matches=()
        fi
        if (( rank == best_rank )); then
            rdma_dev="${TOPO_ALIAS_TO_RDMA[$alias]:-$alias}"
            netdev="${RDMA_NETDEV[$rdma_dev]:-unknown-netdev}"
            matches+=("${netdev} (${rdma_dev}, ${rel})")
        fi
    done

    if (( ${#matches[@]} == 0 )); then
        printf 'N/A'
    else
        join_by '; ' "${matches[@]}"
    fi
}

best_gpus_for_alias() {
    local alias="$1"
    local best_rank=999
    local -a matches=()
    local gpu_key rel rank

    for gpu_key in "${TOPO_GPU_KEYS[@]}"; do
        rel="${TOPO_REL[$alias|$gpu_key]:-}"
        [[ -z "$rel" || "$rel" == "X" ]] && continue
        rank=$(relation_rank "$rel")
        if (( rank < best_rank )); then
            best_rank=$rank
            matches=()
        fi
        if (( rank == best_rank )); then
            matches+=("${gpu_key} (${rel})")
        fi
    done

    if (( ${#matches[@]} == 0 )); then
        printf 'N/A'
    else
        join_by '; ' "${matches[@]}"
    fi
}

nvlink_summary() {
    local i j a b rel total_pairs=0 nv_pairs=0
    local -a pairs=()
    for ((i = 0; i < ${#TOPO_GPU_KEYS[@]}; i++)); do
        for ((j = i + 1; j < ${#TOPO_GPU_KEYS[@]}; j++)); do
            a="${TOPO_GPU_KEYS[$i]}"
            b="${TOPO_GPU_KEYS[$j]}"
            rel="${TOPO_REL[$a|$b]:-}"
            ((total_pairs += 1))
            if [[ "$rel" == NV* ]]; then
                ((nv_pairs += 1))
                pairs+=("${a}<->${b} (${rel})")
            fi
        done
    done

    if (( total_pairs == 0 )); then
        printf 'No multi-GPU topology available'
    elif (( nv_pairs == 0 )); then
        printf 'No NVLink relationships reported across %d GPU pairs' "$total_pairs"
    elif (( nv_pairs == total_pairs )); then
        printf 'NVLink reported on all %d GPU pairs: %s' \
            "$total_pairs" "$(join_by '; ' "${pairs[@]}")"
    else
        printf 'NVLink reported on %d/%d GPU pairs: %s' \
            "$nv_pairs" "$total_pairs" "$(join_by '; ' "${pairs[@]}")"
    fi
}

emit_host_summary() {
    section "Host Summary"
    print_kv "hostname" "$(hostname 2>/dev/null || echo N/A)"
    print_kv "kernel" "$(uname -r 2>/dev/null || echo N/A)"
    print_kv "arch" "$(uname -m 2>/dev/null || echo N/A)"
    print_kv "date_utc" "$(date -u '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || echo N/A)"
}

emit_netdev_inventory() {
    section "Physical NIC Inventory"
    if (( ${#NETDEVS_PHYS[@]} == 0 )); then
        printf 'No physical netdevs found.\n'
        return
    fi

    local dev rdma_info kind rdma_dev rdma_types
    for dev in "${NETDEVS_PHYS[@]}"; do
        subsection "$dev"
        if [[ -n "${NETDEV_RDMA[$dev]:-}" ]]; then
            kind="RDMA-capable Ethernet NIC"
            IFS=',' read -r -a rdma_types <<< "${NETDEV_RDMA[$dev]}"
            local -a rdma_parts=()
            for rdma_dev in "${rdma_types[@]}"; do
                rdma_parts+=("${rdma_dev} [$(rdma_fabric_type "$rdma_dev"); state=${RDMA_STATE[$rdma_dev]:-N/A}/${RDMA_PHYS_STATE[$rdma_dev]:-N/A}]")
            done
            rdma_info=$(join_by '; ' "${rdma_parts[@]}")
        else
            kind="Regular Ethernet NIC"
            rdma_info="no"
        fi
        print_kv "type" "$kind"
        print_kv "driver" "${NETDEV_DRIVER[$dev]}"
        print_kv "pci_bdf" "${NETDEV_BDF[$dev]}"
        print_kv "pci_desc" "${NETDEV_DESC[$dev]}"
        print_kv "state" "${NETDEV_STATE[$dev]}"
        print_kv "mac" "${NETDEV_MAC[$dev]}"
        print_kv "ips" "${NETDEV_ADDRS[$dev]}"
        print_kv "mtu" "${NETDEV_MTU[$dev]}"
        print_kv "numa" "${NETDEV_NUMA[$dev]}"
        print_kv "local_cpus" "${NETDEV_CPUS[$dev]}"
        print_kv "rdma" "$rdma_info"
    done

    if (( VIRTUAL_OMITTED > 0 )); then
        printf '\nVirtual-only interfaces omitted: %d\n' "$VIRTUAL_OMITTED"
    fi
}

emit_rdma_inventory() {
    section "RDMA Inventory"
    if (( ${#RDMA_DEVS[@]} == 0 )); then
        printf 'No RDMA devices found.\n'
        return
    fi

    local dev
    for dev in "${RDMA_DEVS[@]}"; do
        subsection "$dev"
        print_kv "netdev" "${RDMA_NETDEV[$dev]:-N/A}"
        print_kv "rdma_fabric" "$(rdma_fabric_type "$dev")"
        print_kv "link_layer" "${RDMA_LINK_LAYER[$dev]}"
        print_kv "state" "${RDMA_STATE[$dev]:-N/A}"
        print_kv "phys_state" "${RDMA_PHYS_STATE[$dev]:-N/A}"
        print_kv "pci_bdf" "${RDMA_BDF[$dev]}"
        print_kv "pci_desc" "$(pci_desc "${RDMA_BDF[$dev]}")"
        print_kv "firmware" "${RDMA_FW[$dev]}"
        print_kv "hca_type" "${RDMA_HCA[$dev]}"
        print_kv "numa" "${RDMA_NUMA[$dev]}"
        print_kv "local_cpus" "${RDMA_CPUS[$dev]}"
    done
}

emit_gpu_inventory() {
    section "GPU Inventory"
    if (( ${#GPU_INDEXES[@]} == 0 )); then
        printf 'No NVIDIA GPUs found or nvidia-smi unavailable.\n'
        return
    fi

    local idx
    for idx in "${GPU_INDEXES[@]}"; do
        subsection "GPU$idx"
        print_kv "name" "${GPU_NAME[$idx]}"
        print_kv "uuid" "${GPU_UUID[$idx]}"
        print_kv "pci_bdf" "${GPU_BDF[$idx]}"
        print_kv "numa" "${GPU_NUMA[$idx]}"
        print_kv "local_cpus" "${GPU_CPUS[$idx]}"
    done
}

emit_topology_matrix() {
    section "NVIDIA Topology Matrix"
    if [[ -z "$TOPO_RAW" ]]; then
        printf 'nvidia-smi topo -m unavailable.\n'
        return
    fi
    printf '%s\n' "$TOPO_RAW"
}

emit_affinity_summary() {
    section "Derived GPU / NIC Affinity"
    if [[ -z "$TOPO_RAW" || ${#TOPO_NIC_KEYS[@]} -eq 0 || ${#TOPO_GPU_KEYS[@]} -eq 0 ]]; then
        printf 'Detailed GPU/NIC topology unavailable.\n'
        return
    fi

    local idx alias rdma_dev netdev

    printf 'Closest RDMA NICs per GPU:\n'
    for idx in "${GPU_INDEXES[@]}"; do
        printf '  GPU%s -> %s\n' "$idx" "$(best_nics_for_gpu "GPU$idx")"
    done

    printf '\nClosest GPUs per RDMA NIC:\n'
    for alias in "${TOPO_NIC_KEYS[@]}"; do
        rdma_dev="${TOPO_ALIAS_TO_RDMA[$alias]:-$alias}"
        netdev="${RDMA_NETDEV[$rdma_dev]:-unknown-netdev}"
        printf '  %s / %s -> %s\n' "$netdev" "$rdma_dev" "$(best_gpus_for_alias "$alias")"
    done

    printf '\nNVLink summary:\n'
    printf '  %s\n' "$(nvlink_summary)"
}

emit_final_summary() {
    section "Final Summary"

    local -a regular_nics=()
    local -a rdma_nics=()
    local dev rdma_dev alias summary_line

    for dev in "${NETDEVS_PHYS[@]}"; do
        if [[ -n "${NETDEV_RDMA[$dev]:-}" ]]; then
            rdma_nics+=("$dev")
        else
            regular_nics+=("$dev")
        fi
    done

    if (( ${#regular_nics[@]} > 0 )); then
        printf 'Regular Ethernet NICs: %s\n' "$(join_by ', ' "${regular_nics[@]}")"
    else
        printf 'Regular Ethernet NICs: none\n'
    fi

    if (( ${#rdma_nics[@]} > 0 )); then
        local -a rdma_lines=()
        for dev in "${rdma_nics[@]}"; do
            IFS=',' read -r -a aliases <<< "${NETDEV_RDMA[$dev]}"
            local -a nic_parts=()
            for rdma_dev in "${aliases[@]}"; do
                nic_parts+=("${rdma_dev} ($(rdma_fabric_type "$rdma_dev"))")
            done
            rdma_lines+=("${dev}: $(join_by '; ' "${nic_parts[@]}")")
        done
        printf 'RDMA NICs for inter-node traffic: %s\n' "$(join_by ' | ' "${rdma_lines[@]}")"
    else
        printf 'RDMA NICs for inter-node traffic: none\n'
    fi

    if [[ -n "$TOPO_RAW" && ${#TOPO_GPU_KEYS[@]} -gt 1 ]]; then
        printf 'Intra-node GPU fabric: %s\n' "$(nvlink_summary)"
    elif (( ${#GPU_INDEXES[@]} > 1 )); then
        printf 'Intra-node GPU fabric: multiple GPUs detected, but NVLink topology is unavailable\n'
    else
        printf 'Intra-node GPU fabric: not applicable\n'
    fi

    if [[ -n "$TOPO_RAW" && ${#TOPO_NIC_KEYS[@]} -gt 0 ]]; then
        printf 'Best GPU to RDMA-NIC pairings:\n'
        local idx
        for idx in "${GPU_INDEXES[@]}"; do
            summary_line=$(best_nics_for_gpu "GPU$idx")
            printf '  GPU%s -> %s\n' "$idx" "$summary_line"
        done
    elif (( ${#GPU_INDEXES[@]} > 0 && ${#RDMA_DEVS[@]} > 0 )); then
        printf 'Best GPU to RDMA-NIC pairings: topology matrix unavailable; use shared NUMA node as first approximation\n'
    else
        printf 'Best GPU to RDMA-NIC pairings: not applicable\n'
    fi

    printf '\nInterpretation:\n'
    printf '  - NVLink is an intra-node GPU fabric. NICs themselves are not NVLink devices.\n'
    printf '  - RDMA NICs are the inter-node data path when GPUDirect RDMA / NCCL / UCX use the network.\n'
    printf '  - The topology matrix shows proximity, not proof that a live workload is using GDRDMA.\n'
}

main() {
    if (( $# > 0 )); then
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            *)
                printf 'Unknown argument: %s\n\n' "$1" >&2
                usage >&2
                exit 1
                ;;
        esac
    fi

    collect_netdevs
    collect_rdma
    collect_gpus
    parse_topology

    emit_host_summary
    emit_netdev_inventory
    emit_rdma_inventory
    emit_gpu_inventory
    emit_topology_matrix
    emit_affinity_summary
    emit_final_summary
}

main "$@"
