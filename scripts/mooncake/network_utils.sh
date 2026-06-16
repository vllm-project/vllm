#!/usr/bin/env bash

first_global_ipv4() {
    local host_ip=""

    if command -v ip >/dev/null 2>&1; then
        host_ip="$(ip -o -4 addr show up scope global 2>/dev/null | awk 'NR == 1 {split($4, a, "/"); print a[1]}')"
        if [[ -n "${host_ip//[[:space:]]/}" ]]; then
            printf '%s\n' "$host_ip"
            return 0
        fi
    fi

    host_ip="$(hostname -I 2>/dev/null | awk '{for (i = 1; i <= NF; ++i) if ($i !~ /^127\\./) { print $i; exit }}')"
    if [[ -n "${host_ip//[[:space:]]/}" ]]; then
        printf '%s\n' "$host_ip"
        return 0
    fi

    return 1
}

detect_local_advertise_host() {
    local preferred="${1:-}"
    local host_name=""

    if [[ -n "${preferred//[[:space:]]/}" ]]; then
        printf '%s\n' "$preferred"
        return 0
    fi

    if first_global_ipv4; then
        return 0
    fi

    host_name="$(hostname -f 2>/dev/null || hostname 2>/dev/null || true)"
    if [[ -n "${host_name//[[:space:]]/}" ]]; then
        printf '%s\n' "$host_name"
        return 0
    fi

    return 1
}

tcp_port_is_listening() {
    local port=$1
    ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "(^|[:.])${port}$"
}

all_tcp_ports_listening() {
    local port=""
    for port in "$@"; do
        if ! tcp_port_is_listening "$port"; then
            return 1
        fi
    done
    return 0
}

any_tcp_ports_in_use() {
    local port=""
    for port in "$@"; do
        if tcp_port_is_listening "$port"; then
            return 0
        fi
    done
    return 1
}

wait_for_tcp_port() {
    local port=$1
    local timeout_s=${2:-30}
    local pid=${3:-}

    for _ in $(seq "$timeout_s"); do
        if [[ -n "${pid:-}" ]] && ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        if tcp_port_is_listening "$port"; then
            return 0
        fi
        sleep 1
    done
    return 1
}
