#!/usr/bin/env bash
# SSH to every gb200-rack1-* node visible via getent and force-kill
# mooncake_client if it is running.
set -euo pipefail

SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-5}"
SSH_OPTIONS=(
    -o BatchMode=yes
    -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
    -o StrictHostKeyChecking=accept-new
)

discover_hosts() {
    getent hosts \
        | awk '{print $2}' \
        | grep -E '^gb200-rack1-[0-9]+$' \
        | sort -uV
}

mapfile -t HOSTS < <(discover_hosts)

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    echo "No hosts matching gb200-rack1-* found via getent hosts." >&2
    exit 1
fi

echo "Targeting ${#HOSTS[@]} hosts: ${HOSTS[*]}"

killed=0
not_running=0
ssh_failed=0

for host in "${HOSTS[@]}"; do
    if ! output="$(
        ssh "${SSH_OPTIONS[@]}" "$host" \
            'rm -rf /mnt/data/mooncake_offload; rm -rf /tmp/yifanqiao/mooncake_offload
             if pgrep -x mooncake_client >/dev/null; then
                 pkill -9 -x mooncake_client && echo killed
             else
                 echo not-running
             fi' \
            2>&1
    )"; then
        printf '%s: ssh-failed (%s)\n' "$host" "$output"
        ((ssh_failed += 1))
        continue
    fi

    case "$output" in
        killed)
            printf '%s: killed\n' "$host"
            ((killed += 1))
            ;;
        not-running)
            printf '%s: not-running\n' "$host"
            ((not_running += 1))
            ;;
        *)
            printf '%s: unexpected-output (%s)\n' "$host" "$output"
            ((ssh_failed += 1))
            ;;
    esac
done

printf 'Summary: killed=%d not-running=%d ssh-failed=%d\n' \
    "$killed" "$not_running" "$ssh_failed"

if [[ $ssh_failed -ne 0 ]]; then
    exit 1
fi
