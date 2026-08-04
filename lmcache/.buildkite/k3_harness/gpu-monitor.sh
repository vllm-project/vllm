#!/usr/bin/env bash
# Host-level GPU monitoring script.
# Detects non-K8s processes holding GPU memory and logs warnings.
#
# Install as a cron job on the CI machine:
#   sudo crontab -e
#   */10 * * * * /path/to/gpu-monitor.sh >> /var/log/gpu-monitor.log 2>&1
#
# Or install with setup-gpu-monitor.sh for automatic setup.

set -euo pipefail

LOG_PREFIX="[gpu-monitor $(date '+%Y-%m-%d %H:%M:%S')]"
K8S_PIDS_FILE=$(mktemp)
trap 'rm -f "$K8S_PIDS_FILE"' EXIT

if ! command -v nvidia-smi &>/dev/null; then
    echo "$LOG_PREFIX ERROR: nvidia-smi not found"
    exit 1
fi

# Collect all PIDs using GPU compute
mapfile -t GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | sort -u | grep -v '^$'
)

if [[ ${#GPU_PIDS[@]} -eq 0 ]]; then
    # No GPU processes -- all clean
    exit 0
fi

# Collect K8s container PIDs (all processes running inside any K8s pod).
# K3s uses containerd; enumerate all container cgroups.
{
    shopt -s nullglob
    for cgroup_dir in /sys/fs/cgroup/*/kubepods* /sys/fs/cgroup/kubepods*; do
        if [[ -d "$cgroup_dir" ]]; then
            find "$cgroup_dir" -name "cgroup.procs" -exec cat {} \; 2>/dev/null
        fi
    done
    shopt -u nullglob
} | sort -u > "$K8S_PIDS_FILE" 2>/dev/null || true

# Also check via crictl if available (more reliable)
if command -v crictl &>/dev/null; then
    for container_id in $(crictl ps -q 2>/dev/null); do
        pid=$(crictl inspect --output go-template --template '{{.info.pid}}' "$container_id" 2>/dev/null || true)
        if [[ -n "$pid" && "$pid" != "0" ]]; then
            # Include the container PID and all its descendants
            echo "$pid" >> "$K8S_PIDS_FILE"
            pgrep -P "$pid" --ns "$pid" 2>/dev/null >> "$K8S_PIDS_FILE" || true
        fi
    done
fi

# Check each GPU process
STALE_FOUND=false
for pid in "${GPU_PIDS[@]}"; do
    [[ -z "$pid" ]] && continue

    # Is this PID a K8s process?
    if grep -qw "$pid" "$K8S_PIDS_FILE" 2>/dev/null; then
        continue
    fi

    # Check if this PID is inside any cgroup containing "kubepods"
    if [[ -f "/proc/$pid/cgroup" ]] && grep -q "kubepods" "/proc/$pid/cgroup" 2>/dev/null; then
        continue
    fi

    # This is a non-K8s process using a GPU
    STALE_FOUND=true
    local_cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | head -c 200 || echo "<unknown>")

    echo "$LOG_PREFIX STALE GPU PROCESS: PID=$pid cmd='$local_cmdline'"
    nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader 2>/dev/null \
        | grep "^${pid}," | while IFS=, read -r _ bus mem; do
            echo "$LOG_PREFIX   GPU bus=$bus memory=$mem"
        done

    # Check process age
    if [[ -f "/proc/$pid/stat" ]]; then
        local_start=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || echo "0")
        local_uptime=$(awk '{print int($1)}' /proc/uptime)
        local_clk_tck=$(getconf CLK_TCK)
        if [[ "$local_start" -gt 0 && "$local_clk_tck" -gt 0 ]]; then
            local_age_secs=$(( local_uptime - local_start / local_clk_tck ))
            local_age_hours=$(( local_age_secs / 3600 ))
            echo "$LOG_PREFIX   Age: ~${local_age_hours}h (${local_age_secs}s)"
        fi
    fi
done

if [[ "$STALE_FOUND" == "true" ]]; then
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX === GPU SUMMARY ==="
    nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv,noheader \
        | while IFS=, read -r idx used total free; do
            echo "$LOG_PREFIX   GPU $idx: used=$(echo $used | xargs), total=$(echo $total | xargs), free=$(echo $free | xargs)"
        done
    echo "$LOG_PREFIX WARNING: Stale non-K8s processes are holding GPU memory. CI tests may fail."
    echo "$LOG_PREFIX To fix: kill the stale PIDs listed above."
    exit 1
fi
