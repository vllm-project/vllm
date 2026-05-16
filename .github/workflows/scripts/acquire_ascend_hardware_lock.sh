#!/bin/bash
set -euo pipefail

lock_file=${ASCEND_HARDWARE_LOCK_FILE:-/tmp/vllm-hust-ascend-hardware.lock}
timeout_seconds=${ASCEND_HARDWARE_LOCK_TIMEOUT_SECONDS:-7200}
lease_seconds=${ASCEND_HARDWARE_LOCK_LEASE_SECONDS:-10800}
runtime_dir=${RUNNER_TEMP:-/tmp}
status_file=$(mktemp "${runtime_dir%/}/ascend-hardware-lock-status.XXXXXX")
holder_script=$(mktemp "${runtime_dir%/}/ascend-hardware-lock-holder.XXXXXX")
log_file=${ASCEND_HARDWARE_LOCK_LOG:-${runtime_dir%/}/ascend-hardware-lock-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}.log}

if ! command -v flock >/dev/null 2>&1; then
  echo "[ERROR] flock is required to serialize Ascend hardware jobs on this host" >&2
  exit 1
fi

mkdir -p "$(dirname "$lock_file")" "$(dirname "$log_file")"

cat > "$holder_script" <<'HOLDER'
#!/bin/bash
set -euo pipefail

lock_file=$1
timeout_seconds=$2
lease_seconds=$3
status_file=$4
log_file=$5

exec >>"$log_file" 2>&1
echo "[INFO] Waiting for Ascend hardware lock: $lock_file"
echo "[INFO] run=${GITHUB_RUN_ID:-manual} attempt=${GITHUB_RUN_ATTEMPT:-1} repo=${GITHUB_REPOSITORY:-unknown} workspace=${GITHUB_WORKSPACE:-unknown} pid=$$"

exec 9>"$lock_file"
if ! flock -w "$timeout_seconds" 9; then
  echo "failed timeout" > "$status_file"
  echo "[ERROR] Timed out waiting for Ascend hardware lock after ${timeout_seconds}s"
  exit 1
fi

{
  echo "run=${GITHUB_RUN_ID:-manual}"
  echo "attempt=${GITHUB_RUN_ATTEMPT:-1}"
  echo "repo=${GITHUB_REPOSITORY:-unknown}"
  echo "workspace=${GITHUB_WORKSPACE:-unknown}"
  echo "pid=$$"
  date -Iseconds
} >&9

echo "acquired $$" > "$status_file"
echo "[INFO] Acquired Ascend hardware lock"

end_time=$((SECONDS + lease_seconds))
while (( SECONDS < end_time )); do
  sleep 30
done

echo "[WARN] Ascend hardware lock lease expired after ${lease_seconds}s; releasing lock"
HOLDER
chmod +x "$holder_script"

setsid_enabled=0
if command -v setsid >/dev/null 2>&1; then
  setsid_enabled=1
  setsid "$holder_script" "$lock_file" "$timeout_seconds" "$lease_seconds" "$status_file" "$log_file" &
else
  "$holder_script" "$lock_file" "$timeout_seconds" "$lease_seconds" "$status_file" "$log_file" &
fi
holder_pid=$!

deadline=$((SECONDS + timeout_seconds + 15))
while (( SECONDS < deadline )); do
  if [[ -s "$status_file" ]]; then
    read -r status _ < "$status_file"
    if [[ "$status" == "acquired" ]]; then
      break
    fi
    echo "[ERROR] Failed to acquire Ascend hardware lock"
    tail -n 80 "$log_file" || true
    exit 1
  fi

  if ! kill -0 "$holder_pid" 2>/dev/null; then
    echo "[ERROR] Ascend hardware lock holder exited before acquiring the lock"
    tail -n 80 "$log_file" || true
    exit 1
  fi

  sleep 1
done

if [[ ! -s "$status_file" ]]; then
  echo "[ERROR] Timed out waiting for Ascend hardware lock acquisition status"
  kill "$holder_pid" 2>/dev/null || true
  tail -n 80 "$log_file" || true
  exit 1
fi

rm -f "$status_file"

echo "[OK] Acquired Ascend hardware lock: $lock_file"
echo "[OK] Lock holder PID: $holder_pid"
echo "[OK] Lock log: $log_file"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "ASCEND_HARDWARE_LOCK_PID=$holder_pid"
    echo "ASCEND_HARDWARE_LOCK_SETSID=$setsid_enabled"
    echo "ASCEND_HARDWARE_LOCK_LOG=$log_file"
    echo "ASCEND_HARDWARE_LOCK_HOLDER_SCRIPT=$holder_script"
  } >> "$GITHUB_ENV"
fi