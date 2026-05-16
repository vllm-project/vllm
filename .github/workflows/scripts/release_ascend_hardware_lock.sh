#!/bin/bash
set -euo pipefail

lock_pid=${ASCEND_HARDWARE_LOCK_PID:-}
setsid_enabled=${ASCEND_HARDWARE_LOCK_SETSID:-0}
log_file=${ASCEND_HARDWARE_LOCK_LOG:-}
holder_script=${ASCEND_HARDWARE_LOCK_HOLDER_SCRIPT:-}

if [[ -z "$lock_pid" ]]; then
  echo "[INFO] No Ascend hardware lock PID recorded; nothing to release"
  exit 0
fi

if ! kill -0 "$lock_pid" 2>/dev/null; then
  echo "[INFO] Ascend hardware lock holder is no longer running: $lock_pid"
  [[ -n "$holder_script" ]] && rm -f "$holder_script"
  exit 0
fi

echo "[INFO] Releasing Ascend hardware lock held by PID $lock_pid"
if [[ "$setsid_enabled" == "1" ]]; then
  kill -TERM -- "-$lock_pid" 2>/dev/null || kill -TERM "$lock_pid" 2>/dev/null || true
else
  kill -TERM "$lock_pid" 2>/dev/null || true
fi

for _ in $(seq 1 10); do
  if ! kill -0 "$lock_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done

if kill -0 "$lock_pid" 2>/dev/null; then
  echo "[WARN] Ascend hardware lock holder did not exit after TERM; killing it"
  if [[ "$setsid_enabled" == "1" ]]; then
    kill -KILL -- "-$lock_pid" 2>/dev/null || kill -KILL "$lock_pid" 2>/dev/null || true
  else
    kill -KILL "$lock_pid" 2>/dev/null || true
  fi
fi

[[ -n "$holder_script" ]] && rm -f "$holder_script"

if [[ -n "$log_file" && -f "$log_file" ]]; then
  echo "[INFO] Ascend hardware lock log tail:"
  tail -n 40 "$log_file" || true
fi

echo "[OK] Released Ascend hardware lock"