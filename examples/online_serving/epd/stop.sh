#!/usr/bin/env bash
#
# Reads pid.txt created by run_servers.sh and kills every process.
#

set -euo pipefail

PID_FILE="./pid.txt"
[[ -f "$PID_FILE" ]] || {
  echo "No $PID_FILE found – nothing to stop."
  exit 0
}

echo "Stopping processes listed in $PID_FILE …"

while read -r pid; do
  [[ -z "$pid" ]] && continue   # skip blank lines
  if kill -0 "$pid" 2>/dev/null; then
    echo "  → SIGTERM $pid"
    kill "$pid"
    # wait up to 5 s, escalate to SIGKILL if still alive
    for _ in {1..5}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "  → SIGKILL $pid"
      kill -9 "$pid" || true
    fi
  else
    echo "  → PID $pid is already gone"
  fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "Done."