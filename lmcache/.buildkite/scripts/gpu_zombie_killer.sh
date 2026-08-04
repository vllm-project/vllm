#!/usr/bin/env bash

# Helper daemon to run on your CI machine to kill GPU processes that are running for too long
# Copy somewhere
# chmod +x gpu_zombie_killer.sh
# nohup sudo ./gpu_zombie_killer.sh > /dev/null 2>&1 &

# Check logs:
# cat /var/log/gpu_zombie_killer.log

# Kill it:
# sudo kill -9 $(cat /tmp/gpu_zombie_killer.pid)

PIDFILE="/tmp/gpu_zombie_killer.pid"
LOGFILE="/var/log/gpu_zombie_killer.log"
MAX_SECONDS=$((60 * 60))   # 1 hour
SLEEP_INTERVAL=60          # check every 60s

# Track date for log rotation
LAST_DATE=$(date +%F)

# Save PID for tracking
echo $$ > "$PIDFILE"
echo "[GPU ZOMBIE KILLER] Started with PID $$ at $(date)" >> "$LOGFILE"

# Clean up on termination
trap 'echo "[GPU ZOMBIE KILLER] Stopping at $(date)" >> "$LOGFILE"; rm -f "$PIDFILE"; exit 0' SIGINT SIGTERM

while true; do
  now=$(date +%s)

  # Rotate logs if date changed
  current_date=$(date +%F)
  if [ "$current_date" != "$LAST_DATE" ]; then
    echo "[GPU ZOMBIE KILLER] Clearing logs at midnight ($current_date)" > "$LOGFILE"
    LAST_DATE="$current_date"
  fi

  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | awk '{print $1}' \
  | while read -r pid; do
      [ -d "/proc/$pid" ] || continue
      start_ticks=$(awk '{print $22}' /proc/$pid/stat)
      hertz=$(getconf CLK_TCK)
      boot_time=$(awk '/btime/ {print $2}' /proc/stat)
      start_time=$((boot_time + start_ticks / hertz))
      now=$(date +%s)
      runtime=$((now - start_time))

      if (( runtime > MAX_SECONDS )); then
        echo "[GPU ZOMBIE KILLER] Killing PID $pid (runtime ${runtime}s) at $(date)" >> "$LOGFILE"
        kill -9 "$pid"
      fi
    done

  sleep "$SLEEP_INTERVAL"
done
