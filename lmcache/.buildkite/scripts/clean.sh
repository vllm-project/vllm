#!/usr/bin/env bash
set -euo pipefail

# Read each PID (one-per-line) into an array
mapfile -t pids < <(
  nvidia-smi \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits
)

if (( ${#pids[@]} == 0 )); then
  echo "✔ No GPU processes found."
else
  echo "The following GPU processes will be terminated:"
  printf '→ %s\n' "${pids[@]}"

  for pid in "${pids[@]}"; do
    if kill -0 "$pid" &>/dev/null; then
      echo "→ Killing PID $pid"
      kill -9 "$pid"
    else
      echo "⚠ PID $pid does not exist or has already exited"
    fi
  done
fi

docker system prune -af --volumes
docker builder prune -af
rm -rf ~/.cache/huggingface

echo "Done."
