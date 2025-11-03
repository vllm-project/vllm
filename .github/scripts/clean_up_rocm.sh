#!/bin/bash
set -euo pipefail

echo
echo "==== ROCm GPU Status (Before Cleanup) ===="
rocm-smi

echo
echo "==== ROCm GPU Processes (Before Cleanup) ===="
rocm-smi --showpids

echo
echo "==== Cleaning up ROCm GPU Processes... ===="
# Kill all processes using the ROCm GPUs by extracting their PIDs from rocm-smi output
PIDS=$(rocm-smi --showpids | awk '{if($1 ~ /^[0-9]+$/) print $1}')
if [ -n "$PIDS" ]; then
    echo "Killing the following PIDs: $PIDS"
    echo "$PIDS" | xargs -r sudo kill -9 || true
else
    echo "No ROCm GPU processes found to clean up."
fi

echo
echo "==== ROCm GPU Status (After Cleanup) ===="
rocm-smi

echo
echo "==== ROCm GPU Processes (After Cleanup) ===="
rocm-smi --showpids
