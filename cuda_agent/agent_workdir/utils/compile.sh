#!/usr/bin/env bash
# utils/compile.sh
# Shell wrapper: compiles CUDA kernels and reports elapsed time.
# Do NOT modify this file.
#
# Usage:
#   TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh

set -euo pipefail

START_TIME=$(date +%s.%N)

python3 -m utils.compile
EXIT_CODE=$?

END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {printf \"%.2f\", $END_TIME - $START_TIME}")
echo "[TIME] Compilation took ${ELAPSED}s"

exit "$EXIT_CODE"
