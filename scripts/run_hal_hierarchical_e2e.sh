#!/bin/bash
# Hal XPU e2e for hierarchical expert staging (Mixtral-8x22B Q4 default).
set -euo pipefail
cd "${VLLM_ROOT:-/work/vllm}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HIER_MODEL="${HIER_MODEL:-/tank/nas/models/Mixtral-8x22B-Instruct-v0.1-AWQ}"
exec python3 scripts/run_hal_hierarchical_e2e.py
