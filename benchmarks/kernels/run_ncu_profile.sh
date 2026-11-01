#!/bin/bash
# NCU profiling wrapper script for MHC fusion kernels.
# Usage: ./run_ncu_profile.sh [kernel_name]
#   kernel_name: all, post_hc_head, post_hc_head_norm, post_mean (default: all)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$VLLM_DIR" || exit 1

export PYTHONPATH="$VLLM_DIR:$PYTHONPATH"

sudo -E /usr/local/cuda/bin/ncu \
    --metrics dram__bytes.sum \
    --target-processes all \
    python3 benchmarks/kernels/profile_mhc_fusions.py --kernel "${1:-all}"
