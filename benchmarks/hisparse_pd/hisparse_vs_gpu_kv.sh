#!/usr/bin/env bash
# A/B benchmark: plain GPU-KV decode vs HiSparse host-pool decode on one
# fixed P/D topology and traffic grid.
#
# This is the comparison shape of #46326 (identical topology and traffic,
# decode side differs) run over #53781's ISL/OSL points. Each arm is a full
# pd_bench.sh invocation (engines restart between arms; the decode-side
# landing policy cannot be toggled at runtime).
#
# Usage: export the same env pd_bench.sh takes (MODEL is required), plus:
#   HOST_POOL_GIB       decode host pool per rank for the hisparse arm
#   DEVICE_BUFFER_SIZE  optional hot rows/request
#   ARMS                arms to run (default "gpu-kv hisparse")
#   RESULTS_ROOT        output root (default bench_results/hisparse_pd_ab_<ts>)
#   PREFILL_HISPARSE=1  only when baselining against the pre-rework branch,
#                        where P also required hisparse_config
#
# Example (single node, 8 GPUs):
#   MODEL=zai-org/GLM-5.2-FP8 P_TP=4 D_TP=4 HOST_POOL_GIB=64 \
#     ./benchmarks/hisparse_pd/hisparse_vs_gpu_kv.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

ARMS=${ARMS:-"gpu-kv hisparse"}
HOST_POOL_GIB=${HOST_POOL_GIB:-32}
DEVICE_BUFFER_SIZE=${DEVICE_BUFFER_SIZE:-}
PREFILL_HISPARSE=${PREFILL_HISPARSE:-0}
RESULTS_ROOT=${RESULTS_ROOT:-bench_results/hisparse_pd_ab_$(date +%Y%m%d_%H%M%S)}

HISPARSE_AC='{"hisparse_config":{"host_pool_gib":'"$HOST_POOL_GIB"
if [[ -n "$DEVICE_BUFFER_SIZE" ]]; then
    HISPARSE_AC="$HISPARSE_AC"',"device_buffer_size":'"$DEVICE_BUFFER_SIZE"
fi
HISPARSE_AC="$HISPARSE_AC"'}}'

if [[ "$PREFILL_HISPARSE" == "1" ]]; then
    export PREFILL_ATTENTION_CONFIG="$HISPARSE_AC"
fi
# pd_bench.sh auto-builds a hisparse config from HOST_POOL_GIB when
# DECODE_ATTENTION_CONFIG is unset; drop both so each arm is fully explicit.
unset HOST_POOL_GIB DEVICE_BUFFER_SIZE PREFILL_HISPARSE

mkdir -p "$RESULTS_ROOT"

for ARM in $ARMS; do
    case "$ARM" in
        gpu-kv)
            export DECODE_ATTENTION_CONFIG=""
            ;;
        hisparse)
            export DECODE_ATTENTION_CONFIG="$HISPARSE_AC"
            ;;
        *)
            echo "ERROR: unknown arm '$ARM' (expected gpu-kv or hisparse)"
            exit 1
            ;;
    esac

    echo ""
    echo "###########################################################"
    echo "# Arm: $ARM"
    echo "# decode attention config: ${DECODE_ATTENTION_CONFIG:-<none>}"
    echo "###########################################################"
    ARM_TAG="$ARM" OUTPUT_DIR="$RESULTS_ROOT/$ARM" "$SCRIPT_DIR/pd_bench.sh"
done

echo ""
echo "A/B results in: $RESULTS_ROOT"
"$SCRIPT_DIR/summarize_ab.py" --root "$RESULTS_ROOT"
