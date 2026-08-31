#!/bin/bash
# Slurm driver for the HiSparse P/D benchmarks (see ../README.md).
#
# Single 8-GPU node, one P engine (tp=P_TP) + one D engine (tp=D_TP).
# BENCH_MODE selects the driver:
#   pd  -> benchmarks/hisparse_pd/pd_bench.sh        (one arm, pd_bench env)
#   ab  -> benchmarks/hisparse_pd/hisparse_vs_gpu_kv.sh (gpu-kv + hisparse)
#
# All pd_bench.sh knobs pass through the submission environment, e.g.:
#   BENCH_MODE=pd ISL_OSL_PAIRS="8192:512" CONCURRENCIES="4" NUM_PROMPTS=8 \
#     sbatch benchmarks/hisparse_pd/slurm/bench.sbatch
#
#SBATCH --job-name=hisparse-pd-bench
#SBATCH --partition=b200
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=10:00:00
#SBATCH --chdir=/mnt/vast/home/nicolo.lucchesi/upstream/vllm
#SBATCH --output=/mnt/vast/home/nicolo.lucchesi/upstream/vllm/slurm-hisparse-bench-%j.out

set -euo pipefail

GIT_ROOT=/mnt/vast/home/nicolo.lucchesi/upstream/vllm

# See envcheck.sbatch for the NCCL/HPC-X vs nixl-wheel UCX conflict.
export NCCL_NET_PLUGIN=none
export HF_HOME=${HF_HOME:-/mnt/vast/hf-shared-cache}

source "${GIT_ROOT}/.venv/bin/activate"
cd "${GIT_ROOT}"

# Benchmark defaults; any of these can be overridden from the submission env.
export MODEL=${MODEL:-nvidia/GLM-5.2-NVFP4}
export P_TP=${P_TP:-4}
export D_TP=${D_TP:-4}
export HOST_POOL_GIB=${HOST_POOL_GIB:-128}
export MAX_NUM_SEQS_D=${MAX_NUM_SEQS_D:-64}
BENCH_MODE=${BENCH_MODE:-ab}

echo "=== bench driver: mode=$BENCH_MODE ==="
echo "    model=$MODEL  P_TP=$P_TP  D_TP=$D_TP  host_pool_gib=$HOST_POOL_GIB"
echo "    grid: pairs=[${ISL_OSL_PAIRS:-default}] concurrencies=[${CONCURRENCIES:-default}] prompts=${NUM_PROMPTS:-100}"

case "$BENCH_MODE" in
    pd) exec benchmarks/hisparse_pd/pd_bench.sh ;;
    ab) exec benchmarks/hisparse_pd/hisparse_vs_gpu_kv.sh ;;
    *)  echo "ERROR: unknown BENCH_MODE '$BENCH_MODE' (expected pd or ab)"; exit 1 ;;
esac
