#!/bin/bash
# Slurm env check for the HiSparse P/D rework branch.
#
# Verifies the shared-filesystem venv works on a b200 compute node and runs
# the test files the rework plan requires to stay green (first GPU run of the
# rework — pytest was never run for slice A).
#
#   sbatch benchmarks/hisparse_pd/slurm/envcheck.sbatch
#
#SBATCH --job-name=hisparse-envcheck
#SBATCH --partition=b200
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/vast/home/nicolo.lucchesi/upstream/vllm
#SBATCH --output=/mnt/vast/home/nicolo.lucchesi/upstream/vllm/slurm-hisparse-envcheck-%j.out

set -euo pipefail

GIT_ROOT=/mnt/vast/home/nicolo.lucchesi/upstream/vllm

# NCCL's HPC-X net plugin drags a second UCX tree next to the one bundled with
# the nixl wheel; UCX then dies with "Invalid value for SCOPY_" at TP>=2.
export NCCL_NET_PLUGIN=none
export HF_HOME=${HF_HOME:-/mnt/vast/hf-shared-cache}

source "${GIT_ROOT}/.venv/bin/activate"
cd "${GIT_ROOT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'EOF'
import torch
import vllm

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| devices:", torch.cuda.device_count())
print("vllm", vllm.__version__)
import nixl
print("nixl import ok")
EOF

echo "=== unit: nixl desc geometry ==="
pytest tests/v1/kv_connector/unit/test_nixl_desc_geometry.py -x --no-header -q

echo "=== unit: hisparse connector ==="
pytest tests/v1/kv_connector/unit/test_hisparse_connector.py -x --no-header -q

echo "=== unit: nixl connector ==="
pytest tests/v1/kv_connector/unit/test_nixl_connector.py -x --no-header -q

echo "=== unit: nixl connector hma ==="
pytest tests/v1/kv_connector/unit/test_nixl_connector_hma.py --no-header -q

echo "=== e2e: hisparse standalone (spill + prefix restore) ==="
pytest tests/v1/e2e/general/test_hisparse.py -x --no-header -q

echo "=== ENV CHECK PASSED ==="
