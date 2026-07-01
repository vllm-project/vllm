#!/bin/bash
#SBATCH --job-name=hisparse-coord-test
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=20
#SBATCH --output=/root/fares/vllm-0240-coord/logs/pytest_cycle2.log
set -euo pipefail
cd /root/fares/vllm-0240-coord
source .venv/bin/activate
python -m pytest tests/v1/attention/test_sparse_mla_backends.py -k hisparse -v 2>&1 | tail -40
echo "=== CYCLE2 PYTEST DONE rc=$? ==="
