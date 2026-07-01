#!/bin/bash
#SBATCH --job-name=hisparse-overlap-mb
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20
#SBATCH --output=/root/fares/vllm-0240-coord/logs/overlap_microbench.log
set -euo pipefail
cd /root/fares/vllm-0240-coord
source .venv/bin/activate
python -c "import torch, vllm; assert hasattr(torch.ops._C_cache_ops,'hisparse_gather_plan'); print('ops ok', vllm.__version__)"
python overlap_microbench.py
echo "=== OVERLAP MICROBENCH DONE rc=$? ==="
