#!/bin/bash
#SBATCH --job-name=hisparse-capture-spike
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=15
#SBATCH --output=/root/fares/vllm-0240-coord/logs/capture_spike.log
set -euo pipefail
cd /root/fares/prime-rl-pd-test
source .venv/bin/activate
python /root/fares/vllm-0240-coord/capture_spike.py
echo "=== CAPTURE SPIKE DONE rc=$? ==="
