#!/bin/bash
#SBATCH --job-name=hisparse-po-capture
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=15
#SBATCH --output=/root/fares/vllm-0240-coord/logs/plan_once_capture.log
set -euo pipefail
cd /root/fares/vllm-0240-coord
source .venv/bin/activate
python plan_once_capture_test.py
echo "=== PO CAPTURE DONE rc=$? ==="
