#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 03:00:00
#SBATCH -A cis250224p
#SBATCH --gpus=v100-32:1
#SBATCH --job-name=vllm_bench
#SBATCH --output=vllm_bench.log

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set — export it before running}"
export HF_HOME="/jet/home/rnagaraj/workspace/vllm/hf_cache"
export TRITON_CACHE_DIR="/jet/home/rnagaraj/workspace/vllm/triton_cache"
export XDG_CACHE_HOME="/jet/home/rnagaraj/workspace/vllm/xdg_cache"

# Load required modules (verified on PSC Bridges-2)
source /etc/profile.d/modules.sh
module load cuda/12.4.0
module load gcc/10.2.0

cd ~/workspace/vllm
source .venv/bin/activate

# Reinstall torch pinned to the loaded CUDA module
pip install "torch==2.5.1" "numpy<2" setuptools wheel \
    --index-url https://download.pytorch.org/whl/cu124 -q

# Re-build vLLM extensions
pip install -e ".[dev]" -q

echo "=========================================="
echo "Starting KV Cache Tiering Benchmark Suite"
echo "Model: meta-llama/Llama-3.2-1B-Instruct"
echo "GPU Memory Utilization: 0.30 (forcing real eviction pressure)"
echo "Policies: lru, attention, hybrid"
echo "Num prompts: 200 x 512 max tokens (synthetic)"
echo "=========================================="

mkdir -p ~/workspace/vllm/benchmark_results

python3 -m kv_cache_tiering.benchmarks.benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --policies lru attention hybrid \
    --dataset synthetic \
    --num-prompts 200 \
    --max-tokens 512 \
    --gpu-mem-util 0.30 \
    --cpu-bytes 4000000000 \
    --output ~/workspace/vllm/benchmark_results/results_$(date +%Y%m%d_%H%M%S).json

echo "Benchmark complete! Results saved to ~/workspace/vllm/benchmark_results/"
ls -lh ~/workspace/vllm/benchmark_results/
