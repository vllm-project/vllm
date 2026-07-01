#!/bin/bash
#SBATCH --job-name=hisparse-plan-once-e2e
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=200
#SBATCH --output=/root/fares/vllm-0240-coord/logs/e2e_plan_once.log
set -euo pipefail
set -x
# Reuse the full-deps prime-rl-pd-test env (deep_gemm, flashinfer, NIXL, etc.),
# swapping in the coord branch's vLLM (recompiled vs this env's torch).
ENV=/root/fares/prime-rl-pd-test
FORK=/root/fares/vllm-0240-coord
export UV_CACHE_DIR=/root/.cache/uv
export HF_HOME=/root/.cache/huggingface
export TMPDIR=/tmp/e2epo.$$
mkdir -p "$TMPDIR" "$FORK/logs"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$ENV"
source .venv/bin/activate
export RUSTUP_HOME="$TMPDIR/rustup" CARGO_HOME="$TMPDIR/cargo"
command -v cargo >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable
export PATH="$CARGO_HOME/bin:$PATH"

echo "=== recompile coord-branch vLLM into the pd-test env (clean vs its torch) ==="
rm -rf "$FORK"/build "$FORK"/vllm/*.abi3.so "$FORK"/vllm/*.so 2>/dev/null || true
MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST=9.0a CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    uv pip install --reinstall --no-deps -e "$FORK" --no-build-isolation
python -c "import torch,vllm; assert hasattr(torch.ops._C_cache_ops,'hisparse_gather_plan'); print('coord vllm', vllm.__version__, 'gather_plan present')"

echo "=== GLM-5.2-FP8 + HiSparse plan-once coherence (eager) ==="
export UV_NO_SYNC=1
python "$FORK/validate_plan_once.py"
echo "=== E2E PLAN-ONCE DONE rc=$? ==="
