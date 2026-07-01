#!/bin/bash
#SBATCH --job-name=hisparse-coord-build
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=150
#SBATCH --output=/root/fares/vllm-0240-coord/logs/build_test_coord.log
set -euo pipefail
set -x
ROOT=/root/fares/vllm-0240-coord
VENV=$ROOT/.venv
export UV_CACHE_DIR=/root/.cache/uv
export TMPDIR=/tmp/hscoord.$$
mkdir -p "$TMPDIR" "$ROOT/logs"
trap 'rm -rf "$TMPDIR"' EXIT
nvcc --version | tail -2

rm -rf "$VENV"
uv venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
uv pip install torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install pip "cmake>=3.26.1" ninja "packaging>=24.2" "setuptools>=77,<81" \
    "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel jinja2 pytest tblib \
    flashinfer-python==0.6.12
export RUSTUP_HOME="$TMPDIR/rustup" CARGO_HOME="$TMPDIR/cargo"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable
export PATH="$CARGO_HOME/bin:$PATH"; cargo --version

cd "$ROOT"
MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST=9.0a CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    uv pip install -e . --no-build-isolation

python - <<'PY'
import torch, vllm
try:
    import vllm._C_stable_libtorch  # noqa: F401
except ImportError:
    import vllm._C  # noqa: F401
ops = torch.ops._C_cache_ops
assert hasattr(ops, "hisparse_swap_in"), "hisparse_swap_in missing"
assert hasattr(ops, "hisparse_gather_plan"), "hisparse_gather_plan NOT registered"
sw = str(ops.hisparse_swap_in.default._schema)
assert "miss_mask" in sw, f"miss_mask not in swap_in schema: {sw}"
assert "num_real_reqs" in sw, "num_real_reqs regressed"
print("OK new ops present:", vllm.__version__)
print("swap_in schema:", sw)
print("gather_plan schema:", str(ops.hisparse_gather_plan.default._schema))
PY

# Cycle 1 gate: existing 13 tests must still pass (miss_mask optional -> no
# behavior change on the existing path; gather_plan new/unused).
python -m pytest tests/v1/attention/test_sparse_mla_backends.py -k hisparse -v 2>&1 \
    | tee "$ROOT/logs/pytest_coord.log"
echo "=== COORD BUILD+TEST DONE rc=$? ==="
