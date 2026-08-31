#!/bin/bash
# Source build of vLLM for the HiSparse P/D rework branch.
#
# The shared venv was originally installed with VLLM_USE_PRECOMPILED=1, so the
# precompiled extension has no HiSparse kernels (csrc/libtorch_stable/
# hisparse_kernels.cu). This job rebuilds vllm from source inside the CUDA
# 13.3 devel container (the compute nodes only ship CUDA 12.8, which does not
# match the venv's torch 2.13.0+cu130) and installs it editable in place:
# the resulting .so files land in the repo tree on the shared filesystem, so
# every later job picks them up by just sourcing the venv.
#
# NOTE: the image is ubuntu22.04 based (like the compute nodes) so the built
# .so does not require a newer glibc than the host, and CUDA 13.0 to match
# the venv's torch cu130 build. A 24.04-based image produces GLIBC_2.38
# dependencies that fail to load on the native nodes.
#
#   sbatch benchmarks/hisparse_pd/slurm/build.sbatch
#
#SBATCH --job-name=hisparse-build
#SBATCH --partition=b200
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/vast/home/nicolo.lucchesi/upstream/vllm
#SBATCH --output=/mnt/vast/home/nicolo.lucchesi/upstream/vllm/slurm-hisparse-build-%j.out
#SBATCH --container-image=nvidia/cuda:13.0.0-devel-ubuntu22.04
#SBATCH --container-mounts=/mnt/vast:/mnt/vast

set -euo pipefail

REPO=/mnt/vast/home/nicolo.lucchesi/upstream/vllm
HOME_DIR=/mnt/vast/home/nicolo.lucchesi

export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${HOME_DIR}/.local/bin:${HOME_DIR}/.cargo/bin:${PATH}"
# uv needs a writable cache also reachable from the container.
export UV_CACHE_DIR="${HOME_DIR}/.cache/uv"

echo "=== node: $(hostname) | container: $(cat /etc/os-release | grep PRETTY | cut -d'"' -f2) ==="
nvcc --version | tail -1
gcc --version | head -1
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader

cd "${REPO}"
source .venv/bin/activate

# --no-deps:            runtime deps are already pinned in the shared venv;
#                       re-resolving could churn it.
# --no-build-isolation: build against the venv's own torch 2.13.0+cu130
#                       (an isolated env would pull a PyPI default torch
#                       with a different CUDA variant for arch detection).
# VLLM_VERSION_OVERRIDE: the container image has no git, so setuptools-scm
#                       cannot derive the version from the repo; pin it to
#                       the version the venv already reports.
# *_SRC_DIR:            the container has no git-remote-https, so FetchContent
#                       cannot clone deps; point every external project at its
#                       pinned checkout in .deps/ (revisions match the GIT_TAGs
#                       in cmake/external_projects/*.cmake).
export MAX_JOBS=64
export NVCC_THREADS=1
export VLLM_VERSION_OVERRIDE="0.28.1rc1.dev54+gc95d429e8"
export VLLM_CUTLASS_SRC_DIR="${REPO}/.deps/cutlass-src"
export DEEPGEMM_SRC_DIR="${REPO}/.deps/deepgemm-src"
export FMHA_SM100_SRC_DIR="${REPO}/.deps/fmha_sm100-src"
export FLASH_MLA_SRC_DIR="${REPO}/.deps/flashmla-src"
export FLASH_KDA_SRC_DIR="${REPO}/.deps/flashkda-src"
export QUTLASS_SRC_DIR="${REPO}/.deps/qutlass-src"
export TML_FA4_SRC_DIR="${REPO}/.deps/tml_fa4-src"
export VLLM_FLASH_ATTN_SRC_DIR="${REPO}/.deps/vllm-flash-attn-src"
# triton_kernels expects the python package dir itself when overridden.
export TRITON_KERNELS_SRC_DIR="${REPO}/.deps/triton_kernels-src/python/triton_kernels/triton_kernels"
uv pip install -e . --no-deps --no-build-isolation

# The VLLM_FLASH_ATTN_SRC_DIR path installs flash_attn/cute as a plain copy
# WITHOUT the flash_attn.cute -> vllm.vllm_flash_attn.cute import rewrite that
# the FetchContent path applies (cmake/external_projects/vllm_flash_attn.cmake),
# leaving unresolvable top-level `flash_attn` imports. Replicate the rewrite here.
find "${REPO}/vllm/vllm_flash_attn/cute" -name '*.py' \
  -exec sed -i 's/flash_attn\.cute/vllm.vllm_flash_attn.cute/g' {} +
rm -rf "${REPO}/vllm/vllm_flash_attn/cute/__pycache__"

echo "=== verify hisparse kernels landed in the extension ==="
strings "${REPO}/vllm/_C_stable_libtorch.abi3.so" | grep -c hisparse || true
python - <<'EOF'
import torch
import vllm._C_stable_libtorch  # noqa: F401
print(
    "hisparse_swap_in registered:",
    hasattr(torch.ops._C_cache_ops, "hisparse_swap_in"),
)
assert hasattr(torch.ops._C_cache_ops, "hisparse_swap_in"), (
    "hisparse ops still missing after rebuild"
)
EOF

echo "=== BUILD DONE ==="
