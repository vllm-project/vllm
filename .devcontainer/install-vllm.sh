#!/bin/bash
# Build and install vLLM in editable mode for AMD RX 6900 XT (gfx1030 / RDNA2)
#
# Run this once after the devcontainer starts. It compiles HIP kernels and
# will take 30-60+ minutes depending on MAX_JOBS and CPU speed.
#
# NOTE: gfx1030 is not in vLLM's official supported arch list (they target
# enterprise MI-series and RDNA3+). Most things work via HSA_OVERRIDE_GFX_VERSION
# and PYTORCH_ROCM_ARCH=gfx1030, but some custom kernels may fall back to
# slower paths or require patches.

set -e

cd /workspace

export PYTORCH_ROCM_ARCH=gfx1030
export MAX_JOBS=${MAX_JOBS:-8}

echo "=== Installing vLLM for ROCm / RX 6900 XT (gfx1030) ==="
echo "    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
echo "    MAX_JOBS=${MAX_JOBS}"
echo ""
echo "This compiles HIP kernels. Expect 30-60+ minutes..."
echo ""

pip install -e .

echo ""
echo "=== Done! Test with: ==="
echo "    python -c \"import vllm; print(vllm.__version__)\""
