#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ROCm equivalent of python_only_compile.sh.
#
# Goal: verify that a user without any C/C++ compiler can install and import
# vLLM from a pre-built ROCm wheel (i.e., all HIP kernel .so files are already
# compiled into the wheel — no recompilation is triggered at install time).
#
# This differs from the CUDA version in one key way: there is no
# wheels.vllm.ai equivalent for ROCm, so we reinstall from the wheel that was
# baked into the test image at /opt/vllm-wheels/ during the Docker build
# (COPY --from=export_vllm /*.whl /opt/vllm-wheels/ in Dockerfile.rocm).

set -e

WHEEL_DIR="/opt/vllm-wheels"

echo "=== ROCm Python-only Installation Test ==="
echo "Verifies vLLM is installable and importable without a C++ compiler."
echo ""

# Confirm the wheel is present in the image
if ! ls "${WHEEL_DIR}"/*.whl &>/dev/null; then
    echo "ERROR: No wheel found at ${WHEEL_DIR}/*.whl"
    echo "The Dockerfile.rocm test stage must have COPY --from=export_vllm /*.whl /opt/vllm-wheels/"
    exit 1
fi

WHEEL_PATH=$(ls "${WHEEL_DIR}"/*.whl | head -1)
echo "Found wheel: ${WHEEL_PATH}"

cd /vllm-workspace/

# Remove any preinstalled editable/package copy before testing the wheel.
# Keep the workspace source hidden under src/ so imports cannot accidentally
# resolve to /vllm-workspace/vllm instead of the installed wheel.
pip3 uninstall -y vllm

echo ""
echo "=== Removing C/C++ compilers ==="
apt-get remove --purge build-essential -y
apt-get autoremove -y
echo "Compilers removed. Verifying cc/g++ are gone:"
! command -v cc  && echo "  cc:  not found (expected)"
! command -v g++ && echo "  g++: not found (expected)"

echo ""
echo "=== Installing vLLM from pre-built wheel (no compiler) ==="
echo "Wheel: ${WHEEL_PATH}"
# --no-build-isolation + --no-deps: install exactly the wheel, no setup.py
# compilation triggered; HIP .so files are already inside the wheel.
pip3 install --no-build-isolation --no-deps "${WHEEL_PATH}"

echo ""
echo "=== Verifying no CUDA/NVIDIA Python packages were pulled in ==="
unexpected_cuda_pkgs=$(pip3 list --format=freeze | grep -E '^(nvidia-|cuda-)' || true)
if [[ -n "${unexpected_cuda_pkgs}" ]]; then
    echo "ERROR: unexpected CUDA/NVIDIA Python packages are installed:"
    echo "${unexpected_cuda_pkgs}"
    exit 1
fi

echo ""
echo "=== Importing vLLM ==="
cd /tmp
python3 - <<'PY'
from pathlib import Path

import vllm

module_path = Path(vllm.__file__).resolve()
print(f"vLLM {vllm.__version__} imported successfully from {module_path}")
if "/vllm-workspace/" in str(module_path):
    raise SystemExit(
        "ERROR: imported vLLM from the checkout instead of the installed wheel"
    )
PY

echo ""
echo "=== ROCm Python-only Installation Test PASSED ==="
