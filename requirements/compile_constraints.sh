#!/bin/bash
# Script to regenerate all constraint files from requirements files
# This ensures deterministic, pinned versions for all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS_DIR="${SCRIPT_DIR}/constraints"

echo "Regenerating constraint files..."

# Python version to use for compilation
PYTHON_VERSION="3.12"

# Create constraints directory if it doesn't exist
mkdir -p "${CONSTRAINTS_DIR}"

# Base requirements files (no -r dependencies)
echo "Compiling base requirements..."
uv pip compile "${SCRIPT_DIR}/lint.txt" -o "${CONSTRAINTS_DIR}/lint.txt" --python-version ${PYTHON_VERSION}
uv pip compile "${SCRIPT_DIR}/common.txt" -o "${CONSTRAINTS_DIR}/common.txt" --python-version ${PYTHON_VERSION}
uv pip compile "${SCRIPT_DIR}/build.txt" -o "${CONSTRAINTS_DIR}/build.txt" --python-version ${PYTHON_VERSION}
uv pip compile "${SCRIPT_DIR}/docs.txt" -o "${CONSTRAINTS_DIR}/docs.txt" --python-version ${PYTHON_VERSION}
uv pip compile "${SCRIPT_DIR}/kv_connectors.txt" -o "${CONSTRAINTS_DIR}/kv_connectors.txt" --python-version ${PYTHON_VERSION}
uv pip compile "${SCRIPT_DIR}/nightly_torch_test.txt" -o "${CONSTRAINTS_DIR}/nightly_torch_test.txt" --python-version ${PYTHON_VERSION}

# test.txt is already compiled, just copy it
cp "${SCRIPT_DIR}/test.txt" "${CONSTRAINTS_DIR}/test.txt"

# CUDA requirements
echo "Compiling CUDA requirements..."
uv pip compile "${SCRIPT_DIR}/cuda.txt" -o "${CONSTRAINTS_DIR}/cuda.txt" \
    --python-version ${PYTHON_VERSION} \
    --python-platform x86_64-manylinux_2_28

# ROCm requirements (need unsafe-best-match for custom indexes)
echo "Compiling ROCm requirements..."
uv pip compile "${SCRIPT_DIR}/rocm.txt" -o "${CONSTRAINTS_DIR}/rocm.txt" \
    --python-version ${PYTHON_VERSION}

uv pip compile "${SCRIPT_DIR}/rocm-build.txt" -o "${CONSTRAINTS_DIR}/rocm-build.txt" \
    --python-version ${PYTHON_VERSION} \
    --index-strategy unsafe-best-match

uv pip compile "${SCRIPT_DIR}/rocm-test.txt" -o "${CONSTRAINTS_DIR}/rocm-test.txt" \
    --python-version ${PYTHON_VERSION}

# TPU requirements
echo "Compiling TPU requirements..."
uv pip compile "${SCRIPT_DIR}/tpu.txt" -o "${CONSTRAINTS_DIR}/tpu.txt" \
    --python-version ${PYTHON_VERSION}

# XPU requirements (need unsafe-best-match for custom indexes)
echo "Compiling XPU requirements..."
uv pip compile "${SCRIPT_DIR}/xpu.txt" -o "${CONSTRAINTS_DIR}/xpu.txt" \
    --python-version ${PYTHON_VERSION} \
    --index-strategy unsafe-best-match

# Note: cpu.txt, cpu-build.txt, and dev.txt cannot be compiled to constraints
# because they reference platform-specific torch builds (+cpu, +cu129 suffixes)
# that are not available through standard resolution. For these files, use
# the requirements file itself as a constraint since they already specify exact versions.
echo ""
echo "Note: Skipping cpu.txt, cpu-build.txt, and dev.txt (use requirements files directly as constraints)"

echo ""
echo "Constraint files regenerated successfully in ${CONSTRAINTS_DIR}/"
echo ""
echo "Usage: pip install -r requirements/<name>.txt -c requirements/constraints/<name>.txt"
