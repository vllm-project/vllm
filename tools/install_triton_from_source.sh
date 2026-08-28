#!/usr/bin/env bash
set -euxo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <wheel-directory> <checksum-file>" >&2
    exit 1
fi

TRITON_DIST="$1"
TRITON_CHECKSUM="$2"

test -d "${TRITON_DIST}"
test -f "${TRITON_CHECKSUM}"
(cd "${TRITON_DIST}" && sha256sum -c "${TRITON_CHECKSUM}")

shopt -s nullglob
TRITON_WHEELS=("${TRITON_DIST}"/triton-*.whl)
if [[ ${#TRITON_WHEELS[@]} -ne 1 ]]; then
    echo "Expected exactly one Triton wheel in ${TRITON_DIST}" >&2
    exit 1
fi

uv pip install --system --reinstall --no-deps "${TRITON_WHEELS[0]}"

if [[ -n "${TRITON_SYSTEM_PTXAS_PATH:-}" \
    || -n "${TRITON_EXPECTED_PTXAS_VERSION:-}" ]]; then
    : "${TRITON_SYSTEM_PTXAS_PATH:?TRITON_SYSTEM_PTXAS_PATH must be set}"
    : "${TRITON_EXPECTED_PTXAS_VERSION:?TRITON_EXPECTED_PTXAS_VERSION must be set}"
    PTXAS_PATH="${TRITON_SYSTEM_PTXAS_PATH}"
    test -x "${PTXAS_PATH}"
    TRITON_NVIDIA_BIN=$(python3 -c \
        'from pathlib import Path; import triton; print(Path(triton.__file__).parent / "backends" / "nvidia" / "bin")')
    mkdir -p "${TRITON_NVIDIA_BIN}"
    ln -sfn "${PTXAS_PATH}" \
        "${TRITON_NVIDIA_BIN}/ptxas-blackwell"

    env -u TRITON_PTXAS_BLACKWELL_PATH \
        TRITON_EXPECTED_PTXAS_PATH="${PTXAS_PATH}" \
        python3 -c \
            'import os; from triton.backends.nvidia.compiler import get_ptxas; tool = get_ptxas(107); assert tool.version == os.environ["TRITON_EXPECTED_PTXAS_VERSION"]; assert os.path.samefile(tool.path, os.environ["TRITON_EXPECTED_PTXAS_PATH"])'
fi
