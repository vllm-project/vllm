#!/usr/bin/env bash
set -euxo pipefail

: "${TARGETARCH:?TARGETARCH must be set}"
: "${TRITON_INSTALL_FROM_SOURCE_REPO:?TRITON_INSTALL_FROM_SOURCE_REPO must be set}"

TRITON_REVISION="${TRITON_INSTALL_FROM_SOURCE_REVISION:-main}"
TRITON_WORKSPACE="${TRITON_WORKSPACE:-/tmp/triton_from_source_workspace}"
TRITON_SOURCE="${TRITON_WORKSPACE}/src"
TRITON_DIST="${TRITON_WORKSPACE}/dist"
TRITON_BUILD_JOBS="${MAX_JOBS:-$(nproc)}"

case "${TARGETARCH}" in
    amd64) TRITON_LLVM_SYSTEM_SUFFIX=almalinux-x64 ;;
    arm64) TRITON_LLVM_SYSTEM_SUFFIX=almalinux-arm64 ;;
    *)
        echo "Unsupported Triton build architecture: ${TARGETARCH}" >&2
        exit 1
        ;;
esac
export TRITON_LLVM_SYSTEM_SUFFIX

mkdir -p "${TRITON_DIST}"
git init -q "${TRITON_SOURCE}"
git -C "${TRITON_SOURCE}" remote add origin \
    "${TRITON_INSTALL_FROM_SOURCE_REPO}"
git -C "${TRITON_SOURCE}" fetch --depth=1 origin "${TRITON_REVISION}"
git -C "${TRITON_SOURCE}" checkout -q --detach FETCH_HEAD
git -C "${TRITON_SOURCE}" rev-parse HEAD

uv pip install --python /opt/venv/bin/python3 \
    -r "${TRITON_SOURCE}/python/requirements.txt"

TRITON_BUILD_WITH_CCACHE=true \
MAX_JOBS="${TRITON_BUILD_JOBS}" \
    uv build --python /opt/venv/bin/python3 \
        --wheel \
        --no-build-isolation \
        --out-dir "${TRITON_DIST}" \
        "${TRITON_SOURCE}"

(cd "${TRITON_DIST}" && sha256sum triton-*.whl > wheel.sha256)
rm -rf "${TRITON_SOURCE}"
