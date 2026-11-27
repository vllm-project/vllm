#!/usr/bin/env bash
# Install the optional KV connectors into the release image. Run by the
# vllm-openai-base stage of docker/Dockerfile when INSTALL_KV_CONNECTORS=true.
#
# Usage: install_kv_connectors.sh <requirements_dir>
#   requirements_dir  holds kv_connectors.txt and kv_connectors_cu12.txt
# Env: CUDA_VERSION (e.g. 13.2.1)
set -eu

requirements_dir=$1
CUDA_MAJOR="${CUDA_VERSION%%.*}"
CUDA_VERSION_DASH=$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr '.' '-')
# lmcache requires explicit specifying CUDA_HOME.
# libcublas-dev is already installed in vllm-runtime-base; do not add it here
# or the fallback's apt-get purge would remove it from the runtime image.
export CUDA_HOME=/usr/local/cuda
BUILD_PKGS="libcusparse-dev-${CUDA_VERSION_DASH} \
            libcusolver-dev-${CUDA_VERSION_DASH}"

KV_REQUIREMENTS="${requirements_dir}/kv_connectors.txt"
if [ "${CUDA_MAJOR}" = "12" ]; then
    KV_REQUIREMENTS="${requirements_dir}/kv_connectors_cu12.txt"
fi

# Preserve the stack used to compile vLLM when resolving optional dependencies.
uv pip freeze --system > /tmp/kv-installed.txt
grep -E '^(torch|torchaudio|torchvision|triton|cuda-bindings|cuda-python|cuda-toolkit)==' \
    /tmp/kv-installed.txt > /tmp/kv-constraints.txt
# shellcheck disable=SC2086
uv pip install --system -c /tmp/kv-constraints.txt -r "${KV_REQUIREMENTS}" --no-build || (
    # if the above fails, install from source
    apt-get update -y &&
    apt-get install -y --no-install-recommends --allow-change-held-packages ${BUILD_PKGS} &&
    uv pip install --system -c /tmp/kv-constraints.txt -r "${KV_REQUIREMENTS}" --no-build-isolation &&
    apt-get purge -y ${BUILD_PKGS} &&
    # clean up -dev packages, keep runtime libraries
    rm -rf /var/lib/apt/lists/*
)

# The default mooncake wheel links libcudart.so.12, so on CUDA 13 swap in the
# cuda13 variant. Uninstall first; both ship the `mooncake` package.
# The CI image does the same via requirements/test/kv_connectors_cuda.in.
MOONCAKE_VERSION=$(python3 -c "import importlib.metadata as m; print(m.version('mooncake-transfer-engine'))" 2>/dev/null || true)
if [ "${CUDA_MAJOR}" = "13" ] && [ -n "${MOONCAKE_VERSION}" ]; then
    uv pip uninstall --system mooncake-transfer-engine 2>/dev/null || true
    uv pip install --system -c /tmp/kv-constraints.txt "mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
fi
rm -f /tmp/kv-installed.txt /tmp/kv-constraints.txt
