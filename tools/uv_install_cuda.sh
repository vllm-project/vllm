#!/usr/bin/env bash
# `uv pip install --system` with the index options every runtime stage of
# docker/Dockerfile uses, so the release and CI test images resolve packages
# the same way.
#
# Usage: uv_install_cuda.sh <torch_nightly> <what>...
#   torch_nightly  "1" first installs the torch libs pinned in
#                  ./torch_lib_versions.txt from the nightly index, then
#                  resolves against that index. Anything else uses the
#                  release index.
#   what           passed to `uv pip install`: wheel files, or -r <file>
# Env: CUDA_VERSION (e.g. 13.2.1), PYTORCH_CUDA_INDEX_BASE_URL
#
# The release path searches PyPI first and the CUDA index second: uv ranks
# --extra-index-url above --index-url. PyTorch is already installed from the
# CUDA index by then (vllm-runtime-base); searching that index first here would
# take its older copies of common packages, such as urllib3 1.26.
set -euo pipefail

torch_nightly=$1
shift
cuda_tag="cu$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr -d '.')"

if [ "${torch_nightly}" = "1" ]; then
    nightly_index="${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/${cuda_tag}"
    uv pip install --system -r torch_lib_versions.txt --pre \
        --index-url "${nightly_index}"
    uv pip install --system "$@" --verbose \
        --extra-index-url "${nightly_index}"
else
    uv pip install --system "$@" --verbose \
        --index-url "${PYTORCH_CUDA_INDEX_BASE_URL}/${cuda_tag}" \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match
fi
