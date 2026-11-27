#!/usr/bin/env bash
# `uv pip install --system` with the index options every runtime stage of
# docker/Dockerfile uses, so the release and CI test images resolve packages
# the same way.
#
# Usage: uv_install_cuda.sh <torch_nightly> <cuda_version> <index_base_url> <what>...
#   torch_nightly   "1" first installs the torch libs pinned in
#                   ./torch_lib_versions.txt from the nightly index, then
#                   resolves against that index. Anything else uses the
#                   release index.
#   cuda_version    e.g. 13.2.1
#   index_base_url  PyTorch wheel index, e.g. https://download.pytorch.org/whl
#   what            passed to `uv pip install`: wheel files, or -r <file>
#
# The release path searches PyPI first and the CUDA index second: uv ranks
# --extra-index-url above --index-url. PyTorch is already installed from the
# CUDA index by then (vllm-runtime-base); searching that index first here would
# take its older copies of common packages, such as urllib3 1.26.
set -euo pipefail

torch_nightly=$1
cuda_version=$2
index_base_url=$3
shift 3
cuda_tag="cu$(echo "${cuda_version}" | cut -d. -f1,2 | tr -d '.')"

if [ "${torch_nightly}" = "1" ]; then
    # uv only warns on an empty -r file, which would leave torch unpinned.
    test -s torch_lib_versions.txt || {
        echo "torch_lib_versions.txt is missing or empty; nightly torch would be unpinned" >&2
        exit 1
    }
    nightly_index="${index_base_url}/nightly/${cuda_tag}"
    uv pip install --system -r torch_lib_versions.txt --pre \
        --index-url "${nightly_index}"
    uv pip install --system "$@" --verbose \
        --extra-index-url "${nightly_index}"
else
    uv pip install --system "$@" --verbose \
        --index-url "${index_base_url}/${cuda_tag}" \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match
fi
