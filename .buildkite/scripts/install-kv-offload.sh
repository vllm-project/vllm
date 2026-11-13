#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

if [ -z "${CUDA_VERSION:-}" ]; then
    exit 0
fi

# TODO: Use the PyPI wheel once it is published.
uv pip install --system \
    "kvcr @ git+https://github.com/ai-dynamo/kvcr.git@5ce23b1527725cf623a3fc7feb94dc6d3d5fafaa"

# Keep only the NIXL wheel matching the CI image's CUDA runtime.
NIXL_VERSION=$(uv pip show --system nixl | sed -n 's/^Version: //p')
uv pip uninstall --system nixl-cu12 nixl-cu13
uv pip install --system --no-deps "nixl-cu${CUDA_VERSION%%.*}==${NIXL_VERSION}"
