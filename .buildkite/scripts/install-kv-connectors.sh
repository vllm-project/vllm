#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

if python3 -c "import torch; raise SystemExit(0 if torch.version.hip is not None else 1)"; then
    uv pip install --system -r /vllm-workspace/requirements/kv_connectors_rocm.txt
    exit 0
fi

REQUIREMENTS_FILE="${KV_CONNECTORS_REQUIREMENTS:-/vllm-workspace/requirements/kv_connectors.txt}"

# lmcache wheels on PyPI are compiled against a single pinned torch ABI. When
# the CI image ships a different torch, the wheel install succeeds but
# lmcache.c_ops fails to load at runtime (C++ ABI mismatch). Build lmcache
# from source against the torch in this image instead; --no-build-isolation
# makes the PEP 517 build reuse the installed torch.
grep -v '^lmcache' "${REQUIREMENTS_FILE}" > /tmp/kv_connectors_rest.txt
uv pip install --system -r /tmp/kv_connectors_rest.txt
uv pip install --system 'lmcache >= 0.3.9' --no-binary lmcache --no-build-isolation-package lmcache

KV_METADATA=$(python3 - <<'PY'
import importlib.metadata as metadata

import torch

cuda_version = torch.version.cuda
if cuda_version is None:
    raise SystemExit("torch.version.cuda is not set")

try:
    mooncake_version = metadata.version("mooncake-transfer-engine")
except metadata.PackageNotFoundError:
    mooncake_version = ""

print(cuda_version.split(".", 1)[0], metadata.version("nixl"), mooncake_version)
PY
)
read -r CUDA_MAJOR NIXL_VERSION MOONCAKE_VERSION <<<"${KV_METADATA}"
MOONCAKE_VERSION="${MOONCAKE_VERSION:-}"

# nixl>=1.1.0 can install multiple CUDA wheel variants. Keep only the variant
# matching this CI image so nixl_ep_cpp links against the available libcudart.
uv pip uninstall --system nixl-cu12 nixl-cu13 2>/dev/null || true
uv pip install --system --no-deps "nixl-cu${CUDA_MAJOR}==${NIXL_VERSION}"

python3 - <<'PY'
import importlib.metadata as metadata

for package_name in ("nixl", "nixl-cu12", "nixl-cu13"):
    try:
        version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        version = "not installed"
    print(f"{package_name}: {version}")
PY

# The default mooncake-transfer-engine PyPI wheel is built against CUDA 12; its
# engine.so links libcudart.so.12, absent from the CUDA 13 runtime image. On a
# CUDA 13 image, swap it for the cuda13 variant (same version), which links
# libcudart.so.13. Both expose the `mooncake` package, so uninstall the CUDA 12
# build first to avoid a clash.
if [ "${CUDA_MAJOR}" = "13" ] && [ -n "${MOONCAKE_VERSION}" ]; then
    uv pip uninstall --system mooncake-transfer-engine 2>/dev/null || true
    uv pip install --system "mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
fi
