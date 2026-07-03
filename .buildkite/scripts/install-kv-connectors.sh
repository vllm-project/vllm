#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

# do-not-merge: resolve the NIXL 1.3.1 RC2 prerelease from TestPyPI for CI.
export UV_PRERELEASE=allow
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:+${UV_EXTRA_INDEX_URL} }https://test.pypi.org/simple/"

if python3 -c "import torch; raise SystemExit(0 if torch.version.hip is not None else 1)"; then
    uv pip install --system -r /vllm-workspace/requirements/kv_connectors_rocm.txt
    exit 0
fi

REQUIREMENTS_FILE="${KV_CONNECTORS_REQUIREMENTS:-/vllm-workspace/requirements/kv_connectors.txt}"

uv pip install --system -r "${REQUIREMENTS_FILE}"

NIXL_METADATA=$(python3 - <<'PY'
import importlib.metadata as metadata

import torch

cuda_version = torch.version.cuda
if cuda_version is None:
    raise SystemExit("torch.version.cuda is not set")

print(cuda_version.split(".", 1)[0], metadata.version("nixl"))
PY
)
read -r CUDA_MAJOR NIXL_VERSION <<<"${NIXL_METADATA}"

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
