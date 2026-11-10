#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# KV connector packages are baked into the CI test images from hash-locked
# requirements (requirements/test/kv_connectors_cuda.txt on CUDA, the ROCm
# test lock on ROCm). This script only verifies them; jobs must not install
# packages at runtime.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 - <<'PY'
import importlib.metadata as metadata
import sys

import torch

if torch.version.hip is not None:
    required = ("tblib", "lm-eval")
else:
    cuda_major = torch.version.cuda.split(".", 1)[0]
    if cuda_major != "13":
        sys.exit(
            f"KV connectors are only baked into the CUDA 13 CI image, got CUDA {cuda_major}"
        )
    required = (
        "lmcache",
        "nixl",
        "nixl-cu13",
        "mooncake-transfer-engine-cuda13",
        "cupy-cuda13x",
    )

missing = []
for package_name in required:
    try:
        print(f"{package_name}: {metadata.version(package_name)}")
    except metadata.PackageNotFoundError:
        missing.append(package_name)

for package_name in ("nixl-cu12", "mooncake-transfer-engine"):
    try:
        metadata.version(package_name)
        missing.append(f"{package_name} must not be installed")
    except metadata.PackageNotFoundError:
        pass

if missing:
    sys.exit(
        "KV connector packages are not baked into this image: "
        + ", ".join(missing)
        + ". Rebuild the CI image instead of installing them at runtime."
    )
PY

if python3 -c 'import sys, torch; sys.exit(torch.version.hip is not None)'; then
    python3 "$root/tools/kv_connector_lock.py" verify \
        "$root/requirements/test/kv_connectors_cuda.txt"
fi
