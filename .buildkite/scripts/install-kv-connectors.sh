#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

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

# The mooncake-transfer-engine PyPI wheel is built against CUDA 12: its compiled
# extension hard-links libcudart.so.12, which the CUDA 13 runtime image does not
# ship. Supply the CUDA 12 runtime so `import mooncake.engine` resolves.
if ! python3 -c "import mooncake.engine" 2>/dev/null; then
    echo "mooncake import failed; installing CUDA 12 runtime for libcudart.so.12"
    uv pip install --system nvidia-cuda-runtime-cu12
    CUDART12=$(python3 - <<'PY'
import importlib.util
import os.path

spec = importlib.util.find_spec("nvidia.cuda_runtime")
path = ""
if spec and spec.origin:
    candidate = os.path.join(os.path.dirname(spec.origin), "lib", "libcudart.so.12")
    if os.path.exists(candidate):
        path = candidate
print(path)
PY
)
    if [ -n "${CUDART12}" ]; then
        ln -sf "${CUDART12}" /usr/local/cuda/lib64/libcudart.so.12
        ldconfig 2>/dev/null || true
    fi
fi

# Env diagnostics + import canary. Surfaces the real reason mooncake can't load
# (instead of the silent "Mooncake is not available" at engine startup) and, on
# failure, runs ldd on the compiled extension to name the unresolved library.
echo "=== KV connector env diagnostics ==="
echo "python: $(command -v python3)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "--- relevant packages ---"
uv pip list 2>/dev/null | grep -iE 'mooncake|nixl|cupy|lmcache' || true
echo "--- libcudart on disk ---"
ls -l /usr/local/cuda/lib64/libcudart.so* 2>/dev/null || true
ldconfig -p 2>/dev/null | grep -i libcudart || true

if ! python3 -c "import mooncake.engine; print('mooncake.engine import OK')"; then
    echo "=== mooncake import failed; ldd on the package's shared objects ==="
    MOONCAKE_DIR=$(python3 -c "import importlib.util, os.path; \
spec = importlib.util.find_spec('mooncake'); \
print(os.path.dirname(spec.origin) if spec and spec.origin else '')" 2>/dev/null || true)
    if [ -n "${MOONCAKE_DIR}" ]; then
        echo "package dir: ${MOONCAKE_DIR}"
        find "${MOONCAKE_DIR}" -name '*.so' -print -exec ldd {} \; || true
    fi
    exit 1
fi
