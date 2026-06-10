#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

# do-not-merge: integration-test the NIXL 1.3.0rc0 RC published to TestPyPI.
#
# 1.3.0 reworks NIXL packaging: the per-CUDA wheels now install into separate
# namespaces (nixl_cu12/, nixl_cu13/ and nixl_ep_cu12/, nixl_ep_cu13/) instead of
# a shared nixl/, and the `nixl` / `nixl_ep` meta-packages are dispatchers that
# import the variant matching torch's CUDA major at runtime. Both CUDA variants
# can therefore coexist, so the old "uninstall every variant, reinstall the
# matching one" workaround is no longer required. This PR drops it to verify the
# packaging fix end-to-end.
#
# uv must be told pre-releases are allowed (the `nixl` meta requests
# nixl-cu1{2,3}==1.3.0rc0 transitively, which uv otherwise rejects) and pointed
# at TestPyPI where the RC lives.
export UV_PRERELEASE=allow
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:+${UV_EXTRA_INDEX_URL} }https://test.pypi.org/simple/"

REQUIREMENTS_FILE="${KV_CONNECTORS_REQUIREMENTS:-/vllm-workspace/requirements/kv_connectors.txt}"

uv pip install --system -r "${REQUIREMENTS_FILE}"

# Report what got installed. With the 1.3.0 dispatcher both CUDA variants remain
# installed and `import nixl` / `import nixl_ep` select the torch-matching one.
python3 - <<'PY'
import importlib.metadata as metadata

for package_name in ("nixl", "nixl-cu12", "nixl-cu13"):
    try:
        version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        version = "not installed"
    print(f"{package_name}: {version}")
PY
