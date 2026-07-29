#!/usr/bin/env bash
# Fast structural smoke test for the full ROCm CI image.

set -euo pipefail

image_ref="${VLLM_CI_SMOKE_IMAGE:-rocm/vllm-ci:${BUILDKITE_COMMIT:?BUILDKITE_COMMIT is required}}"

docker run --rm --network=none --entrypoint /bin/bash "${image_ref}" -ec '
  if [ ! -d /vllm-workspace ]; then echo Missing directory: /vllm-workspace >&2; exit 1; fi
  if [ ! -d /vllm-workspace/tests ]; then echo Missing directory: /vllm-workspace/tests >&2; exit 1; fi
  if [ ! -d /vllm-workspace/src/vllm ]; then echo Missing directory: /vllm-workspace/src/vllm >&2; exit 1; fi
  if [ ! -x /vllm-workspace/src/vllm/vllm-rs ]; then echo Missing executable: /vllm-workspace/src/vllm/vllm-rs >&2; exit 1; fi

  command -v python3
  command -v uv
  command -v pytest
  cargo --version
  rustup --version
  protoc --version

  if ! find /opt/rocshmem \( -type f -o -type l \) -print -quit | grep -q .; then
    echo Missing ROCShmem runtime files >&2
    exit 1
  fi

  if ! command -v amd-smi >/dev/null 2>&1 && ! command -v rocminfo >/dev/null 2>&1; then
    echo No ROCm CLI found in image >&2
    exit 1
  fi

  python3 - <<PY
import importlib.util

import torch
import vllm

for module_name in ("deep_ep", "nixl"):
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing Python module: {module_name}")

print(torch.__version__)
print(vllm.__version__)
PY

  echo AMD image smoke OK
'
