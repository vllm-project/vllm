#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ASCEND_TORCH_VERSION="${ASCEND_TORCH_VERSION:-2.10.0}"
ASCEND_TORCH_NPU_VERSION="${ASCEND_TORCH_NPU_VERSION:-2.10.0}"
ASCEND_SETUPTOOLS_SPEC="${ASCEND_SETUPTOOLS_SPEC:-setuptools>=77.0.3,<81.0.0}"

check_stack() {
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" - "$ASCEND_TORCH_VERSION" "$ASCEND_TORCH_NPU_VERSION" <<'PY'
import importlib.metadata as md
import sys

expected_torch, expected_torch_npu = sys.argv[1:3]

def normalize(version: str) -> str:
    return version.split("+", 1)[0]

def dist_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

torch_version = dist_version("torch")
torch_npu_version = dist_version("torch-npu") or dist_version("torch_npu")
setuptools_version = dist_version("setuptools")

print(f"torch={torch_version}")
print(f"torch-npu={torch_npu_version}")
print(f"setuptools={setuptools_version}")

if normalize(torch_version or "") != expected_torch:
    raise SystemExit(1)
if normalize(torch_npu_version or "") != expected_torch_npu:
    raise SystemExit(1)
if setuptools_version is None:
    raise SystemExit(1)
major = int(setuptools_version.split(".", 1)[0])
if major >= 81:
    raise SystemExit(1)
PY
}

if check_stack; then
  echo "Ascend torch stack already matches the pinned CI matrix."
else
  echo "Installing pinned Ascend torch stack:"
  echo "  torch==$ASCEND_TORCH_VERSION"
  echo "  torch-npu==$ASCEND_TORCH_NPU_VERSION"
  echo "  $ASCEND_SETUPTOOLS_SPEC"

  "$PYTHON_BIN" -m pip uninstall -y torchvision torchaudio 2>/dev/null || true
  PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON_BIN" -m pip install --upgrade --force-reinstall \
    "torch==$ASCEND_TORCH_VERSION" \
    "torch-npu==$ASCEND_TORCH_NPU_VERSION" \
    "$ASCEND_SETUPTOOLS_SPEC"
fi

TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu

print("torch", torch.__version__, torch.__file__)
print("torch_npu", torch_npu.__version__, torch_npu.__file__)
print("has torch.npu", hasattr(torch, "npu"))
PY
