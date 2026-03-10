#!/usr/bin/env bash
set -euo pipefail

OUTFILE="${1:-env_summary.txt}"

# ---------- helpers ----------
line() {
  printf '%s\n' "$1" >> "$OUTFILE"
}

section() {
  printf '\n%s\n' "$1" >> "$OUTFILE"
}

pkg_version() {
  # Print installed version or NOT INSTALLED
  python - <<PY 2>/dev/null
import importlib.metadata as m
try:
    print(m.version("$1"))
except Exception:
    print("NOT INSTALLED")
PY
}

pip_freeze_match() {
  # Print matching pip freeze line if present, else NOT INSTALLED
  python -m pip freeze 2>/dev/null | grep -E "^$1(==| @ |-e )|#egg=$1\$" || true
}

# ---------- start fresh ----------
: > "$OUTFILE"

# ---------- repo ----------
REPO_NAME="$(basename "$(git rev-parse --show-toplevel 2>/dev/null || pwd)")"
REPO_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'UNKNOWN')"

line "Repo: $REPO_NAME"
line "Repo commit: $REPO_COMMIT"

# ---------- python ----------
section "Python:"
PY_VER="$(python --version 2>&1 || echo 'Python UNKNOWN')"
line "$PY_VER"

# ---------- key packages ----------
section "Key packages:"

FLASHINFER_VER="$(pkg_version flashinfer-python)"
TRITON_PKG_VER="$(pkg_version pytorch-triton)"
TORCH_VER="$(pkg_version torch)"
TRITON_VER="$(pkg_version triton)"

line "flashinfer-python==$FLASHINFER_VER"
line "pytorch-triton==$TRITON_PKG_VER"
line "torch==$TORCH_VER"
line "triton==$TRITON_VER"

TORCH_CUDA="$(python - <<'PY' 2>/dev/null
try:
    import torch
    print(torch.version.cuda or "None")
except Exception:
    print("UNKNOWN")
PY
)"
line "torch cuda: $TORCH_CUDA"

TV_VER="$(pkg_version torchvision)"
TA_VER="$(pkg_version torchaudio)"

if [[ "$TV_VER" == "NOT INSTALLED" ]]; then
  line "torchvision: NOT INSTALLED"
else
  line "torchvision==$TV_VER"
fi

if [[ "$TA_VER" == "NOT INSTALLED" ]]; then
  line "torchaudio: NOT INSTALLED"
else
  line "torchaudio==$TA_VER"
fi

# Try to capture editable/source install line for vllm from pip freeze
VLLM_FREEZE_LINE="$(python -m pip freeze 2>/dev/null | grep -E '(^-e .*(#egg=vllm|#egg=.+)|^vllm(==| @ ))' | head -n 1 || true)"
if [[ -n "$VLLM_FREEZE_LINE" ]]; then
  line "$VLLM_FREEZE_LINE"
else
  # Fallback: if inside git repo, synthesize a useful line
  REPO_URL="$(git remote get-url origin 2>/dev/null || true)"
  if [[ -n "$REPO_URL" && "$REPO_COMMIT" != "UNKNOWN" ]]; then
    line "-e git+$REPO_URL@$REPO_COMMIT#egg=vllm"
  else
    line "vllm: NOT FOUND IN pip freeze"
  fi
fi

# ---------- nvcc ----------
section "CUDA compiler:"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version >> "$OUTFILE" 2>&1
else
  line "nvcc: NOT FOUND"
fi

# ---------- nvidia-smi ----------
section "NVIDIA driver:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >> "$OUTFILE" 2>&1
else
  line "nvidia-smi: NOT FOUND"
fi

echo "Wrote $OUTFILE"
