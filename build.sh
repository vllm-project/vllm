#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-"$ROOT/build-logs"}
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/build-$STAMP.log"
VERSION=${VERSION:-0.1.0}

banner() {
  cat <<EOF
============================================================
 vLLM 3080 Ti PCIe Definitive Edition v$VERSION
 One-click source build for 4x RTX 3080 Ti / SM86
============================================================
EOF
}

fail() {
  echo
  echo "BUILD FAILED"
  echo "Log: $LOG"
  echo "$*" >&2
  exit 1
}

is_positive_integer() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

detect_cpu_threads() {
  local threads
  threads=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
  if ! is_positive_integer "$threads"; then
    threads=$(nproc 2>/dev/null || true)
  fi
  if ! is_positive_integer "$threads"; then
    threads=4
  fi
  echo "$threads"
}

select_max_jobs() {
  local threads=$1
  if (( threads <= 4 )); then
    echo "$threads"
  else
    echo "$((threads - 2))"
  fi
}

run_step() {
  local title=$1
  shift
  echo
  echo "------------------------------------------------------------"
  echo "$title"
  echo "------------------------------------------------------------"
  "$@" 2>&1 | tee -a "$LOG"
}

banner
mkdir -p "$LOG_DIR"
touch "$LOG"

echo "Build log: $LOG"
echo "Source: $ROOT"

CPU_THREADS=${CPU_THREADS:-$(detect_cpu_threads)}
if ! is_positive_integer "$CPU_THREADS"; then
  fail "CPU_THREADS must be a positive integer when set explicitly."
fi
if [[ -z "${MAX_JOBS:-}" ]]; then
  MAX_JOBS=$(select_max_jobs "$CPU_THREADS")
fi
export CPU_THREADS
export MAX_JOBS

cd "$ROOT"

if [[ ! -f pyproject.toml || ! -d vllm ]]; then
  fail "Run this script from the vLLM source tree."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found. Install NVIDIA driver first."
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  else
    fail "CUDA_HOME is not set and nvcc was not found under /usr/local/cuda*."
  fi
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  fail "nvcc not found at $CUDA_HOME/bin/nvcc."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv with the official installer."
  curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tee -a "$LOG"
  export PATH="$HOME/.local/bin:$PATH"
fi

command -v uv >/dev/null 2>&1 || fail "uv install did not put uv on PATH."

export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}

run_step "GPU summary" nvidia-smi
run_step "CUDA compiler" "$CUDA_HOME/bin/nvcc" --version

if [[ ! -d .venv ]]; then
  run_step "Create Python virtualenv" uv venv --python "${PYTHON_VERSION:-3.11}" .venv
fi

if [[ ! -x .venv/bin/python ]]; then
  fail ".venv/bin/python was not created."
fi

run_step "Upgrade build frontend" uv pip install --python .venv/bin/python -U pip setuptools wheel

if [[ -f requirements/build/cuda.txt ]]; then
  run_step "Install CUDA build requirements" uv pip install --python .venv/bin/python -r requirements/build/cuda.txt --torch-backend=auto
fi

if [[ -f requirements/cuda.txt ]]; then
  run_step "Install CUDA runtime requirements" uv pip install --python .venv/bin/python -r requirements/cuda.txt --torch-backend=auto
fi

run_step "Build and install vLLM from source for SM86" \
  env \
    CUDA_HOME="$CUDA_HOME" \
    CUDA_PATH="$CUDA_PATH" \
    CUDACXX="$CUDACXX" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    MAX_JOBS="$MAX_JOBS" \
    CMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    FLASHINFER_ENABLE_AOT="$FLASHINFER_ENABLE_AOT" \
    uv pip install --python .venv/bin/python --no-build-isolation -e .

run_step "Runtime check" .venv/bin/python - <<'PY'
import importlib.util
import torch
import vllm

print(f"vllm={getattr(vllm, '__version__', 'unknown')}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"cuda_device_{idx}={torch.cuda.get_device_name(idx)}")
print(f"flashinfer_available={importlib.util.find_spec('flashinfer') is not None}")
PY

echo
echo "BUILD OK"
echo "Log: $LOG"
echo "Next steps:"
echo "  bash scripts/topo_probe.sh"
echo "  source .venv/bin/activate"
echo
