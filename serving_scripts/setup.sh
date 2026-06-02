module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_DIR="${REPO_ROOT}/.venv"
echo "REPO_ROOT=$REPO_ROOT"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_VENV_BASE" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

PYTHON_PATH="$(command -v python)"
echo "Using python: $PYTHON_PATH"

EXPECTED_PYTHON="$VENV_DIR/bin/python"
if [ "$PYTHON_PATH" != "$EXPECTED_PYTHON" ]; then
  echo "Error: python did not resolve to venv interpreter." >&2
  echo "Expected: $EXPECTED_PYTHON" >&2
  echo "Got:      $PYTHON_PATH" >&2
  exit 1
fi

INSTALL_DEPS="${INSTALL_DEPS:-0}"

if [ "${INSTALL_DEPS}" = "1" ]; then
  echo "INSTALL_DEPS=1; installing dependencies and editable vLLM..."

  python -m pip install -U pip
  python -m pip install -r "$REPO_ROOT/requirements/cuda.txt"
  python -m pip install -r "$REPO_ROOT/requirements/build/cuda.txt"

  (
    cd "$REPO_ROOT" || exit 1
    export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"

    # Avoid transient shared-filesystem Git/versioning failures.
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM:-0.19.2.dev0}"
    export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.19.2.dev0}"

    install_ok=0
    for attempt in 1 2 3; do
      echo "Editable vLLM install attempt ${attempt}/3..."
      if python -m pip install -e . ${VLLM_PIP_INSTALL_EXTRA_ARGS:-}; then
        install_ok=1
        break
      fi
      echo "Editable install failed on attempt ${attempt}; retrying in 30s..."
      sleep 30
    done

    if [ "${install_ok}" != "1" ]; then
      echo "Error: editable vLLM install failed after 3 attempts." >&2
      exit 1
    fi
  )
else
  echo "INSTALL_DEPS=${INSTALL_DEPS}; skipping pip install steps and using existing venv."
fi

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --disable-custom-all-reduce
