#!/usr/bin/env bash
set -euo pipefail

VLLM_DEBUG_TAG="v0.27.1"
VLLM_DEBUG_BRANCH="debug-v0.27.1"
VLLM_DEBUG_COMMIT="6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
VLLM_WHEEL_VARIANT="cu129"
VLLM_VENV_DIR=".venv"
VLLM_REPOSITORY_URL="https://github.com/vllm-project/vllm.git"
CUDA_COMPAT_PACKAGE="cuda-compat-12-9"
CUDA_COMPAT_PATH="${CUDA_COMPAT_PATH:-}"
TORCHCODEC_VERSION="0.16.0+cu129"
PYTORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu129"

log() {
    echo "[vLLM debug setup] $*"
}

fail() {
    echo "[vLLM debug setup] ERROR: $*" >&2
    exit 1
}

if [[ $# -gt 1 ]]; then
    fail "Usage: $0 [VLLM_REPOSITORY_DIRECTORY]"
fi

requested_repo="${1:-}"
if [[ -z "${requested_repo}" ]]; then
    if repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
        requested_repo="${repo_root}"
    else
        requested_repo="${PWD}/vllm"
    fi
fi

if ! git -C "${requested_repo}" rev-parse --git-dir >/dev/null 2>&1; then
    if [[ -e "${requested_repo}" ]]; then
        fail "${requested_repo} exists but is not a Git repository."
    fi
    log "Cloning ${VLLM_DEBUG_TAG} into ${requested_repo}..."
    git clone --branch "${VLLM_DEBUG_TAG}" --depth 1 \
        "${VLLM_REPOSITORY_URL}" "${requested_repo}"
fi

repo_root="$(git -C "${requested_repo}" rev-parse --show-toplevel)"
cd "${repo_root}"

if [[ ! -f pyproject.toml || ! -d vllm ]]; then
    fail "${repo_root} does not look like a vLLM repository."
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    fail "This setup targets a Linux NVIDIA CUDA server."
fi

command -v nvidia-smi >/dev/null 2>&1 || \
    fail "nvidia-smi was not found. Check the NVIDIA driver first."

log "Detected GPU configuration:"
nvidia-smi \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader

if ! git cat-file -e "${VLLM_DEBUG_TAG}^{commit}" 2>/dev/null; then
    git fetch origin \
        "refs/tags/${VLLM_DEBUG_TAG}:refs/tags/${VLLM_DEBUG_TAG}"
fi

git cat-file -e "${VLLM_DEBUG_TAG}^{commit}" 2>/dev/null || \
    fail "Tag ${VLLM_DEBUG_TAG} was not found after fetching it."

tag_commit="$(git rev-parse "${VLLM_DEBUG_TAG}^{commit}")"
if [[ "${tag_commit}" != "${VLLM_DEBUG_COMMIT}" ]]; then
    fail "${VLLM_DEBUG_TAG} resolved to unexpected commit ${tag_commit}."
fi

current_commit="$(git rev-parse HEAD)"
if [[ "${current_commit}" != "${VLLM_DEBUG_COMMIT}" ]]; then
    if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
        fail "Tracked changes exist. Commit or stash them before switching versions."
    fi

    if git show-ref --verify --quiet "refs/heads/${VLLM_DEBUG_BRANCH}"; then
        branch_commit="$(git rev-parse "${VLLM_DEBUG_BRANCH}")"
        if [[ "${branch_commit}" != "${VLLM_DEBUG_COMMIT}" ]]; then
            fail "Branch ${VLLM_DEBUG_BRANCH} exists at ${branch_commit}, not ${VLLM_DEBUG_COMMIT}."
        fi
        git switch "${VLLM_DEBUG_BRANCH}"
    else
        git switch -c "${VLLM_DEBUG_BRANCH}" "${VLLM_DEBUG_TAG}"
    fi
elif [[ "$(git branch --show-current)" != "${VLLM_DEBUG_BRANCH}" ]]; then
    if git show-ref --verify --quiet "refs/heads/${VLLM_DEBUG_BRANCH}"; then
        git switch "${VLLM_DEBUG_BRANCH}"
    else
        git switch -c "${VLLM_DEBUG_BRANCH}"
    fi
fi

if command -v rpm >/dev/null 2>&1 && \
    rpm -q "${CUDA_COMPAT_PACKAGE}" >/dev/null 2>&1; then
    log "${CUDA_COMPAT_PACKAGE} is already installed."
else
    if [[ "$(id -u)" -ne 0 ]]; then
        fail "Run as root to install ${CUDA_COMPAT_PACKAGE}."
    fi
    if command -v yum >/dev/null 2>&1; then
        log "Installing ${CUDA_COMPAT_PACKAGE} with yum..."
        yum install -y "${CUDA_COMPAT_PACKAGE}"
    elif command -v dnf >/dev/null 2>&1; then
        log "Installing ${CUDA_COMPAT_PACKAGE} with dnf..."
        dnf install -y "${CUDA_COMPAT_PACKAGE}"
    else
        fail "yum or dnf is required to install ${CUDA_COMPAT_PACKAGE}."
    fi
fi

resolve_cuda_compat_path() {
    local -a candidates=()
    local pkg_file=""
    local dir=""

    if command -v rpm >/dev/null 2>&1 && \
        rpm -q "${CUDA_COMPAT_PACKAGE}" >/dev/null 2>&1; then
        while IFS= read -r pkg_file; do
            [[ "${pkg_file}" == */libcuda.so.1 ]] && \
                candidates+=("$(dirname "${pkg_file}")")
        done < <(rpm -ql "${CUDA_COMPAT_PACKAGE}" 2>/dev/null)
    fi

    if command -v dpkg-query >/dev/null 2>&1 && \
        dpkg-query -s "${CUDA_COMPAT_PACKAGE}" >/dev/null 2>&1; then
        while IFS= read -r pkg_file; do
            [[ "${pkg_file}" == */libcuda.so.1 ]] && \
                candidates+=("$(dirname "${pkg_file}")")
        done < <(dpkg-query -L "${CUDA_COMPAT_PACKAGE}" 2>/dev/null)
    fi

    candidates+=(
        "/usr/local/cuda-12.9/cuda-compat"
        "/usr/local/cuda/cuda-compat"
        "/usr/lib64/cuda-compat-12-9"
        "/usr/lib/x86_64-linux-gnu/cuda-compat-12-9"
    )

    for dir in "${candidates[@]}"; do
        if [[ -d "${dir}" && -e "${dir}/libcuda.so.1" ]]; then
            printf '%s\n' "${dir}"
            return 0
        fi
    done
    return 1
}

if [[ -z "${CUDA_COMPAT_PATH}" ]]; then
    CUDA_COMPAT_PATH="$(resolve_cuda_compat_path)" || \
        fail "Could not locate the libcuda.so.1 shipped by ${CUDA_COMPAT_PACKAGE}. Set CUDA_COMPAT_PATH manually."
fi

if [[ ! -d "${CUDA_COMPAT_PATH}" ]]; then
    fail "CUDA compatibility directory ${CUDA_COMPAT_PATH} was not found."
fi

log "Using CUDA compatibility libraries from ${CUDA_COMPAT_PATH}."

export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="${CUDA_COMPAT_PATH}"
export LD_LIBRARY_PATH="${CUDA_COMPAT_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if ! command -v uv >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || \
        fail "curl is required to install uv."
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [[ -x "${HOME}/.local/bin/uv" ]]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
fi

command -v uv >/dev/null 2>&1 || \
    fail "uv was installed but is not available on PATH. Start a new shell and rerun."

if [[ ! -x "${VLLM_VENV_DIR}/bin/python" ]]; then
    log "Creating the Python 3.12 virtual environment..."
    uv venv --python 3.12 "${VLLM_VENV_DIR}"
else
    log "Reusing ${VLLM_VENV_DIR}."
fi

activation_marker="# vLLM CUDA 12.9 compatibility"
if ! grep -Fq "${activation_marker}" "${VLLM_VENV_DIR}/bin/activate"; then
    printf '\n%s\n%s\n%s\n' \
        "${activation_marker}" \
        "export VLLM_ENABLE_CUDA_COMPATIBILITY=1" \
        "export VLLM_CUDA_COMPATIBILITY_PATH=${CUDA_COMPAT_PATH}" \
        >>"${VLLM_VENV_DIR}/bin/activate"
fi

ld_library_path_export="export LD_LIBRARY_PATH=\"${CUDA_COMPAT_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\""
if ! grep -Fqx \
    "${ld_library_path_export}" "${VLLM_VENV_DIR}/bin/activate"; then
    printf '%s\n' "${ld_library_path_export}" \
        >>"${VLLM_VENV_DIR}/bin/activate"
fi

log "Installing lint tools and pre-commit hooks..."
uv pip install --python "${VLLM_VENV_DIR}/bin/python" \
    -r requirements/lint.txt
"${VLLM_VENV_DIR}/bin/pre-commit" install

log "Installing editable vLLM with the verified ${VLLM_WHEEL_VARIANT} wheel..."
VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_DEBUG_COMMIT}" \
VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_WHEEL_VARIANT}" \
VLLM_USE_PRECOMPILED=1 \
uv pip install --python "${VLLM_VENV_DIR}/bin/python" \
    --editable . \
    --torch-backend=auto

log "Installing the matching TorchCodec CUDA wheel..."
uv pip install --python "${VLLM_VENV_DIR}/bin/python" \
    --reinstall \
    --no-deps \
    "torchcodec==${TORCHCODEC_VERSION}" \
    --index-url "${PYTORCH_CUDA_INDEX}"

log "Verifying the installation..."
"${VLLM_VENV_DIR}/bin/python" -c \
    'import importlib.metadata; import vllm; import torch; assert torch.cuda.is_available(); print("vLLM:", vllm.__version__); print("TorchCodec:", importlib.metadata.version("torchcodec")); print("GPU:", torch.cuda.get_device_name(0)); print("Source:", vllm.__file__)'

log "Setup completed. Activate the environment with:"
echo "source ${VLLM_VENV_DIR}/bin/activate"
log "For breakpoint-friendly execution, use:"
echo "VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=DEBUG vllm serve <MODEL> --enforce-eager"
