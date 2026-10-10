#!/usr/bin/env bash
set -ex

# usage: ./install_python_libraries.sh [options]
#   --workspace <dir>    workspace directory (default: ./ep_kernels_workspace)
#   --mode <mode>        "install" (default) or "wheel"
#   --deepep-ref <commit> DeepEP commit hash
#   --nvshmem-ver <ver>  NVSHMEM version 

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# Pinned in full: an abbreviated hash is not a ref, so a consumer that
# fetches the pin directly ("git fetch origin <sha>") cannot resolve it.
DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb"}

NVSHMEM_VER=${NVSHMEM_VER:-"3.3.24"}  # Default supports both CUDA 12 and 13
WORKSPACE=${WORKSPACE:-$(pwd)/ep_kernels_workspace}
MODE=${MODE:-install}
# Directory holding vendored DeepEP patches applied by this script
PATCHES_DIR=${PATCHES_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches"}
CUDA_VERSION_MAJOR=$("${CUDA_HOME}"/bin/nvcc --version | grep -E -o "release [0-9]+" | cut -d ' ' -f 2)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workspace)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --workspace requires an argument." >&2
                exit 1
            fi
            WORKSPACE="$2"
            shift 2
            ;;
        --mode)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --mode requires an argument." >&2
                exit 1
            fi
            MODE="$2"
            shift 2
            ;;
        --deepep-ref)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --deepep-ref requires an argument." >&2
                exit 1
            fi
            DEEPEP_COMMIT_HASH="$2"
            shift 2
            ;;
        --nvshmem-ver)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --nvshmem-ver requires an argument." >&2
                exit 1
            fi
            if [[ "$2" =~ / ]]; then
                echo "Error: NVSHMEM version should not contain slashes." >&2
                exit 1
            fi
            NVSHMEM_VER="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            exit 1
            ;;
    esac
done

# Validate NVSHMEM_VER to prevent path traversal attacks
# Only allow alphanumeric characters, dots, and hyphens (typical version string chars)
if [[ ! "$NVSHMEM_VER" =~ ^[a-zA-Z0-9.-]+$ ]]; then
    echo "Error: NVSHMEM_VER contains invalid characters. Only alphanumeric, dots, and hyphens are allowed." >&2
    exit 1
fi

mkdir -p "$WORKSPACE"

WHEEL_DIR="$WORKSPACE/dist"
mkdir -p "$WHEEL_DIR"

pushd "$WORKSPACE"

# install dependencies if not installed
UV_PIP_TARGET=()
if [ -z "$VIRTUAL_ENV" ]; then
  UV_PIP_TARGET=(--system)
fi
uv pip install "${UV_PIP_TARGET[@]}" cmake torch ninja

# DeepEPv2 needs NCCL >= 2.30.4 at both build time and runtime, but PyTorch
# pins an older release.
DEEPEP_V2_MIN_NCCL="2.30.4"
NCCL_PACKAGE="nvidia-nccl-cu${CUDA_VERSION_MAJOR}"
NCCL_WARNING=""
NCCL_INSTALLED=$(uv pip show "${UV_PIP_TARGET[@]}" "$NCCL_PACKAGE" 2>/dev/null | sed -n 's/^Version: //p')
if [ -n "$NCCL_INSTALLED" ] && \
    [ "$(printf '%s\n' "$DEEPEP_V2_MIN_NCCL" "$NCCL_INSTALLED" | sort -V | head -n1)" != "$DEEPEP_V2_MIN_NCCL" ]; then
    NCCL_WARNING="WARNING: ${NCCL_PACKAGE} ${NCCL_INSTALLED} is installed, but the deepep_v2 backend
requires NCCL >= ${DEEPEP_V2_MIN_NCCL} when DeepEP is built and when it runs. To use deepep_v2, run
    uv pip install \"${NCCL_PACKAGE}>=${DEEPEP_V2_MIN_NCCL}\" --no-deps
and then run this script again. See tools/ep_kernels/README.md."
    echo "$NCCL_WARNING" >&2
fi

# fetch nvshmem
ARCH=$(uname -m)
case "${ARCH,,}" in
  x86_64|amd64)
    NVSHMEM_SUBDIR="linux-x86_64"
    ;;
  aarch64|arm64)
    NVSHMEM_SUBDIR="linux-sbsa"
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

NVSHMEM_FILE="libnvshmem-${NVSHMEM_SUBDIR}-${NVSHMEM_VER}_cuda${CUDA_VERSION_MAJOR}-archive.tar.xz"
NVSHMEM_URL="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/${NVSHMEM_SUBDIR}/${NVSHMEM_FILE}"

pushd "$WORKSPACE"
echo "Downloading NVSHMEM ${NVSHMEM_VER} for ${NVSHMEM_SUBDIR} ..."
curl -fSL --retry 3 --retry-delay 2 "${NVSHMEM_URL}" -o "${NVSHMEM_FILE}"
tar -xf "${NVSHMEM_FILE}"
rm -rf nvshmem
mv "${NVSHMEM_FILE%.tar.xz}" nvshmem
rm -f "${NVSHMEM_FILE}"
rm -rf nvshmem/lib/bin nvshmem/lib/share
popd

export CMAKE_PREFIX_PATH=$WORKSPACE/nvshmem/lib/cmake:$CMAKE_PREFIX_PATH

is_git_dirty() {
    local dir=$1
    pushd "$dir" > /dev/null
    if [ -d ".git" ] && [ -n "$(git status --porcelain 3>/dev/null)" ]; then
        popd > /dev/null
        return 0
    else
        popd > /dev/null
        return 1
    fi
}

clone_repo() {
    local repo_url=$1
    local dir_name=$2
    local key_file=$3
    local commit_hash=$4
    if [ -d "$dir_name" ]; then
        if is_git_dirty "$dir_name"; then
            echo "$dir_name directory is dirty, skipping clone"
        elif [ ! -d "$dir_name/.git" ] || [ ! -f "$dir_name/$key_file" ]; then
            echo "$dir_name directory exists but clone appears incomplete, cleaning up and re-cloning"
            rm -rf "$dir_name"
            git clone "$repo_url"
            if [ -n "$commit_hash" ]; then
                cd "$dir_name"
                git checkout "$commit_hash"
                cd ..
            fi
        else
            echo "$dir_name directory exists and appears complete"
        fi
    else
        git clone "$repo_url"
        if [ -n "$commit_hash" ]; then
            cd "$dir_name"
            git checkout "$commit_hash"
            cd ..
        fi
    fi
}

do_build() {
    local repo=$1
    local name=$2
    local key=$3
    local commit=$4
    local extra_env=$5

    pushd "$WORKSPACE"
    clone_repo "$repo" "$name" "$key" "$commit"
    cd "$name"

    # DeepEP GIN barrier data-visibility fix (temporary until upstream ships a fix):
    # the world-team barrier signals completion on context 0 only while data puts ride
    # contexts 1..N (RDMA) and NVLink TMA stores (same-node peers), so the completion
    # signal can overtake in-flight data and receivers read stale raw-buffer rows
    # (silent dispatch corruption -> combine illegal memory accesses).
    # The patch is pinned to DeepEP d4f41e4e; skip loudly on any other ref.
    if [[ "$name" == "DeepEP" ]] && \
        [[ "$(git rev-parse HEAD)" == d4f41e4e* ]] && \
        ! grep -q "Signal on every context, not just context 0" \
            deep_ep/include/deep_ep/common/comm.cuh; then
        patch --batch -p1 < "${PATCHES_DIR}/deepep-gin-barrier-data-visibility.patch"
    fi

    # DeepEP CUDA 13 patch
    if [[ "$name" == "DeepEP" && "${CUDA_VERSION_MAJOR}" -ge 13 ]]; then
        sed -i "s|f'{nvshmem_dir}/include']|f'{nvshmem_dir}/include', '${CUDA_HOME}/include/cccl']|" "setup.py"
    fi

    # DeepEPv2 requires Linux 5.6+ at runtime for pidfd_getfd (pidfd_open was
    # added in Linux 5.3), but manylinux headers predate both definitions.
    # DeepEP is built as a separate wheel for the vLLM container image and is
    # not included in the vLLM wheel, so this does not change its manylinux ABI.
    if [[ "$name" == "DeepEP" ]] && \
        ! grep -q "vLLM manylinux syscall compatibility" \
            csrc/kernels/backend/symmetric.hpp; then
        sed -i '1i\
// vLLM manylinux syscall compatibility\
#if defined(__x86_64__) || defined(__aarch64__)\
#ifndef SYS_pidfd_open\
#ifdef __NR_pidfd_open\
#define SYS_pidfd_open __NR_pidfd_open\
#else\
#define SYS_pidfd_open 434\
#endif\
#endif\
#ifndef SYS_pidfd_getfd\
#ifdef __NR_pidfd_getfd\
#define SYS_pidfd_getfd __NR_pidfd_getfd\
#else\
#define SYS_pidfd_getfd 438\
#endif\
#endif\
#endif' csrc/kernels/backend/symmetric.hpp
    fi

    if [[ "$name" == "DeepEP" ]]; then
        # DeepEP links against the CUDA driver API in driverless build images.
        local cuda_driver_stub
        local cuda_driver_stub_dir
        cuda_driver_stub=$(
            find -H "$CUDA_HOME" -path "*/stubs/libcuda.so" -print -quit
        )
        if [[ -z "$cuda_driver_stub" ]]; then
            echo "CUDA driver stub not found under $CUDA_HOME" >&2
            exit 1
        fi
        cuda_driver_stub_dir=$(dirname "$cuda_driver_stub")
        export LIBRARY_PATH="${cuda_driver_stub_dir}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    fi

    if [ "$MODE" = "install" ]; then
        echo "Installing $name into environment"
        eval "$extra_env" uv pip install --no-build-isolation -vvv .
    else
        echo "Building $name wheel into $WHEEL_DIR"
        eval "$extra_env" uv build --wheel --no-build-isolation -vvv --out-dir "$WHEEL_DIR" .
    fi
    popd
}

# build DeepEP
do_build \
    "https://github.com/deepseek-ai/DeepEP" \
    "DeepEP" \
    "setup.py" \
    "$DEEPEP_COMMIT_HASH" \
    "export NVSHMEM_DIR=$WORKSPACE/nvshmem; "

if [ "$MODE" = "wheel" ]; then
    echo "All wheels written to $WHEEL_DIR"
    ls -l "$WHEEL_DIR"
fi

# Repeat after the verbose build output, where the first copy is easy to miss.
if [ -n "$NCCL_WARNING" ]; then
    echo "$NCCL_WARNING" >&2
fi
