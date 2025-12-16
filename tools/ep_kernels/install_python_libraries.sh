#!/usr/bin/env bash
set -ex

# usage: ./install_python_libraries.sh [options]
#   --workspace <dir>    workspace directory (default: ./ep_kernels_workspace)
#   --mode <mode>        "install" (default) or "wheel"
#   --pplx-ref <commit>  pplx-kernels commit hash
#   --deepep-ref <commit> DeepEP commit hash

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PPLX_COMMIT_HASH=${PPLX_COMMIT_HASH:-"12cecfd"}
DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"73b6ea4"}
NVSHMEM_VER=3.3.24  # Suppports both CUDA 12 and 13
WORKSPACE=${WORKSPACE:-$(pwd)/ep_kernels_workspace}
MODE=${MODE:-install}
CUDA_VERSION_MAJOR=$(${CUDA_HOME}/bin/nvcc --version | egrep -o "release [0-9]+" | cut -d ' ' -f 2)

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
        --pplx-ref)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --pplx-ref requires an argument." >&2
                exit 1
            fi
            PPLX_COMMIT_HASH="$2"
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
        *)
            echo "Error: Unknown argument '$1'" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$WORKSPACE"

WHEEL_DIR="$WORKSPACE/dist"
mkdir -p "$WHEEL_DIR"

pushd "$WORKSPACE"

# install dependencies if not installed
if [ -z "$VIRTUAL_ENV" ]; then
  uv pip install --system cmake torch ninja
else
  uv pip install cmake torch ninja
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
curl -fSL "${NVSHMEM_URL}" -o "${NVSHMEM_FILE}"
tar -xf "${NVSHMEM_FILE}"
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

    # DeepEP CUDA 13 patch
    if [[ "$name" == "DeepEP" && "${CUDA_VERSION_MAJOR}" -ge 13 ]]; then
        sed -i "s|f'{nvshmem_dir}/include']|f'{nvshmem_dir}/include', '${CUDA_HOME}/include/cccl']|" "setup.py"
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

# build pplx-kernels
do_build \
    "https://github.com/ppl-ai/pplx-kernels" \
    "pplx-kernels" \
    "setup.py" \
    "$PPLX_COMMIT_HASH" \
    ""

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
