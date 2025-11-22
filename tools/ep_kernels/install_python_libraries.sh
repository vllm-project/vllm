#!/usr/bin/env bash
set -ex

# usage: ./build.sh [workspace_dir] [mode]
#   mode: "install" (default) → install directly into current Python env
#         "wheel"              → build wheels into WORKSPACE/dist

WORKSPACE=${1:-$(pwd)/ep_kernels_workspace}
MODE=${2:-install}
mkdir -p "$WORKSPACE"

WHEEL_DIR="$WORKSPACE/dist"
mkdir -p "$WHEEL_DIR"
NVSHMEM_VER=3.3.9

pushd "$WORKSPACE"

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

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
    NVSHMEM_FILE="libnvshmem-linux-x86_64-${NVSHMEM_VER}_cuda12-archive.tar.xz"
    ;;
  aarch64|arm64)
    NVSHMEM_SUBDIR="linux-sbsa"
    NVSHMEM_FILE="libnvshmem-linux-sbsa-${NVSHMEM_VER}_cuda12-archive.tar.xz"
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

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
    "12cecfd" \
    ""

# build DeepEP
do_build \
    "https://github.com/deepseek-ai/DeepEP" \
    "DeepEP" \
    "setup.py" \
    "73b6ea4" \
    "export NVSHMEM_DIR=$WORKSPACE/nvshmem; "

if [ "$MODE" = "wheel" ]; then
    echo "All wheels written to $WHEEL_DIR"
    ls -l "$WHEEL_DIR"
fi
