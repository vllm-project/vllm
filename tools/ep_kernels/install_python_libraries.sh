set -ex

# prepare workspace directory
WORKSPACE=$1
if [ -z "$WORKSPACE" ]; then
    export WORKSPACE=$(pwd)/ep_kernels_workspace
fi

if [ ! -d "$WORKSPACE" ]; then
    mkdir -p $WORKSPACE
fi

# install dependencies if not installed
pip3 install cmake torch ninja

# build nvshmem
pushd $WORKSPACE
mkdir -p nvshmem_src
wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
tar -xvf nvshmem_src_3.2.5-1.txz -C nvshmem_src --strip-components=1
pushd nvshmem_src
wget https://github.com/deepseek-ai/DeepEP/raw/main/third-party/nvshmem.patch
git init
git apply -vvv nvshmem.patch

# assume CUDA_HOME is set correctly
if [ -z "$CUDA_HOME" ]; then
    echo "CUDA_HOME is not set, please set it to your CUDA installation directory."
    exit 1
fi

# assume TORCH_CUDA_ARCH_LIST is set correctly
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    echo "TORCH_CUDA_ARCH_LIST is not set, please set it to your desired architecture."
    exit 1
fi

# disable all features except IBGDA
export NVSHMEM_IBGDA_SUPPORT=1

export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_IBRC_SUPPORT=0
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0

cmake -G Ninja -S . -B $WORKSPACE/nvshmem_build/ -DCMAKE_INSTALL_PREFIX=$WORKSPACE/nvshmem_install
cmake --build $WORKSPACE/nvshmem_build/ --target install

popd

export CMAKE_PREFIX_PATH=$WORKSPACE/nvshmem_install:$CMAKE_PREFIX_PATH

is_git_dirty() {
    local dir=$1
    pushd "$dir" > /dev/null

    if [ -d ".git" ] && [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        popd > /dev/null
        return 0  # dirty (true)
    else
        popd > /dev/null
        return 1  # clean (false)
    fi
}

# Function to handle git repository cloning with dirty/incomplete checks
clone_repo() {
    local repo_url=$1
    local dir_name=$2
    local key_file=$3

    if [ -d "$dir_name" ]; then
        # Check if directory has uncommitted changes (dirty)
        if is_git_dirty "$dir_name"; then
            echo "$dir_name directory is dirty, skipping clone"
        # Check if clone failed (directory exists but not a valid git repo or missing key files)
        elif [ ! -d "$dir_name/.git" ] || [ ! -f "$dir_name/$key_file" ]; then
            echo "$dir_name directory exists but clone appears incomplete, cleaning up and re-cloning"
            rm -rf "$dir_name"
            git clone "$repo_url"
        else
            echo "$dir_name directory exists and appears complete; manually update if needed"
        fi
    else
        git clone "$repo_url"
    fi
}

# build and install pplx, require pytorch installed
pushd $WORKSPACE
clone_repo "https://github.com/ppl-ai/pplx-kernels" "pplx-kernels" "setup.py"
cd pplx-kernels
# see https://github.com/pypa/pip/issues/9955#issuecomment-838065925
# PIP_NO_BUILD_ISOLATION=0 disables build isolation
PIP_NO_BUILD_ISOLATION=0 pip install -vvv -e  .
popd

# build and install deepep, require pytorch installed
pushd $WORKSPACE
clone_repo "https://github.com/deepseek-ai/DeepEP" "DeepEP" "setup.py"
cd DeepEP
export NVSHMEM_DIR=$WORKSPACE/nvshmem_install
PIP_NO_BUILD_ISOLATION=0 pip install -vvv -e  .
popd
