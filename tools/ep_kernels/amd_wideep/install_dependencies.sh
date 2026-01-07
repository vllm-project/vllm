#!/usr/bin/env bash
set -ex
set -u

## Check if it is a ROCM system
if command -v rocm-smi &> /dev/null; then
    echo "ROCm is available."
else
    echo "Not a ROCm system. Abort."
    exit
fi

## Check if it is a wide-ep supporting GPU. Checks only MI300X for now. Expand to other GPUs.
#if rocm-smi --showproductname | grep -iq "AMD Instinct MI300X"; then
#    echo "Has MI300X GPU."
#else
#    echo "GPU not supported for wideep. Abort."
#    exit -1
#fi

export PYTORCH_ROCM_ARCH=gfx942

## Check for CX-7 Connectivity ??

WORKSPACE=${WORKSPACE:-$(pwd)/workspace}
PATH=${PATH:-""}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}

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
    local commit_hash=$3
    local key_file=$4

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

function build_rocshmem() {
    ## Install rocSHMEM from https://github.com/ROCm/DeepEP/blob/main/third-party/README.md

    # Obtain develop branch
    clone_repo https://github.com/ROCm/rocSHMEM.git rocSHMEM "" "README.md" # maybe use someother key file.

    pushd rocSHMEM
    mkdir -p build
    pushd build

    # Build dependencies Open MPI/UCX (used for RO, and as a bootstrap mechanism otherwise)
    export BUILD_DIR=$PWD
    ../scripts/install_dependencies.sh
    export PATH=$PWD/ompi/bin:$PATH
    export LD_LIBRARY_PATH=$PWD/ucx/lib:$PWD/ompi/lib:$LD_LIBRARY_PATH

    # Build rocSHMEM library, library will be installed in $HOME/rocshmem
    mkdir build.mnic && cd build.mnic
    MPI_ROOT=$BUILD_DIR/ompi ../../scripts/build_configs/gda_mlx5 --fresh \
      -DUSE_IPC=ON \
      -DGDA_BNXT=ON

    popd # build
    popd # rocSHMEM
}

function build_deepep() {

    export OMPI_DIR=$PWD/rocSHMEM/build/install/ompi
    export PATH=$OMPI_DIR/bin:$PATH
    export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

    #clone_repo https://github.com/ROCm/DeepEP DeepEP "" setup.py
    clone_repo  https://github.com/varun-sundar-rabindranath/ROCm-DeepEP.git  ROCm-DeepEP "varun/add-hidden-dim" setup.py

 
    #pushd DeepEP
    pushd ROCm-DeepEP
    python3 setup.py --variant rocm build develop
    popd
}


## Create a new workspace
mkdir -p ${WORKSPACE}

pushd ${WORKSPACE}
build_rocshmem
build_deepep
popd 



