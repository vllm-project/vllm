#!/usr/bin/env bash
set -ex

# usage: ./install_python_libraries.sh [options]
#   --workspace <dir>    workspace directory (default: ./ep_kernels_workspace).
#   --mori-ref <commit>  MoRI commit hash.
#   --aiter-ref <commit> Aiter commit hash.
#   --force-install      Avoids installs when an installation exists already.

function resolve_gpu_archs() {
    arch=""

    # attempt to get arch from rocm-info
    if [ -z "$arch" ] && command -v rocminfo >/dev/null 2>&1; then
        # rocminfo Name lines could like, 
        #   Name:                    gfx942                             
        #   Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
        # it is important to add a space before gfx so we pick the first
        arch=$(rocminfo | awk '/Name:/ && / gfx/ {print $2}' | sort -u | paste -sd ';' -)
    fi

    # set defaults if everything else fails
    if [ -z "$arch" ]; then
        arch="gfx942;gfx950"
    fi

    echo "${arch}"
}

# Repo and Hash picked from Dockerfile.rocm_base
MORI_COMMIT_HASH="2d02c6a9"
MORI_REPO="https://github.com/ROCm/mori.git"
MORI_GPU_ARCHS=${GPU_ARCHS:-"$(resolve_gpu_archs)"}

AITER_COMMIT_HASH="v0.1.10.post2"
AITER_REPO="https://github.com/ROCm/aiter.git"
AITER_GPU_ARCHS=${GPU_ARCHS:-"$(resolve_gpu_archs)"}

WORKSPACE=${WORKSPACE:-$(pwd)/ep_kernels_workspace}
FORCE_INSTALL=False

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
        --mori-ref)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --mori-ref requires an argument." >&2
                exit 1
            fi
            MORI_COMMIT_HASH="$2"
            shift 2
            ;;
        --aiter-ref)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --aiter-ref requires an argument." >&2
                exit 1
            fi
            AITER_COMMIT_HASH="$2"
            shift 2
            ;;
        --force-install)
            FORCE_INSTALL=True
            shift 1
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            exit 1
            ;;
    esac
done


mkdir -p "$WORKSPACE"

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

init_submodules() {
    # Check for submodules and update if they exist
    if [ -f ".gitmodules" ]; then
        echo "Submodules detected, initializing and updating..."
        git submodule update --init --recursive
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
                init_submodules
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
            init_submodules
            cd ..
        fi
    fi
}

do_build() {
    local package_name=$1
    local extra_env=$2

    ORANGE='\033[38;5;208m'
    NC='\033[0m'

    # Check if the package is already installed
    if [ "$FORCE_INSTALL" != "True" ]; then
        if uv pip show $package_name > /dev/null 2>&1; then 
            echo -e "${ORANGE} Package $package_name is already installed. Skipping installation. Use --force-install to install again.${NC}"
            return 0
        fi
    fi

    echo "Installing $package_name into environment"
    eval "$extra_env" uv pip install --no-build-isolation -vvv .
}

do_build_mori() {
    local repo=$1
    local name=$2
    local key=$3
    local commit=$4
    local extra_env=$5

    pushd "$WORKSPACE"
    clone_repo "$repo" "$name" "$key" "$commit"
    cd "$name"

    # Patch MoRI to let it find uv's python
    sed -i 's/find_package(PythonLibs REQUIRED)/find_package(Python3 COMPONENTS Interpreter Development REQUIRED)/g' src/pybind/CMakeLists.txt
    sed -i 's/\${PYTHON_INCLUDE_DIRS}/${Python3_INCLUDE_DIRS}/g' src/pybind/CMakeLists.txt
    # Set cmake python executable to uv so it finds the right includes / libraries.
    cmake_args="CMAKE_ARGS=\"-DPython3_EXECUTABLE=$(uv python find)\""

    do_build "mori" "$cmake_args $extra_env"

    popd
}

do_build_aiter() {

    local repo=$1
    local name=$2
    local key=$3
    local commit=$4
    local extra_env=$5

    pushd "$WORKSPACE"
    clone_repo "$repo" "$name" "$key" "$commit"
    cd "$name"

    # aiter requirements
    uv pip install -r requirements.txt 
    uv pip install pyyaml
    
    # Set cmake python executable to uv so it finds the right includes / libraries.
    cmake_args="CMAKE_ARGS=\"-DPython3_EXECUTABLE=$(uv python find)\""

    do_build "amd-aiter" "$cmake_args $extra_env"

    popd
}

# build Aiter
do_build_aiter \
    "$AITER_REPO" \
    "aiter" \
    "setup.py" \
    "$AITER_COMMIT_HASH" \
    "GPU_ARCHS=\"${AITER_GPU_ARCHS}\""

# build MoRI
do_build_mori \
    "$MORI_REPO" \
    "mori" \
    "setup.py" \
    "$MORI_COMMIT_HASH" \
    "MORI_GPU_ARCHS=\"${MORI_GPU_ARCHS}\""
