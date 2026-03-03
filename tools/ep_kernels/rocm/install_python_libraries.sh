#!/usr/bin/env bash
set -ex

# usage: ./install_python_libraries.sh [options]
#   --workspace <dir>    workspace directory (default: ./ep_kernels_workspace)
#   --mode <mode>        "install" (default) or "wheel"
#   --mori-ref <commit> MoRI commit hash

function resolve_mori_gpu_archs() {
    arch=${MORI_GPU_ARCHS:-""}

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

MORI_COMMIT_HASH="2d02c6a9"
MORI_REPO="https://github.com/ROCm/mori.git"
MORI_GPU_ARCHS=$(resolve_mori_gpu_archs)

WORKSPACE=${WORKSPACE:-$(pwd)/ep_kernels_workspace}
MODE=${MODE:-install}

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
        --mori-ref)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --mori-ref requires an argument." >&2
                exit 1
            fi
            MORI_COMMIT_HASH="$2"
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
    local repo=$1
    local name=$2
    local key=$3
    local commit=$4
    local extra_env=$5

    pushd "$WORKSPACE"
    clone_repo "$repo" "$name" "$key" "$commit"
    cd "$name"

    cmake_args="CMAKE_ARGS=\"-DPython3_EXECUTABLE=$(uv python find) -DPYTHON_EXECUTABLE=$(uv python find)\""

    if [ "$MODE" = "install" ]; then
        echo "Installing $name into environment"
        eval "$cmake_args $extra_env" uv pip install --no-build-isolation -vvv .
    else
        echo "Building $name wheel into $WHEEL_DIR"
        eval "$cmake_args $extra_env" uv build --wheel --no-build-isolation -vvv --out-dir "$WHEEL_DIR" .
    fi
    popd
}

# build MoRI
do_build \
    "$MORI_REPO" \
    "mori" \
    "setup.py" \
    "$MORI_COMMIT_HASH" \
    "MORI_GPU_ARCHS=${MORI_GPU_ARCHS}"
    
if [ "$MODE" = "wheel" ]; then
    echo "All wheels written to $WHEEL_DIR"
    ls -l "$WHEEL_DIR"
fi
