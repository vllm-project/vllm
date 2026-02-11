#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# Script to build VLLM wheel for TPU with an optional version override.

SCRIPT_PATH_PARAM="$0"
TOOLS_DIR=$(cd "$(dirname "$SCRIPT_PATH_PARAM")" && pwd) # Absolute path to the script's directory
REPO_ROOT=$(cd "$TOOLS_DIR/../../" && pwd) # Absolute path to the repo root
VLLM_DIR="$REPO_ROOT/" # Path to the vllm sources

CHANGE_FILE_LIST=(
  "vllm/entrypoints/cli/main.py"
  "vllm/entrypoints/cli/run_batch.py"
  "vllm/utils/__init__.py"
  "vllm/platforms/__init__.py"
)

# Ensure we are not running from within the vllm directory if SCRIPT_PATH_PARAM is relative like "."
if [ "$TOOLS_DIR" = "$VLLM_DIR" ]; then
    echo "Error: This script should not be run from the vllm directory directly if using relative paths."
    echo "Place it in a subdirectory like 'tools/vllm-tpu' and run it from the repository root or via its full path."
    exit 1
fi

# Optional version argument
if [ -n "$1" ]; then
    USER_VERSION="$1"
    export VLLM_VERSION_OVERRIDE="$USER_VERSION"
    echo "User defined version: $USER_VERSION"
else
    echo "No version override supplied. Using default version from source."
fi

PYPROJECT_FILE="$VLLM_DIR/pyproject.toml"

# Backup and update the project name.
if ! grep -q "name = \"vllm-tpu\"" "$PYPROJECT_FILE"; then
    echo "Patching pyproject.toml project name to vllm-tpu..."
    cp "$PYPROJECT_FILE" "${PYPROJECT_FILE}.bak"
    sed -i '0,/^name = "vllm"/s//name = "vllm-tpu"/' "$PYPROJECT_FILE"

    echo "Patching ${CHANGE_FILE_LIST[@]} vllm to vllm-tpu..."
    # patching
    #   importlib.metadata.version('vllm') -> importlib.metadata.version('vllm-tpu')
    #   importlib.metadata.version("vllm") -> importlib.metadata.version("vllm-tpu")
    #   importlib.metadata.metadata('vllm') -> importlib.metadata.metadata('vllm-tpu')
    #   importlib.metadata.metadata("vllm") -> importlib.metadata.metadata("vllm-tpu")
    #   version('vllm') -> version('vllm-tpu')
    #   version("vllm") -> version("vllm-tpu")
    sed -i \
        -e "s/importlib.metadata.version(\(['\"]\)vllm\1)/importlib.metadata.version(\1vllm-tpu\1)/" \
        -e "s/importlib.metadata.metadata(\(['\"]\)vllm\1)/importlib.metadata.metadata(\1vllm-tpu\1)/" \
        -e "s/version(\(['\"]\)vllm\1)/version(\1vllm-tpu\1)/" \
        "${CHANGE_FILE_LIST[@]}"
    PATCHED=true
else
    PATCHED=false
fi

# Navigate to the vllm directory
cd "$VLLM_DIR"

# Cleanup function to be called on exit or error
cleanup() {
    echo "Cleaning up..."
    if [ "$PATCHED" = true ]; then
        echo "Restoring original pyproject.toml..."
        cp "${PYPROJECT_FILE}.bak" "$PYPROJECT_FILE"
        rm -f "${PYPROJECT_FILE}.bak"

        echo "Restoring vllm code..."
        sed -i \
            -e "s/importlib.metadata.version(\(['\"]\)vllm-tpu\1)/importlib.metadata.version(\1vllm\1)/" \
            -e "s/importlib.metadata.metadata(\(['\"]\)vllm-tpu\1)/importlib.metadata.metadata(\1vllm\1)/" \
            -e "s/version(\(['\"]\)vllm-tpu\1)/version(\1vllm\1)/" \
            "${CHANGE_FILE_LIST[@]}"
    fi
}
trap cleanup EXIT HUP INT QUIT PIPE TERM # Register cleanup function to run on script exit and various signals

echo "Updating pyproject.toml completed. Proceeding with build..."

echo "Building wheel for TPU..."
rm -rf dist/
mkdir -p dist/

# User confirmed to use 'python -m build' directly
if ! VLLM_TARGET_DEVICE=tpu python -m build; then
    echo "Error: Python build command failed. Check if 'python -m build' works and the 'build' module is installed."
    exit 1
fi

trap - EXIT HUP INT QUIT PIPE TERM
cleanup

exit 0 