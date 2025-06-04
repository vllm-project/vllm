#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# Script to build VLLM wheel for TPU with an optional version override.

SCRIPT_PATH_PARAM="$0"
TOOLS_DIR=$(cd "$(dirname "$SCRIPT_PATH_PARAM")" && pwd) # Absolute path to the script's directory
REPO_ROOT=$(cd "$TOOLS_DIR/../../" && pwd) # Absolute path to the repo root
VLLM_DIR="$REPO_ROOT/" # Path to the vllm sources
PATCH_FILE_BASENAME="vllm-tpu.patch"
PATCH_FILE_ORIGINAL="$TOOLS_DIR/$PATCH_FILE_BASENAME"
PATCH_FILE_TEMP="" # Will be set by mktemp

# Ensure we are not running from within the vllm directory if SCRIPT_PATH_PARAM is relative like "."
if [ "$TOOLS_DIR" = "$VLLM_DIR" ]; then
    echo "Error: This script should not be run from the vllm directory directly if using relative paths."
    echo "Place it in a subdirectory like 'tools/vllm-tpu' and run it from the repository root or via its full path."
    exit 1
fi

# Check for version argument
if [ -z "$1" ]; then
    echo "Error: A version string argument is required, e.g. 0.9.0."
    echo "Usage: $0 <version>"
    exit 1
fi
USER_VERSION="$1"

# Create a temporary file for the patch
PATCH_FILE_TEMP=$(mktemp)
if [ -z "$PATCH_FILE_TEMP" ]; then
    echo "Error: Failed to create temporary patch file."
    exit 1
fi

# Ensure original patch file exists
if [ ! -f "$PATCH_FILE_ORIGINAL" ]; then
    echo "Error: Patch file not found at $PATCH_FILE_ORIGINAL"
    rm -f "$PATCH_FILE_TEMP"
    exit 1
fi

echo "User defined version: $USER_VERSION"
# Modify the patch file to insert the user-defined version.
# This sed command targets lines in the patch starting with '+',
# then looks for 'version = "PUT_VERSION_HERE"' and replaces "PUT_VERSION_HERE".
sed "s/^\(+[[:space:]]*version[[:space:]]*=[[:space:]]*\)\"PUT_VERSION_HERE\"/\1\"$USER_VERSION\"/" "$PATCH_FILE_ORIGINAL" > "$PATCH_FILE_TEMP"

# Verify that the sed command actually changed the file by replacing "PUT_VERSION_HERE"
if cmp -s "$PATCH_FILE_ORIGINAL" "$PATCH_FILE_TEMP"; then
    echo "Error: Failed to replace '"PUT_VERSION_HERE"' in the patch file '$PATCH_FILE_ORIGINAL'."
    echo "Please ensure the patch file contains a line like: '+    version = "PUT_VERSION_HERE"' that matches the sed pattern."
    rm -f "$PATCH_FILE_TEMP"
    exit 1
else
    echo "Modified patch for version override '$USER_VERSION' written to $PATCH_FILE_TEMP"
fi

# Navigate to the vllm directory
cd "$VLLM_DIR"

# Cleanup function to be called on exit or error
cleanup() {
    echo "Cleaning up..."
    if [ -f "$PATCH_FILE_TEMP" ]; then
      # Check if reverting is possible (patch was indeed applied)
      # We use the temp patch file for reversion as it contains the version specific changes.
      if git apply --reverse --check "$PATCH_FILE_TEMP" &>/dev/null; then
          echo "Reverting applied patch from $PATCH_FILE_BASENAME (using $PATCH_FILE_TEMP)..."
          git apply -R "$PATCH_FILE_TEMP"
      else
          echo "Could not directly verify or reverse patch ($PATCH_FILE_BASENAME using $PATCH_FILE_TEMP). Patch may not have been applied, or state changed."
          echo "You may need to manually clean up patched files in $VLLM_DIR (e.g., using 'git checkout HEAD -- <filename(s)>')."
          echo "Files commonly patched: setup.py, pyproject.toml"
      fi
    fi
    rm -f "$PATCH_FILE_TEMP"
    echo "Temporary patch file removed."
}
trap cleanup EXIT HUP INT QUIT PIPE TERM # Register cleanup function to run on script exit and various signals

echo "Ensuring working directory ($VLLM_DIR) is suitable for applying patch..."
if ! git diff --quiet HEAD; then
    echo "Warning: There are uncommitted changes in $VLLM_DIR."
    echo "Applying the patch might fail or have unintended consequences."
fi

echo "Checking if patch ($PATCH_FILE_BASENAME from $PATCH_FILE_TEMP) can be applied cleanly..."
if ! git apply --check "$PATCH_FILE_TEMP" &>/dev/null; then
    echo "Error: Patch (from $PATCH_FILE_TEMP) cannot be applied cleanly. Output from 'git apply --check':"
    git apply --check "$PATCH_FILE_TEMP" || true
    echo "Please resolve conflicts or clean your working directory in $VLLM_DIR."
    exit 1
fi

echo "Applying patch $PATCH_FILE_BASENAME (from $PATCH_FILE_TEMP)..."
if ! git apply "$PATCH_FILE_TEMP"; then
    echo "Error: Failed to apply patch (from $PATCH_FILE_TEMP). Output from 'git apply':"
    git apply --stat "$PATCH_FILE_TEMP" || true
    exit 1
fi

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