#!/usr/bin/env bash
# Upload script for macOS wheels to S3 (wheels.vllm.ai).
# Adapted from upload-nightly-wheels.sh for use outside Buildkite
# (e.g., GitHub Actions) and for wheels that don't need the
# linux -> manylinux filename rename.

set -ex

# ======== part 0: setup ========

BUCKET="vllm-wheels"
INDICES_OUTPUT_DIR="indices"
DEFAULT_VARIANT_ALIAS="cu129" # align with vLLM_MAIN_CUDA_VERSION in vllm/envs.py
PYTHON=${PYTHON_PROG:=python3}

# CI-agnostic env vars (set by the calling workflow)
SUBPATH=${COMMIT_SHA:?COMMIT_SHA must be set}
BRANCH=${BRANCH_NAME:?BRANCH_NAME must be set}
IS_PR=${IS_PULL_REQUEST:-false}

S3_COMMIT_PREFIX="s3://$BUCKET/$SUBPATH/"

echo "Using python interpreter: $PYTHON"
echo "Python version: $($PYTHON --version)"

# ========= part 1: collect & upload the wheel ==========
# macOS wheels already have the correct platform tag (e.g., macosx_11_0_arm64)
# so no filename renaming is needed.

wheel_files=(dist/*.whl)

# Check that exactly one wheel is found
if [[ ${#wheel_files[@]} -ne 1 ]]; then
  echo "Error: Expected exactly one wheel file in dist/, but found ${#wheel_files[@]}"
  exit 1
fi
wheel="${wheel_files[0]}"
echo "Wheel to upload: $wheel"

# Extract the version from the wheel
version=$(unzip -p "$wheel" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
echo "Version in wheel: $version"
pure_version="${version%%+*}"
echo "Pure version (without variant): $pure_version"

# Copy wheel to its own bucket
aws s3 cp "$wheel" "$S3_COMMIT_PREFIX"

# ========= part 2: generate and upload indices ==========
# Generate indices for all existing wheels in the commit directory.
# This script might be run multiple times if there are multiple variants being built,
# so we need to guarantee there is little chance for "TOCTOU" issues.

echo "Existing wheels on S3:"
aws s3 ls "$S3_COMMIT_PREFIX"
obj_json="objects.json"
aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$SUBPATH/" --delimiter / --output json > "$obj_json"
mkdir -p "$INDICES_OUTPUT_DIR"

alias_args=()
if [[ -n "$DEFAULT_VARIANT_ALIAS" ]]; then
    alias_args=(--alias-to-default "$DEFAULT_VARIANT_ALIAS")
fi

# HACK: we do not need regex module here, but it is required by pre-commit hook.
# Use python instead of sed -i because macOS sed requires a backup extension argument.
$PYTHON -c "
import pathlib
p = pathlib.Path('.buildkite/scripts/generate-nightly-index.py')
p.write_text(p.read_text().replace('import regex as re', 'import re'))
"

$PYTHON .buildkite/scripts/generate-nightly-index.py \
    --version "$SUBPATH" \
    --current-objects "$obj_json" \
    --output-dir "$INDICES_OUTPUT_DIR" \
    --comment "commit $SUBPATH" \
    "${alias_args[@]}"

# Copy indices to /<commit>/ unconditionally
echo "Uploading indices to $S3_COMMIT_PREFIX"
aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "$S3_COMMIT_PREFIX"

# Copy to /nightly/ only if it is on the main branch and not a PR
if [[ "$BRANCH" == "main" && "$IS_PR" == "false" ]]; then
    echo "Uploading indices to overwrite /nightly/"
    aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/nightly/"
fi

# Re-generate and copy to /<pure_version>/ only if it does not have "dev" in the version
if [[ "$version" != *"dev"* ]]; then
    echo "Re-generating indices for /$pure_version/"
    rm -rf "${INDICES_OUTPUT_DIR:?}/*"
    mkdir -p "$INDICES_OUTPUT_DIR"
    $PYTHON .buildkite/scripts/generate-nightly-index.py \
        --version "$pure_version" \
        --wheel-dir "$SUBPATH" \
        --current-objects "$obj_json" \
        --output-dir "$INDICES_OUTPUT_DIR" \
        --comment "version $pure_version" \
        "${alias_args[@]}"
    aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/$pure_version/"
fi
