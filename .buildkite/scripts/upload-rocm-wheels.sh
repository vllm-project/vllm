#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Upload ROCm wheels to S3 with proper index generation
#
# Required environment variables:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or IAM role)
#   S3_BUCKET (default: vllm-wheels)
#
# S3 path structure:
#   s3://vllm-wheels/rocm/{commit}/     - All wheels for this commit
#   s3://vllm-wheels/rocm/nightly/      - Index pointing to latest nightly
#   s3://vllm-wheels/rocm/{version}/    - Index for release versions

set -ex

# ======== Configuration ========
BUCKET="${S3_BUCKET:-vllm-wheels}"
ROCM_SUBPATH="rocm/${BUILDKITE_COMMIT}"
S3_COMMIT_PREFIX="s3://$BUCKET/$ROCM_SUBPATH/"
INDICES_OUTPUT_DIR="rocm-indices"
PYTHON="${PYTHON_PROG:-python3}"

# ROCm uses manylinux_2_35 (Ubuntu 22.04 based)
MANYLINUX_VERSION="manylinux_2_35"

echo "========================================"
echo "ROCm Wheel Upload Configuration"
echo "========================================"
echo "S3 Bucket: $BUCKET"
echo "S3 Path: $ROCM_SUBPATH"
echo "Commit: $BUILDKITE_COMMIT"
echo "Branch: $BUILDKITE_BRANCH"
echo "========================================"

# ======== Part 0: Setup Python ========

# Detect if python3.12+ is available
has_new_python=$($PYTHON -c "print(1 if __import__('sys').version_info >= (3,12) else 0)" 2>/dev/null || echo 0)
if [[ "$has_new_python" -eq 0 ]]; then
    # Use new python from docker
    # Use --user to ensure files are created with correct ownership (not root)
    docker pull python:3-slim
    PYTHON="docker run --rm --user $(id -u):$(id -g) -v $(pwd):/app -w /app python:3-slim python3"
fi

echo "Using python interpreter: $PYTHON"
echo "Python version: $($PYTHON --version)"

# ======== Part 1: Collect and prepare wheels ========

# Collect all wheels
mkdir -p all-rocm-wheels
cp artifacts/rocm-base-wheels/*.whl all-rocm-wheels/ 2>/dev/null || true
cp artifacts/rocm-vllm-wheel/*.whl all-rocm-wheels/ 2>/dev/null || true

WHEEL_COUNT=$(ls all-rocm-wheels/*.whl 2>/dev/null | wc -l)
echo "Total wheels to upload: $WHEEL_COUNT"

if [ "$WHEEL_COUNT" -eq 0 ]; then
    echo "ERROR: No wheels found to upload!"
    exit 1
fi

# Rename linux to manylinux in wheel filenames
for wheel in all-rocm-wheels/*.whl; do
    if [[ "$wheel" == *"linux"* ]] && [[ "$wheel" != *"manylinux"* ]]; then
        new_wheel="${wheel/linux/$MANYLINUX_VERSION}"
        mv -- "$wheel" "$new_wheel"
        echo "Renamed: $(basename "$wheel") -> $(basename "$new_wheel")"
    fi
done

echo ""
echo "Wheels to upload:"
ls -lh all-rocm-wheels/

# ======== Part 2: Upload wheels to S3 ========

echo ""
echo "Uploading wheels to $S3_COMMIT_PREFIX"
for wheel in all-rocm-wheels/*.whl; do
    aws s3 cp "$wheel" "$S3_COMMIT_PREFIX"
done

# ======== Part 3: Generate and upload indices ========

# List existing wheels in commit directory
echo ""
echo "Generating indices..."
obj_json="rocm-objects.json"
aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$ROCM_SUBPATH/" --delimiter / --output json > "$obj_json"

mkdir -p "$INDICES_OUTPUT_DIR"

# Use the existing generate-nightly-index.py
# HACK: Replace regex module with stdlib re (same as CUDA script)
sed -i 's/import regex as re/import re/g' .buildkite/scripts/generate-nightly-index.py

$PYTHON .buildkite/scripts/generate-nightly-index.py \
    --version "$ROCM_SUBPATH" \
    --current-objects "$obj_json" \
    --output-dir "$INDICES_OUTPUT_DIR" \
    --comment "ROCm commit $BUILDKITE_COMMIT"

# Upload indices to commit directory
echo "Uploading indices to $S3_COMMIT_PREFIX"
aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "$S3_COMMIT_PREFIX"

# Update rocm/nightly/ if on main branch and not a PR
if [[ "$BUILDKITE_BRANCH" == "main" && "$BUILDKITE_PULL_REQUEST" == "false" ]] || [[ "$NIGHTLY" == "1" ]]; then
    echo "Updating rocm/nightly/ index..."
    aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/rocm/nightly/"
fi

# Extract version from vLLM wheel and update version-specific index
VLLM_WHEEL=$(ls all-rocm-wheels/vllm*.whl 2>/dev/null | head -1)
if [ -n "$VLLM_WHEEL" ]; then
    VERSION=$(unzip -p "$VLLM_WHEEL" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
    echo "Version in wheel: $VERSION"
    PURE_VERSION="${VERSION%%+*}"
    PURE_VERSION="${PURE_VERSION%%.rocm}"
    echo "Pure version: $PURE_VERSION"

    if [[ "$VERSION" != *"dev"* ]]; then
        echo "Updating rocm/$PURE_VERSION/ index..."
        aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/rocm/$PURE_VERSION/"
    fi
fi

# ======== Part 4: Summary ========

echo ""
echo "========================================"
echo "ROCm Wheel Upload Complete!"
echo "========================================"
echo ""
echo "Wheels available at:"
echo "  s3://$BUCKET/$ROCM_SUBPATH/"
echo ""
echo "Install command (by commit):"
echo "  pip install vllm --extra-index-url https://${BUCKET}.s3.amazonaws.com/$ROCM_SUBPATH/"
echo ""
if [[ "$BUILDKITE_BRANCH" == "main" ]] || [[ "$NIGHTLY" == "1" ]]; then
    echo "Install command (nightly):"
    echo "  pip install vllm --extra-index-url https://${BUCKET}.s3.amazonaws.com/rocm/nightly/"
fi
echo ""
echo "Wheel count: $WHEEL_COUNT"
echo "========================================"
