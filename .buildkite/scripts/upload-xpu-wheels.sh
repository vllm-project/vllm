#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Upload XPU wheels to S3 with proper index generation.
#
# Two kinds of wheels end up in the per-commit XPU index:
#   1. The vLLM XPU wheel built in this CI run (artifacts/dist/*.whl).
#   2. Redistributed upstream wheels that vLLM ships from its own index
#      (because they are not on PyPI or conflict with PyPI names):
#        - triton (Intel XPU shim, name 'triton', version '3.7.1+xpu')
#        - vllm-xpu-kernels
#      These are uploaded manually (out of band) to
#      ``s3://vllm-wheels/xpu/redist/`` and copied into every per-commit
#      directory at upload time so the per-commit index is self-contained
#      and 'pip install vllm --extra-index-url https://wheels.vllm.ai/xpu/<commit>/'
#      Just Works without a separate index.
#
# Required environment variables:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or IAM role)
#   S3_BUCKET (default: vllm-wheels)
#
# S3 path structure:
#   s3://vllm-wheels/xpu/redist/                - Hand-uploaded shim wheels
#   s3://vllm-wheels/xpu/{commit}/              - Per-commit wheel + index
#   s3://vllm-wheels/xpu/nightly/               - Index pointing to latest main
#   s3://vllm-wheels/xpu/{version}/             - Index for release versions

set -ex

# ======== Configuration ========
BUCKET="${S3_BUCKET:-vllm-wheels}"
XPU_SUBPATH="xpu/${BUILDKITE_COMMIT}"
S3_COMMIT_PREFIX="s3://$BUCKET/$XPU_SUBPATH/"
S3_REDIST_PREFIX="s3://$BUCKET/xpu/redist/"
INDICES_OUTPUT_DIR="xpu-indices"

echo "========================================"
echo "XPU Wheel Upload Configuration"
echo "========================================"
echo "S3 Bucket: $BUCKET"
echo "S3 Path: $XPU_SUBPATH"
echo "Redist source: $S3_REDIST_PREFIX"
echo "Commit: $BUILDKITE_COMMIT"
echo "Branch: $BUILDKITE_BRANCH"
echo "========================================"

# ======== Part 0: Setup Python and helpers ========

# Pick a Python interpreter for index generation.
# shellcheck source=lib/select-python.sh
source .buildkite/scripts/lib/select-python.sh
select_python

# Set up auditwheel-in-a-container for the manylinux retagging step.
# shellcheck source=lib/manylinux.sh
source .buildkite/scripts/lib/manylinux.sh

# ======== Part 1: Collect and prepare wheels ========

mkdir -p all-xpu-wheels

# vLLM wheel built in this run.
if compgen -G "artifacts/dist/*.whl" > /dev/null; then
    cp artifacts/dist/*.whl all-xpu-wheels/
else
    echo "ERROR: No wheel found in artifacts/dist/" >&2
    exit 1
fi

# Pull redistributed upstream wheels (triton-xpu shim, vllm-xpu-kernels).
# These are uploaded manually to s3://$BUCKET/xpu/redist/; if the directory
# is empty the per-commit index will only contain the vLLM wheel.
mkdir -p xpu-redist
if aws s3 ls "$S3_REDIST_PREFIX" 2>/dev/null | grep -q "\.whl"; then
    aws s3 sync "$S3_REDIST_PREFIX" xpu-redist/ --exclude "*" --include "*.whl"
    cp xpu-redist/*.whl all-xpu-wheels/ 2>/dev/null || true
else
    echo "WARNING: No redistributed wheels found at $S3_REDIST_PREFIX"
    echo "         Upload triton-3.7.1+xpu and vllm_xpu_kernels wheels there"
    echo "         before users can install vllm from this index."
fi

WHEEL_COUNT=$(find all-xpu-wheels -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l)
echo "Total wheels to upload: $WHEEL_COUNT"

# Detect manylinux platform tag and rename in place for any wheel that
# still carries the generic ``linux_<arch>`` tag (mirrors ROCm flow).
for wheel in all-xpu-wheels/*.whl; do
    if [[ "$wheel" == *"linux"* ]] && [[ "$wheel" != *"manylinux"* ]]; then
        new_wheel="$(apply_manylinux_tag "$wheel")"
        echo "Renamed: $(basename "$wheel") -> $(basename "$new_wheel")"
    fi
done

echo ""
echo "Wheels to upload:"
ls -lh all-xpu-wheels/

# ======== Part 2: Upload wheels to S3 ========

echo ""
echo "Uploading wheels to $S3_COMMIT_PREFIX"
for wheel in all-xpu-wheels/*.whl; do
    aws s3 cp "$wheel" "$S3_COMMIT_PREFIX"
done

# ======== Part 3: Generate and upload indices ========

echo ""
echo "Generating indices..."
obj_json="xpu-objects.json"
aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$XPU_SUBPATH/" --delimiter / --output json > "$obj_json"

mkdir -p "$INDICES_OUTPUT_DIR"

# HACK: Replace regex module with stdlib re (same as CUDA/ROCm scripts).
sed -i 's/import regex as re/import re/g' .buildkite/scripts/generate-nightly-index.py

$PYTHON .buildkite/scripts/generate-nightly-index.py \
    --version "$XPU_SUBPATH" \
    --current-objects "$obj_json" \
    --output-dir "$INDICES_OUTPUT_DIR" \
    --comment "XPU commit $BUILDKITE_COMMIT"

# Upload indices to commit directory.
echo "Uploading indices to $S3_COMMIT_PREFIX"
aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "$S3_COMMIT_PREFIX"

# Update xpu/nightly/ if on main branch and not a PR.
if [[ "$BUILDKITE_BRANCH" == "main" && "$BUILDKITE_PULL_REQUEST" == "false" ]] || [[ "$NIGHTLY" == "1" ]]; then
    echo "Updating xpu/nightly/ index..."
    aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/xpu/nightly/"
fi

# Extract version from vLLM wheel and update version-specific index.
VLLM_WHEEL=$(find all-xpu-wheels -maxdepth 1 -name 'vllm-*.whl' 2>/dev/null | head -1)
if [ -n "$VLLM_WHEEL" ]; then
    VERSION=$(unzip -p "$VLLM_WHEEL" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
    echo "Version in wheel: $VERSION"
    PURE_VERSION="${VERSION%%+*}"
    PURE_VERSION="${PURE_VERSION%%.xpu}"
    echo "Pure version: $PURE_VERSION"

    if [[ "$VERSION" != *"dev"* ]]; then
        echo "Updating xpu/$PURE_VERSION/ index..."
        aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/xpu/$PURE_VERSION/"
    fi
fi

# ======== Part 4: Summary ========

echo ""
echo "========================================"
echo "XPU Wheel Upload Complete!"
echo "========================================"
echo ""
echo "Wheels available at:"
echo "  s3://$BUCKET/$XPU_SUBPATH/"
echo ""
echo "Install command (by commit):"
echo "  pip install vllm --extra-index-url https://${BUCKET}.s3.amazonaws.com/$XPU_SUBPATH/"
echo ""
if [[ "$BUILDKITE_BRANCH" == "main" ]] || [[ "$NIGHTLY" == "1" ]]; then
    echo "Install command (nightly):"
    echo "  pip install vllm --extra-index-url https://${BUCKET}.s3.amazonaws.com/xpu/nightly/"
fi
echo ""
echo "Wheel count: $WHEEL_COUNT"
echo "========================================"
