#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Upload ROCm wheels to S3 with proper index generation
#
# Required environment variables:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or IAM role)
#   S3_BUCKET (default: vllm-wheels)
# Optional:
#   ROCM_WHEEL_DIRS      - local wheel dirs to upload
#                          (default: artifacts/rocm-base-wheels artifacts/rocm-vllm-wheel)
#   ROCM_WHEEL_SUBDIR    - store wheels under rocm/{commit}/<subdir>/ instead of
#                          rocm/{commit}/, so several ROCm stacks can share a commit
#   ROCM_EXTERNAL_LINKS  - file of absolute URLs to list in the index without hosting
#
# S3 path structure:
#   s3://vllm-wheels/rocm/{commit}/[<subdir>/]  - All wheels for this commit
#   s3://vllm-wheels/rocm/{commit}/<variant>/   - Index per ROCm variant (e.g. rocm100)
#   s3://vllm-wheels/rocm/nightly/<variant>/    - Index pointing to latest nightly
#   s3://vllm-wheels/rocm/{version}/<variant>/  - Index for release versions
# The top-level index.html at each prefix lists every variant present, since
# several ROCm stacks publish into the same prefixes.

set -ex

# ======== Configuration ========
BUCKET="${S3_BUCKET:-vllm-wheels}"
ROCM_SUBPATH="rocm/${BUILDKITE_COMMIT}"
WHEEL_DIRS="${ROCM_WHEEL_DIRS:-artifacts/rocm-base-wheels artifacts/rocm-vllm-wheel}"
WHEEL_SUBPATH="$ROCM_SUBPATH${ROCM_WHEEL_SUBDIR:+/$ROCM_WHEEL_SUBDIR}"
S3_WHEEL_PREFIX="s3://$BUCKET/$WHEEL_SUBPATH/"
INDICES_OUTPUT_DIR="rocm-indices"

echo "========================================"
echo "ROCm Wheel Upload Configuration"
echo "========================================"
echo "S3 Bucket: $BUCKET"
echo "S3 Path: $ROCM_SUBPATH"
echo "Commit: $BUILDKITE_COMMIT"
echo "Branch: $BUILDKITE_BRANCH"
echo "========================================"

# ======== Part 0: Setup Python and helpers ========

# Pick a Python interpreter for index generation -- local if recent
# enough, else a one-shot docker fallback.
# shellcheck source=lib/select-python.sh
source .buildkite/scripts/lib/select-python.sh
select_python

# Set up auditwheel-in-a-container for the manylinux retagging step.
# Distinct from select_python: ``manylinux.sh`` deliberately pins both
# the Python and auditwheel versions (the script reads auditwheel
# internals) and so always runs in a known-good container regardless
# of what's on the agent.
# shellcheck source=lib/manylinux.sh
source .buildkite/scripts/lib/manylinux.sh

# ======== Part 1: Collect and prepare wheels ========

# Collect all wheels
rm -rf all-rocm-wheels "$INDICES_OUTPUT_DIR"
mkdir -p all-rocm-wheels
for dir in $WHEEL_DIRS; do
    cp "$dir"/*.whl all-rocm-wheels/ 2>/dev/null || true
done

WHEEL_COUNT=$(find all-rocm-wheels -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l)
echo "Total wheels to upload: $WHEEL_COUNT"

if [ "$WHEEL_COUNT" -eq 0 ]; then
    echo "ERROR: No wheels found to upload!"
    exit 1
fi

# Detect the appropriate manylinux platform tag for any wheel that still
# carries the generic ``linux_<arch>`` tag, and rename it in place. We use
# auditwheel via ``apply_manylinux_tag`` (see lib/manylinux.sh) rather than
# a hard-coded ``manylinux_2_35`` string so that the label tracks the actual
# glibc symbol versions used by the binaries (and stays correct if the
# rocm_base image is rebased).
#
# The ``linux``/``manylinux`` filter below skips both pre-tagged wheels
# (e.g. upstream torch) and pure-Python ``-any.whl`` wheels.
for wheel in all-rocm-wheels/*.whl; do
    if [[ "$wheel" == *"linux"* ]] && [[ "$wheel" != *"manylinux"* ]]; then
        new_wheel="$(apply_manylinux_tag "$wheel")"
        echo "Renamed: $(basename "$wheel") -> $(basename "$new_wheel")"
    fi
done

echo ""
echo "Wheels to upload:"
ls -lh all-rocm-wheels/

# ======== Part 2: Upload wheels to S3 ========

echo ""
echo "Uploading wheels to $S3_WHEEL_PREFIX"
for wheel in all-rocm-wheels/*.whl; do
    aws s3 cp "$wheel" "$S3_WHEEL_PREFIX"
done

# ======== Part 3: Generate and upload indices ========

# List existing wheels in commit directory
echo ""
echo "Generating indices..."
obj_json="rocm-objects.json"
aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$WHEEL_SUBPATH/" --delimiter / --output json > "$obj_json"

mkdir -p "$INDICES_OUTPUT_DIR"

# Use the existing generate-nightly-index.py
# HACK: Replace regex module with stdlib re (same as CUDA script)
sed -i 's/import regex as re/import re/g' .buildkite/scripts/generate-nightly-index.py

INDEX_ARGS=(--version "$ROCM_SUBPATH" --wheel-dir "${WHEEL_SUBPATH#rocm/}")
if [[ -n "${ROCM_EXTERNAL_LINKS:-}" ]]; then
    INDEX_ARGS+=(--external-links "$ROCM_EXTERNAL_LINKS")
fi
$PYTHON .buildkite/scripts/generate-nightly-index.py \
    "${INDEX_ARGS[@]}" \
    --current-objects "$obj_json" \
    --output-dir "$INDICES_OUTPUT_DIR" \
    --comment "ROCm commit $BUILDKITE_COMMIT"

# Upload this build's variant indices under an S3 prefix (e.g. rocm/nightly), then
# rebuild that prefix's top-level project list from every variant now present.
publish_indices() {
    local prefix="$1"
    local listing variants merged
    echo "Uploading indices to s3://$BUCKET/$prefix/"
    aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/$prefix/" --exclude "index.html"
    listing=$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$prefix/" \
        --delimiter / --query 'CommonPrefixes[].Prefix' --output text)
    variants=$(tr '\t' '\n' <<< "$listing" | sed -n "s|^$prefix/\(rocm[0-9]*\)/$|\1|p" | sort -u)
    merged=$(mktemp)
    {
        echo '<!DOCTYPE html>'
        echo '<html>'
        echo '  <meta name="pypi:repository-version" content="1.0">'
        echo '  <body>'
        for variant in $variants; do
            echo "    <a href=\"$variant/\">$variant/</a><br/>"
        done
        echo '  </body>'
        echo '</html>'
    } > "$merged"
    echo "Variants under $prefix/: $(echo "$variants" | tr '\n' ' ')"
    aws s3 cp "$merged" "s3://$BUCKET/$prefix/index.html"
    rm -f "$merged"
}

publish_indices "$ROCM_SUBPATH"

# Only scheduled nightly builds should update the moving nightly index.
if [[ "${NIGHTLY:-0}" == "1" ]]; then
    publish_indices "rocm/nightly"
fi

# Extract version from vLLM wheel and update version-specific index
VLLM_WHEEL=$(find all-rocm-wheels -maxdepth 1 -name 'vllm*.whl' 2>/dev/null | head -1)
if [ -n "$VLLM_WHEEL" ]; then
    VERSION=$(unzip -p "$VLLM_WHEEL" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
    echo "Version in wheel: $VERSION"
    PURE_VERSION="${VERSION%%+*}"
    PURE_VERSION="${PURE_VERSION%%.rocm}"
    echo "Pure version: $PURE_VERSION"

    if [[ "$VERSION" != *"dev"* ]]; then
        publish_indices "rocm/$PURE_VERSION"
    fi
fi

# ======== Part 4: Summary ========

echo ""
echo "========================================"
echo "ROCm Wheel Upload Complete!"
echo "========================================"
echo ""
echo "Wheels available at:"
echo "  s3://$BUCKET/$WHEEL_SUBPATH/"
echo ""
echo "Install command (by commit):"
echo "  pip install vllm --extra-index-url https://${BUCKET}.s3.amazonaws.com/$ROCM_SUBPATH/"
echo ""
if [[ "${NIGHTLY:-0}" == "1" ]]; then
    echo "Install command (nightly):"
    echo "  pip install vllm --extra-index-url https://${BUCKET}.s3.amazonaws.com/rocm/nightly/"
fi
echo ""
echo "Wheel count: $WHEEL_COUNT"
echo "========================================"
