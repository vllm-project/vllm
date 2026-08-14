#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

readonly BUCKET="vllm-wheels"
readonly PREFIX="xpu"
readonly WHEEL_URL="https://github.com/intel/intel-xpu-backend-for-triton/releases/download/v3.7.2/triton-3.7.2+xpu-py3-none-any.whl"
readonly WHEEL_SHA256="3c822f73e9870512f59a6ecf5dc305a4bcab11fa623f9ce91011f604315227e9"
readonly WHEEL_FILENAME="${WHEEL_URL##*/}"
readonly ENCODED_WHEEL_FILENAME="${WHEEL_FILENAME/+/%2B}"
readonly S3_PREFIX="s3://${BUCKET}/${PREFIX}/"
readonly COMMIT="${BUILDKITE_COMMIT:-}"
readonly DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ "$DRY_RUN" == "0" && ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "BUILDKITE_COMMIT must be a full lowercase commit hash" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Keep temporary files under the repository so the Docker Python fallback,
# which mounts the repository at /app, can access them.
work_dir=$(mktemp -d "$PWD/.xpu-triton-index.XXXXXX")
work_dir=${work_dir#"$PWD"/}
trap 'rm -rf "$work_dir"' EXIT

wheel_path="$work_dir/$WHEEL_FILENAME"
objects_path="$work_dir/objects.json"
index_output_dir="$work_dir/$PREFIX"
index_generator="$work_dir/generate-nightly-index.py"

curl --fail --location --retry 3 --output "$wheel_path" "$WHEEL_URL"

if command -v sha256sum >/dev/null 2>&1; then
    actual_sha=$(sha256sum "$wheel_path")
else
    actual_sha=$(shasum -a 256 "$wheel_path")
fi
actual_sha=${actual_sha%% *}
if [[ "$actual_sha" != "$WHEEL_SHA256" ]]; then
    echo "SHA256 mismatch for $WHEEL_FILENAME" >&2
    echo "Expected: $WHEEL_SHA256" >&2
    echo "Actual:   $actual_sha" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '{"Contents":[{"Key":"%s/%s"}]}' \
        "$PREFIX" "$WHEEL_FILENAME" > "$objects_path"
else
    aws s3 cp "$wheel_path" "$S3_PREFIX"
    aws s3api list-objects-v2 \
        --bucket "$BUCKET" \
        --prefix "$PREFIX/" \
        --delimiter / \
        --output json > "$objects_path"
fi

# Pick Python >= 3.12 locally or use the container fallback.
# shellcheck source=../lib/select-python.sh
source .buildkite/scripts/lib/select-python.sh
select_python

# The index generator only needs stdlib re here. Keep the tracked source intact.
sed 's/import regex as re/import re/' \
    .buildkite/scripts/generate-nightly-index.py > "$index_generator"

# shellcheck disable=SC2086
$PYTHON "$index_generator" \
    --version "$PREFIX" \
    --wheel-dir "$work_dir/$PREFIX" \
    --current-objects "$objects_path" \
    --output-dir "$work_dir" \
    --comment "XPU Triton shim"

grep -Fq 'href="triton/"' "$index_output_dir/index.html"
grep -Fq "href=\"../$ENCODED_WHEEL_FILENAME\"" \
    "$index_output_dir/triton/index.html"
grep -Fq "\"filename\": \"$WHEEL_FILENAME\"" \
    "$index_output_dir/triton/metadata.json"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run succeeded; generated files:"
    find "$index_output_dir" -type f -print | sort
else
    aws s3 cp --recursive "$index_output_dir/" "$S3_PREFIX"
    echo "Published XPU Triton shim index to https://wheels.vllm.ai/$PREFIX/"
    aws s3 cp "$S3_PREFIX$WHEEL_FILENAME" \
        "s3://$BUCKET/$COMMIT/$WHEEL_FILENAME"
    echo "Staged XPU Triton shim for https://wheels.vllm.ai/$COMMIT/xpu/"
fi
