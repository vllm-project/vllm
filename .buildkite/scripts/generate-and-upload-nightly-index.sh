#!/usr/bin/env bash

set -ex

# Generate and upload wheel indices for all wheels in the commit directory.
# This script should run once after all wheels have been built and uploaded.

# ======== setup ========

BUCKET="vllm-wheels"
INDICES_OUTPUT_DIR="indices"
DEFAULT_VARIANT_ALIAS="cu129" # align with vLLM_MAIN_CUDA_VERSION in vllm/envs.py
PYTHON="${PYTHON_PROG:-python3}" # try to read from env var, otherwise use python3
SUBPATH=$BUILDKITE_COMMIT
S3_COMMIT_PREFIX="s3://$BUCKET/$SUBPATH/"

# detect if python3.12+ is available
has_new_python=$($PYTHON -c "print(1 if __import__('sys').version_info >= (3,12) else 0)")
if [[ "$has_new_python" -eq 0 ]]; then
    # use new python from docker
    docker pull python:3-slim
    PYTHON="docker run --rm -v $(pwd):/app -w /app python:3-slim python3"
fi

echo "Using python interpreter: $PYTHON"
echo "Python version: $($PYTHON --version)"

# ======== generate and upload indices ========

# list all wheels in the commit directory
echo "Existing wheels on S3:"
aws s3 ls "$S3_COMMIT_PREFIX"
obj_json="objects.json"
aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$SUBPATH/" --delimiter / --output json > "$obj_json"
mkdir -p "$INDICES_OUTPUT_DIR"

# call script to generate indices for all existing wheels
# these indices have relative paths that work as long as they are next to the wheel directory in s3
# i.e., the wheels are always in s3://vllm-wheels/<commit>/
# and indices can be placed in /<commit>/, or /nightly/, or /<version>/
alias_args=()
if [[ -n "$DEFAULT_VARIANT_ALIAS" ]]; then
    alias_args=(--alias-to-default "$DEFAULT_VARIANT_ALIAS")
fi

# HACK: we do not need regex module here, but it is required by pre-commit hook
# To avoid any external dependency, we simply replace it back to the stdlib re module
sed -i 's/import regex as re/import re/g' .buildkite/scripts/generate-nightly-index.py
$PYTHON .buildkite/scripts/generate-nightly-index.py --version "$SUBPATH" --current-objects "$obj_json" --output-dir "$INDICES_OUTPUT_DIR" --comment "commit $BUILDKITE_COMMIT" "${alias_args[@]}"

# copy indices to /<commit>/ unconditionally
echo "Uploading indices to $S3_COMMIT_PREFIX"
aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "$S3_COMMIT_PREFIX"

# copy to /nightly/ only if it is on the main branch and not a PR
if [[ "$BUILDKITE_BRANCH" == "main" && "$BUILDKITE_PULL_REQUEST" == "false" ]]; then
    echo "Uploading indices to overwrite /nightly/"
    aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/nightly/"
fi

# detect version from any wheel in the commit directory
# download the first wheel we find to extract version metadata
first_wheel_key=$($PYTHON -c "import json; obj=json.load(open('$obj_json')); print(next((c['Key'] for c in obj.get('Contents', []) if c['Key'].endswith('.whl')), ''))")
if [[ -z "$first_wheel_key" ]]; then
    echo "Error: No wheels found in $S3_COMMIT_PREFIX"
    exit 1
fi
first_wheel=$(basename "$first_wheel_key")
aws s3 cp "s3://$BUCKET/${first_wheel_key}" "/tmp/${first_wheel}"
version=$(unzip -p "/tmp/${first_wheel}" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
rm -f "/tmp/${first_wheel}"
echo "Version in wheel: $version"
pure_version="${version%%+*}"
echo "Pure version (without variant): $pure_version"

# re-generate and copy to /<pure_version>/ only if it does not have "dev" in the version
if [[ "$version" != *"dev"* ]]; then
    echo "Re-generating indices for /$pure_version/"
    rm -rf "${INDICES_OUTPUT_DIR:?}"
    mkdir -p "$INDICES_OUTPUT_DIR"
    # wheel-dir is overridden to be the commit directory, so that the indices point to the correct wheel path
    $PYTHON .buildkite/scripts/generate-nightly-index.py --version "$pure_version" --wheel-dir "$SUBPATH" --current-objects "$obj_json" --output-dir "$INDICES_OUTPUT_DIR" --comment "version $pure_version" "${alias_args[@]}"
    aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/$pure_version/"
fi
