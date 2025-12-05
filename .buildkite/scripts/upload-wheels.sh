#!/usr/bin/env bash

set -ex

# ======== part 0: setup ========

BUCKET="vllm-wheels"
INDICES_OUTPUT_DIR="indices"
DEFAULT_VARIANT_ALIAS="cu129" # align with vLLM_MAIN_CUDA_VERSION in vllm/envs.py
PYTHON=${PYTHON_PROG:=python3} # try to read from env var, otherwise use python3
SUBPATH=$BUILDKITE_COMMIT
S3_COMMIT_PREFIX="s3://$BUCKET/$SUBPATH/"

# detect if python3.10+ is available
has_new_python=$($PYTHON -c "print(1 if __import__('sys').version_info >= (3,12) else 0)")
if [[ "$has_new_python" -eq 0 ]]; then
    # use new python from docker
    docker pull python:3-slim
    PYTHON="docker run --rm -v $(pwd):/app -w /app python:3-slim python3"
fi

echo "Using python interpreter: $PYTHON"
echo "Python version: $($PYTHON --version)"

# ========= part 1: collect, rename & upload the wheel ==========

# Assume wheels are in artifacts/dist/*.whl
wheel_files=(artifacts/dist/*.whl)

# Check that exactly one wheel is found
if [[ ${#wheel_files[@]} -ne 1 ]]; then
  echo "Error: Expected exactly one wheel file in artifacts/dist/, but found ${#wheel_files[@]}"
  exit 1
fi
wheel="${wheel_files[0]}"

# current build image uses ubuntu 20.04, which corresponds to manylinux_2_31
# refer to https://github.com/mayeut/pep600_compliance?tab=readme-ov-file#acceptable-distros-to-build-wheels
manylinux_version="manylinux_2_31"

# Rename 'linux' to the appropriate manylinux version in the wheel filename
if [[ "$wheel" != *"linux"* ]]; then
  echo "Error: Wheel filename does not contain 'linux': $wheel"
  exit 1
fi
new_wheel="${wheel/linux/$manylinux_version}"
mv -- "$wheel" "$new_wheel"
wheel="$new_wheel"
echo "Renamed wheel to: $wheel"

# Extract the version from the wheel
version=$(unzip -p "$wheel" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
echo "Version in wheel: $version"
pure_version="${version%%+*}"
echo "Pure version (without variant): $pure_version"

# copy wheel to its own bucket
aws s3 cp "$wheel" "$S3_COMMIT_PREFIX"

# ========= part 2: generate and upload indices ==========
# generate indices for all existing wheels in the commit directory
# this script might be run multiple times if there are multiple variants being built
# so we need to guarantee there is little chance for "TOCTOU" issues
# i.e., one process is generating indices while another is uploading a new wheel
# so we need to ensure no time-consuming operations happen below

# list all wheels in the commit directory
echo "Existing wheels on S3:"
aws s3 ls "$S3_COMMIT_PREFIX"
obj_json="objects.json"
aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$SUBPATH/" --delimiter / --output json > "$obj_json"
mkdir -p "$INDICES_OUTPUT_DIR"

# call script to generate indicies for all existing wheels
# this indices have relative paths that could work as long as it is next to the wheel directory in s3
# i.e., the wheels are always in s3://vllm-wheels/<commit>/
# and indices can be placed in /<commit>/, or /nightly/, or /<version>/
if [[ ! -z "$DEFAULT_VARIANT_ALIAS" ]]; then
    alias_arg="--alias-to-default $DEFAULT_VARIANT_ALIAS"
else
    alias_arg=""
fi

# HACK: we do not need regex module here, but it is required by pre-commit hook
# To avoid any external dependency, we simply replace it back to the stdlib re module
sed -i 's/import regex as re/import re/g' .buildkite/scripts/generate-nightly-index.py
$PYTHON .buildkite/scripts/generate-nightly-index.py --version "$SUBPATH" --current-objects "$obj_json" --output-dir "$INDICES_OUTPUT_DIR" --comment "commit $BUILDKITE_COMMIT" $alias_arg

# copy indices to /<commit>/ unconditionally
echo "Uploading indices to $S3_COMMIT_PREFIX"
aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "$S3_COMMIT_PREFIX"

# copy to /nightly/ only if it is on the main branch and not a PR 
if [[ "$BUILDKITE_BRANCH" == "main" && "$BUILDKITE_PULL_REQUEST" == "false" ]]; then
    echo "Uploading indices to overwrite /nightly/"
    aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/nightly/"
fi

# copy to /<pure_version>/ only if it does not have "dev" in the version
if [[ "$version" != *"dev"* ]]; then
    echo "Uploading indices to overwrite /$pure_version/"
    aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/$pure_version/"
fi
