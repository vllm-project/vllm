#!/usr/bin/env bash

set -ex

BUCKET="vllm-wheels"
INDICES_OUTPUT_DIR="indices"
DEFAULT_VARIANT_ALIAS="cu129" # align with vLLM_MAIN_CUDA_VERSION in vllm/envs.py

# Assume wheels are in artifacts/dist/*.whl
wheel_files=(artifacts/dist/*.whl)

# Check that exactly one wheel is found
if [[ ${#wheel_files[@]} -ne 1 ]]; then
  echo "Error: Expected exactly one wheel file in artifacts/dist/, but found ${#wheel_files[@]}"
  exit 1
fi

# Get the single wheel file
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
aws s3 cp "$wheel" "s3://$BUCKET/$BUILDKITE_COMMIT/"

# list all wheels in the commit directory
echo "Existing wheels on S3:"
aws s3 ls "s3://$BUCKET/$BUILDKITE_COMMIT/"
obj_json="$(mktemp).json"
aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$BUILDKITE_COMMIT/" --delimiter / --output json > "$obj_json"
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
python3 .buildkite/scripts/generate-nightly-index.py --version "$BUILDKITE_COMMIT" --current-objects "$obj_json" --output-dir "$INDICES_OUTPUT_DIR" "$alias_arg"

# copy indices to /<commit>/ unconditionally
aws s3 cp --recursive "$INDICES_OUTPUT_DIR/" "s3://$BUCKET/$BUILDKITE_COMMIT/"

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
