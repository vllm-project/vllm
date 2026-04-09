#!/usr/bin/env bash

set -ex

# Upload a single wheel to S3 (rename linux -> manylinux).
# Index generation is handled separately by generate-and-upload-nightly-index.sh.

BUCKET="vllm-wheels"
SUBPATH=$BUILDKITE_COMMIT
S3_COMMIT_PREFIX="s3://$BUCKET/$SUBPATH/"

# ========= collect, rename & upload the wheel ==========

# Assume wheels are in artifacts/dist/*.whl
wheel_files=(artifacts/dist/*.whl)

# Check that exactly one wheel is found
if [[ ${#wheel_files[@]} -ne 1 ]]; then
  echo "Error: Expected exactly one wheel file in artifacts/dist/, but found ${#wheel_files[@]}"
  exit 1
fi
wheel="${wheel_files[0]}"

# default build image uses ubuntu 20.04, which corresponds to manylinux_2_31
# we also accept params as manylinux tag
# refer to https://github.com/mayeut/pep600_compliance?tab=readme-ov-file#acceptable-distros-to-build-wheels
manylinux_version="${1:-manylinux_2_31}"

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

# copy wheel to its own bucket
aws s3 cp "$wheel" "$S3_COMMIT_PREFIX"

echo "Wheel uploaded. Index generation is handled by a separate step."
