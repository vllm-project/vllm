#!/usr/bin/env bash

set -ex

# Assume wheels are in artifacts/dist/*.whl
wheel_files=(artifacts/dist/*.whl)

# Check that exactly one wheel is found
if [[ ${#wheel_files[@]} -ne 1 ]]; then
  echo "Error: Expected exactly one wheel file in artifacts/dist/, but found ${#wheel_files[@]}"
  exit 1
fi

# Get the single wheel file
wheel="${wheel_files[0]}"

# Rename 'linux' to 'manylinux1' in the wheel filename
new_wheel="${wheel/linux/manylinux1}"
mv -- "$wheel" "$new_wheel"
wheel="$new_wheel"

# Extract the version from the wheel
version=$(unzip -p "$wheel" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
echo "Version: $version"

# If the version contains "dev", rename it to v1.0.0.dev for consistency
if [[ $version == *dev* ]]; then
    new_version="1.0.0.dev"
    new_wheel="${wheel/$version/$new_version}"
    mv -- "$wheel" "$new_wheel"
    wheel="$new_wheel"
    version="$new_version"
fi

# Upload the wheel to S3
aws s3 cp "$wheel" "s3://vllm-wheels/$BUILDKITE_COMMIT/"
aws s3 cp "$wheel" "s3://vllm-wheels/nightly/"
aws s3 cp "$wheel" "s3://vllm-wheels/$version/"