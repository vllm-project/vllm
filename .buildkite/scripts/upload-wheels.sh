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

normal_wheel="$wheel" # Save the original wheel filename

# If the version contains "dev", rename it to v1.0.0.dev for consistency
if [[ $version == *dev* ]]; then
    suffix="${version##*.}"
    if [[ $suffix == cu* ]]; then
        new_version="1.0.0.dev+${suffix}"
    else
        new_version="1.0.0.dev"
    fi
    new_wheel="${wheel/$version/$new_version}"
    # use cp to keep both files in the artifacts directory
    cp -- "$wheel" "$new_wheel"
    wheel="$new_wheel"
    version="$new_version"
fi

# Upload the wheel to S3
python3 .buildkite/generate_index.py --wheel "$normal_wheel"

# generate index for this commit
aws s3 cp "$wheel" "s3://vllm-wheels/$BUILDKITE_COMMIT/"
aws s3 cp "$normal_wheel" "s3://vllm-wheels/$BUILDKITE_COMMIT/"

if [[ $normal_wheel == *"cu118"* ]]; then
    # if $normal_wheel matches cu118, do not upload the index.html
    echo "Skipping index files for cu118 wheels"
elif [[ $normal_wheel == *"cu126"* ]]; then
    # if $normal_wheel matches cu126, do not upload the index.html
    echo "Skipping index files for cu126 wheels"
else
    # only upload index.html for cu128 wheels (default wheels)
    aws s3 cp index.html "s3://vllm-wheels/$BUILDKITE_COMMIT/vllm/index.html"
    aws s3 cp "s3://vllm-wheels/nightly/index.html" "s3://vllm-wheels/$BUILDKITE_COMMIT/index.html"
fi

# generate index for nightly
aws s3 cp "$wheel" "s3://vllm-wheels/nightly/"
aws s3 cp "$normal_wheel" "s3://vllm-wheels/nightly/"

if [[ $normal_wheel == *"cu118"* ]]; then
    # if $normal_wheel matches cu118, do not upload the index.html
    echo "Skipping index files for cu118 wheels"
elif [[ $normal_wheel == *"cu126"* ]]; then
    # if $normal_wheel matches cu126, do not upload the index.html
    echo "Skipping index files for cu126 wheels"
else
    # only upload index.html for cu128 wheels (default wheels)
    aws s3 cp index.html "s3://vllm-wheels/nightly/vllm/index.html"
fi

aws s3 cp "$wheel" "s3://vllm-wheels/$version/"
aws s3 cp index.html "s3://vllm-wheels/$version/vllm/index.html"
