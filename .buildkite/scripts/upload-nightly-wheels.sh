#!/usr/bin/env bash

set -ex

# Upload a single wheel to S3, after detecting and applying the appropriate
# manylinux platform tag with auditwheel.
# Index generation is handled separately by generate-and-upload-nightly-index.sh.

# shellcheck source=lib/manylinux.sh
source .buildkite/scripts/lib/manylinux.sh

BUCKET="vllm-wheels"
SUBPATH=$BUILDKITE_COMMIT
S3_COMMIT_PREFIX="s3://$BUCKET/$SUBPATH/"

# ========= locate the wheel ==========

# Assume wheels are in artifacts/dist/*.whl
wheel_files=(artifacts/dist/*.whl)

# Check that exactly one wheel is found
if [[ ${#wheel_files[@]} -ne 1 ]]; then
  echo "Error: Expected exactly one wheel file in artifacts/dist/, but found ${#wheel_files[@]}"
  exit 1
fi
wheel="${wheel_files[0]}"

# ========= detect manylinux tag and rename ==========

wheel="$(apply_manylinux_tag "$wheel")"
echo "Renamed wheel to: $wheel"

# Extract the version from the wheel
version=$(unzip -p "$wheel" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
echo "Version in wheel: $version"

# copy wheel to its own bucket
aws s3 cp "$wheel" "$S3_COMMIT_PREFIX"

echo "Wheel uploaded. Index generation is handled by a separate step."
