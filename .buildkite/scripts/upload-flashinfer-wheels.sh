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

echo "Processing FlashInfer wheel: $wheel"

# Rename 'linux' to 'manylinux1' in the wheel filename for compatibility
new_wheel="${wheel/linux/manylinux1}"
if [[ "$wheel" != "$new_wheel" ]]; then
  mv -- "$wheel" "$new_wheel"
  wheel="$new_wheel"
  echo "Renamed wheel to: $wheel"
fi

# Extract the version from the wheel
version=$(unzip -p "$wheel" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
echo "FlashInfer version: $version"

# Upload the wheel to S3 under flashinfer directory
aws s3 cp "$wheel" "s3://vllm-wheels/flashinfer/"

# Generate simple index.html for the package (following pip index pattern)
wheel_name=$(basename "$wheel")
cat > flashinfer_index.html << EOF
<!DOCTYPE html>
<html>
<head><title>Links for flashinfer-python</title></head>
<body>
<h1>Links for flashinfer-python</h1>
<a href="$wheel_name">$wheel_name</a><br/>
</body>
</html>
EOF

aws s3 cp flashinfer_index.html "s3://vllm-wheels/flashinfer/index.html"

# Clean up
rm -f flashinfer_index.html

echo "Successfully uploaded FlashInfer wheel $wheel_name (version $version)"