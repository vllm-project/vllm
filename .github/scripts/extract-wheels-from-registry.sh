#!/usr/bin/env bash
set -euo pipefail

# Script to extract wheels from Docker registry without downloading full image
# Uses crane (part of go-containerregistry) to download only specific layers
# Usage: ./extract-wheels-from-registry.sh <image-name> <output-dir>
# Example: ./extract-wheels-from-registry.sh us-central1-docker.pkg.dev/cohere-artifacts/cohere/vllm-rocm:abc123 ./wheels

IMAGE_NAME="${1:?Error: Image name required}"
OUTPUT_DIR="${2:-./wheels}"

echo "========================================="
echo "Extracting wheels from Docker registry"
echo "Image: $IMAGE_NAME"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Check if crane is installed
if ! command -v crane &> /dev/null; then
    echo "Installing crane..."
    # Install crane from go-containerregistry
    VERSION="v0.19.1"
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    case $ARCH in
        x86_64) ARCH="x86_64" ;;
        aarch64) ARCH="arm64" ;;
        arm64) ARCH="arm64" ;;
        *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
    esac

    TARBALL="go-containerregistry_${OS}_${ARCH}.tar.gz"
    URL="https://github.com/google/go-containerregistry/releases/download/${VERSION}/${TARBALL}"

    echo "Downloading crane from $URL"
    curl -sL "$URL" | tar -xz crane
    sudo mv crane /usr/local/bin/
    crane version
fi

# Create temporary directory
TEMP_DIR="$(mktemp -d)"
cleanup() {
    echo "Cleaning up..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Export image filesystem
echo "Exporting amd64 image filesystem (this downloads only necessary layers)..."
crane export --platform=linux/amd64 "$IMAGE_NAME" - | tar -xf - -C "$TEMP_DIR" app/cohere/dist

# check if there is an ARM image
echo "Exporting arm64 wheels (best-effort)..."
if crane export --platform=linux/arm64 "$IMAGE_NAME" - \
  | tar -xf - -C "$TEMP_DIR" app/cohere/dist; then
  echo "✅ arm64 export succeeded"
else
  echo "⚠️ arm64 export failed (no arm64 variant of image found)"
fi

# Check if wheels were found
if [ ! -d "$TEMP_DIR/app/cohere/dist" ]; then
    echo "Error: No wheels found at /app/cohere/dist in image"
    echo "Available paths:"
    crane export "$IMAGE_NAME" - | tar -tf - | grep -E '\.(whl|tar\.gz)$' || echo "No Python packages found"
    exit 1
fi

# Copy wheels to output directory
mkdir -p "$OUTPUT_DIR"
cp "$TEMP_DIR/app/cohere/dist"/*.whl "$OUTPUT_DIR/" 2>/dev/null || {
    echo "Error: No .whl files found"
    exit 1
}

WHEEL_COUNT=$(ls -1 "$OUTPUT_DIR"/*.whl 2>/dev/null | wc -l)
echo "========================================="
echo "✅ Extracted $WHEEL_COUNT wheel(s) to $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
echo "========================================="
