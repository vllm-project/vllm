#!/bin/bash
# Update Dockerfile dependency graph when docker/Dockerfile changes.
# This script is designed to be used as a pre-commit hook.

set -euo pipefail

# Accept file paths as arguments
FILES=("$@")

# Check if docker/Dockerfile is among the provided files
if printf '%s\n' "${FILES[@]}" | grep -q "^docker/Dockerfile$"; then
  echo "docker/Dockerfile has changed, attempting to update dependency graph..."

  # Check if Docker is installed and running
  if ! command -v docker &> /dev/null; then
    echo "Warning: Docker command not found. Skipping Dockerfile graph update."
    echo "Please install Docker to automatically update the graph: https://docs.docker.com/get-docker/"
    exit 0
  fi
  if ! docker info &> /dev/null; then
    echo "Warning: Docker daemon is not running. Skipping Dockerfile graph update."
    echo "Please start Docker to automatically update the graph."
    exit 0
  fi

  # Define the target file path
  TARGET_GRAPH_FILE="docs/assets/contributing/dockerfile-stages-dependency.png"

  # Ensure target directory exists
  mkdir -p "$(dirname "$TARGET_GRAPH_FILE")"

  # Store old image hash in a variable if the file exists
  OLD_HASH=""
  if [ -f "$TARGET_GRAPH_FILE" ]; then
    OLD_HASH=$(sha256sum "$TARGET_GRAPH_FILE")
  fi
  
  # Generate Dockerfile graph
  echo "Running dockerfilegraph tool..."

  # Create a temporary directory for output
  TEMP_OUTPUT_DIR=$(mktemp -d)
  trap "rm -rf '$TEMP_OUTPUT_DIR'" EXIT

  # Copy the Dockerfile to temp directory and ensure it's readable
  chmod 777 "$TEMP_OUTPUT_DIR"
  cp docker/Dockerfile "$TEMP_OUTPUT_DIR/"
  chmod 644 "$TEMP_OUTPUT_DIR/Dockerfile"

  # Run dockerfilegraph with output to temp directory
  docker run \
    --rm \
    --workdir /workspace \
    --volume "$TEMP_OUTPUT_DIR":/workspace \
    --security-opt label=disable \
    ghcr.io/patrickhoefler/dockerfilegraph:alpine \
    --output png \
    --dpi 200 \
    --max-label-length 50 \
    --filename Dockerfile \
    --legend

  # Move the generated file to the target location
  if [ -f "$TEMP_OUTPUT_DIR/Dockerfile.png" ]; then
    echo "Successfully generated Dockerfile.png"
    mv "$TEMP_OUTPUT_DIR/Dockerfile.png" "$TARGET_GRAPH_FILE"
  else
    echo "Error: Could not find the generated PNG file in temp directory"
    ls -la "$TEMP_OUTPUT_DIR"
    exit 1
  fi

  # Check if the graph has changed
  NEW_HASH=$(sha256sum "$TARGET_GRAPH_FILE")
  if [ "$NEW_HASH" != "$OLD_HASH" ]; then
    echo "Graph has changed. Please stage the updated file: $TARGET_GRAPH_FILE"
    exit 1
  else
    echo "No changes in graph detected."
  fi
fi

exit 0
