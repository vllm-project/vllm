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
  docker run \
    --rm \
    --user "$(id -u):$(id -g)" \
    --workdir /workspace \
    --volume "$(pwd)":/workspace \
    ghcr.io/patrickhoefler/dockerfilegraph:alpine \
    --output png \
    --dpi 200 \
    --max-label-length 50 \
    --filename docker/Dockerfile \
    --legend
  
  echo "Finding generated PNG file..."
  # Check for Dockerfile.png in the root directory (most likely location)
  if [ -f "./Dockerfile.png" ]; then
    echo "Found generated file at: ./Dockerfile.png"
    mv "./Dockerfile.png" "$TARGET_GRAPH_FILE"
  else
    # Try to find it elsewhere
    DOCKERFILE_PNG=$(find . -name "Dockerfile.png" -type f | head -1)
    
    if [ -n "$DOCKERFILE_PNG" ]; then
      echo "Found generated file at: $DOCKERFILE_PNG"
      mv "$DOCKERFILE_PNG" "$TARGET_GRAPH_FILE"
    else
      echo "Error: Could not find the generated PNG file"
      find . -name "*.png" -type f -mmin -5
      exit 1
    fi
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
