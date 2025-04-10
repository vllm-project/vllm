#!/bin/bash
# Update Dockerfile dependency graph when docker/Dockerfile changes.
# This script is designed to be used as a pre-commit hook.

set -euo pipefail

# Check if docker/Dockerfile is staged for commit
if git diff --cached --name-only | grep -q "^docker/Dockerfile$"; then
  echo "docker/Dockerfile has changed, updating dependency graph..."
  
  # Ensure target directory exists
  mkdir -p docs/source/assets/contributing
  
  # Store old image hash if exists
  if [ -f "docs/source/assets/contributing/dockerfile-stages-dependency.png" ]; then
    sha256sum docs/source/assets/contributing/dockerfile-stages-dependency.png > /tmp/old_hash.txt
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
    echo "Found Dockerfile.png in root directory"
    mv "./Dockerfile.png" docs/source/assets/contributing/dockerfile-stages-dependency.png
  else
    # Try to find it elsewhere
    DOCKERFILE_PNG=$(find . -name "Dockerfile.png" -type f | head -1)
    
    if [ -n "$DOCKERFILE_PNG" ]; then
      echo "Found generated file at: $DOCKERFILE_PNG"
      mv "$DOCKERFILE_PNG" docs/source/assets/contributing/dockerfile-stages-dependency.png
    else
      echo "Error: Could not find the generated PNG file"
      find . -name "*.png" -type f -mmin -5
      exit 1
    fi
  fi
  
  # Check if the graph has changed
  if [ -f "/tmp/old_hash.txt" ]; then
    NEW_HASH="$(sha256sum "docs/source/assets/contributing/dockerfile-stages-dependency.png")"
    OLD_HASH="$(cat "/tmp/old_hash.txt")"
    if [ "$NEW_HASH" != "$OLD_HASH" ]; then
      echo "Graph has changed, adding to commit."
      git add docs/source/assets/contributing/dockerfile-stages-dependency.png
    else
      echo "No changes in graph detected."
    fi
  else
    echo "First time generating graph, adding to commit."
    git add docs/source/assets/contributing/dockerfile-stages-dependency.png
  fi
  
  # Clean up temp file
  rm -f /tmp/old_hash.txt
fi

exit 0 
