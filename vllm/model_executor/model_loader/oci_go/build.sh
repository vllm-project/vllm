#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -e

# Build script for OCI Go client library

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building OCI Go client library..."

# Download dependencies
echo "Downloading Go dependencies..."
go mod download

# Build as shared library for the current platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Building for macOS..."
    go build -buildmode=c-shared -o liboci.dylib oci_client.go
    echo "Built liboci.dylib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Building for Linux..."
    go build -buildmode=c-shared -o liboci.so oci_client.go
    echo "Built liboci.so"
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

# Also build static archive for static linking option
echo "Building static archive..."
go build -buildmode=c-archive -o liboci.a oci_client.go
echo "Built liboci.a"

echo "Build complete!"
