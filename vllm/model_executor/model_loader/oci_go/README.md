# OCI Go Client Library

This directory contains the Go-based OCI client library that uses [go-containerregistry](https://github.com/google/go-containerregistry) for robust OCI registry operations with full authentication support.

## Overview

The OCI Go client provides:
- Manifest pulling from OCI registries
- Blob/layer downloading
- Docker login authentication support (reads from `~/.docker/config.json`)
- Support for Docker credential helpers
- Cross-platform compatibility (Linux, macOS)

## Building

### Prerequisites

- Go 1.24 or later
- C compiler (for CGO)

### Build Instructions

Run the build script:

```bash
./build.sh
```

This will:
1. Download Go dependencies
2. Build the shared library (`liboci.so` on Linux, `liboci.dylib` on macOS)
3. Build the static archive (`liboci.a`)

### Manual Build

For Linux:
```bash
go build -buildmode=c-shared -o liboci.so oci_client.go
```

For macOS:
```bash
go build -buildmode=c-shared -o liboci.dylib oci_client.go
```

For static linking:
```bash
go build -buildmode=c-archive -o liboci.a oci_client.go
```

## Architecture

### Go Library

The `oci_client.go` file exports C-compatible functions via CGO:

- `PullManifest`: Pulls an OCI manifest for a given image reference
- `PullBlob`: Downloads a blob/layer to a specified path
- `TestAuthentication`: Tests authentication for an image reference
- `FreeString`: Frees C strings allocated by Go

### Python Wrapper

The `oci_go_client.py` file provides a Python wrapper using `ctypes` to call the Go functions:

```python
from vllm.model_executor.model_loader.oci_go_client import OciGoClient

client = OciGoClient()

# Pull manifest
manifest = client.pull_manifest("docker.io/user/image:tag")

# Download blob
client.pull_blob("docker.io/user/image:tag", "sha256:abc123...", "/path/to/output")

# Test authentication
error = client.test_authentication("docker.io/user/image:tag")
```

## Authentication

The library uses the Docker config file (`~/.docker/config.json`) for authentication. Users should run:

```bash
docker login [registry]
```

To authenticate with registries. The library automatically:
- Reads credentials from Docker config
- Uses Docker credential helpers when configured
- Supports all OCI-compliant registries

## Platform Support

The library is built for:
- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)

The shared library is included in the Python package and loaded at runtime.

## Integration with vLLM

The Go client is integrated into the `OciModelLoader` class in `oci_loader.py`, replacing the previous Python-based implementation that used the `requests` library.

Benefits:
- Proper Docker authentication support
- Better error handling
- Consistent behavior across platforms
- Static linking option for deployment

## Development

### Testing

To test the Go library:

```bash
# Build the library
./build.sh

# Run Python tests
cd /path/to/vllm
pytest tests/model_executor/model_loader/test_oci_loader.py
```

### Debugging

To enable verbose logging from the Go library, set environment variables:

```bash
export GODEBUG=http2debug=1
```

## Dependencies

The Go library depends on:
- `github.com/google/go-containerregistry` - Core OCI registry operations
- Standard Go libraries for HTTP, JSON, and file I/O

All dependencies are vendored during the build process for static linking.
