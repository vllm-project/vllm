# Go Containerregistry Integration Notes

This document describes the integration of the `go-containerregistry` library into vLLM for OCI registry operations.

## Overview

vLLM now uses Google's `go-containerregistry` library (the same library used by tools like `crane` and `ko`) to handle OCI registry operations. This replaces the previous Python-based implementation that used the `requests` library.

## Benefits

1. **Docker Login Support**: Full support for `docker login` authentication via Docker config files
2. **Credential Helpers**: Supports Docker credential helpers for cloud providers (AWS ECR, GCR, ACR)
3. **Cross-Platform**: Works on Linux and macOS with static linking
4. **Robust**: Battle-tested library used by many container tools
5. **Standards Compliant**: Fully OCI-compliant with proper authentication flows

## Architecture

### Components

```
vllm/model_executor/model_loader/
├── oci_loader.py           # Main OCI loader (uses Go client)
├── oci_go_client.py        # Python wrapper for Go library (ctypes)
└── oci_go/
    ├── oci_client.go       # Go implementation using go-containerregistry
    ├── go.mod              # Go dependencies
    ├── build.sh            # Build script
    ├── liboci.so/.dylib    # Compiled shared library (platform-specific)
    ├── README.md           # Go client documentation
    └── test_client.py      # Standalone tests
```

### Data Flow

```
Python Application
    ↓
OciModelLoader (oci_loader.py)
    ↓
OciGoClient (oci_go_client.py - ctypes wrapper)
    ↓
liboci.so/dylib (Go shared library)
    ↓
go-containerregistry (Go library)
    ↓
OCI Registry (Docker Hub, GHCR, etc.)
```

## Building

### Prerequisites

- Go 1.24 or later
- C compiler (for CGO)
- Docker CLI (for authentication)

### Build Process

The Go library is automatically built during `pip install` or `python setup.py build_ext`:

1. `setup.py` calls `cmake_build_ext.build_go_oci_client()`
2. This executes `vllm/model_executor/model_loader/oci_go/build.sh`
3. The build script:
   - Downloads Go dependencies
   - Builds `liboci.so` (Linux) or `liboci.dylib` (macOS)
   - Also builds `liboci.a` for static linking option

### Manual Build

```bash
cd vllm/model_executor/model_loader/oci_go
./build.sh
```

## Usage

### Basic Usage

```python
from vllm import LLM

# Load from Docker Hub (default registry)
llm = LLM(
    model="username/model:tag",
    load_format="oci"
)
```

### Private Registries

```bash
# Authenticate with Docker CLI
docker login ghcr.io
# or
docker login registry.example.com
```

```python
from vllm import LLM

# Load from private registry
llm = LLM(
    model="ghcr.io/myorg/private-model:latest",
    load_format="oci"
)
```

The Go library automatically reads credentials from `~/.docker/config.json`.

## Authentication

### Docker Config

The library reads authentication from:
- `~/.docker/config.json` (default location)
- `$DOCKER_CONFIG/config.json` (if `$DOCKER_CONFIG` is set)

### Credential Helpers

Supports Docker credential helpers:
- `docker-credential-ecr-login` (AWS ECR)
- `docker-credential-gcr` (Google Container Registry)
- `docker-credential-acr-env` (Azure Container Registry)
- Custom credential helpers

## Testing

### Standalone Tests

```bash
cd vllm/model_executor/model_loader/oci_go
python3 test_client.py
```

This will:
1. Test library loading
2. Test function signatures
3. Pull a manifest from Docker Hub (requires network)

### Unit Tests

```bash
pytest tests/model_executor/model_loader/test_oci_loader.py
```

## Deployment

### Static Linking

For deployments, you can use the static archive (`liboci.a`) to avoid runtime dependencies:

```python
# In oci_go_client.py, modify to load liboci.a
# This requires linking the Go runtime statically
```

### Cross-Platform

The library is built for:
- Linux x86_64, aarch64 (`.so`)
- macOS x86_64, arm64 (`.dylib`)

The Python wrapper automatically detects and loads the correct library.

## Troubleshooting

### Library Not Found

```
RuntimeError: OCI Go library not found
```

**Solution**: Build the Go library:
```bash
cd vllm/model_executor/model_loader/oci_go
./build.sh
```

### Authentication Errors

```
RuntimeError: Failed to pull manifest: ... authentication required
```

**Solution**: Run `docker login` for the registry:
```bash
docker login [registry]
```

### Network Errors

```
RuntimeError: Failed to pull manifest: ... connection refused
```

**Solution**: Check network connectivity and registry URL.

## Performance

### Caching

The OCI loader caches downloaded layers in:
```
~/.cache/vllm/oci/{normalized-reference}/
├── manifest.json
├── layers/
│   └── *.safetensors
└── config/
```

Subsequent loads use cached layers, avoiding re-downloads.

### Memory

The Go library uses Go's garbage collector for memory management. Note on CGO memory:
- Strings returned from Go functions (via `C.CString`) remain valid for the lifetime of the program
- The current implementation does not explicitly free these strings, which is acceptable for long-lived processes
- For high-frequency operations, consider implementing a proper free mechanism
- Memory allocated on the Go side is managed by Go's GC and doesn't require Python intervention

## Security Considerations

1. **Credential Storage**: 
   - Docker config files (`~/.docker/config.json`) may contain plaintext credentials
   - **Recommended**: Use Docker credential helpers instead:
     - AWS ECR: `docker-credential-ecr-login`
     - Google GCR: `docker-credential-gcr`
     - Azure ACR: `docker-credential-acr-env`
   - Ensure Docker config file has restrictive permissions: `chmod 600 ~/.docker/config.json`
   - Consider using keychain-based credential storage on desktop systems
2. **TLS**: All registry communication uses HTTPS/TLS by default.
3. **Digest Verification**: OCI digests (SHA256) are verified during download.
4. **Private Images**: Always authenticate before pulling private images to avoid exposing registry URLs

## Future Improvements

Potential enhancements:
1. Support for OCI image index (multi-platform manifests)
2. Parallel layer downloads
3. Progress reporting during downloads
4. Registry mirror support
5. Offline mode with local cache

## References

- [go-containerregistry](https://github.com/google/go-containerregistry)
- [OCI Distribution Spec](https://github.com/opencontainers/distribution-spec)
- [Docker Config](https://docs.docker.com/engine/reference/commandline/cli/#configuration-files)
- [Credential Helpers](https://github.com/docker/docker-credential-helpers)
