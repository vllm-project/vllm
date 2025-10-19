# OCI Registry Support

vLLM supports loading models directly from OCI (Open Container Initiative) registries, enabling efficient distribution and deployment of large language models through container registries.

## Overview

OCI Registry support allows you to:

- Store and distribute models as OCI artifacts in container registries
- Pull models directly from Docker Hub, GitHub Container Registry (ghcr.io), or any OCI-compliant registry
- Leverage existing registry infrastructure for model distribution
- Benefit from registry features like versioning, access control, and content-addressable storage

## OCI Model Format

Models are stored as OCI artifacts with the following structure:

### Layers

1. **Safetensors Layers** (`application/vnd.docker.ai.safetensors`)
   - Contains model weights in safetensors format
   - Multiple layers are supported for sharded models
   - Layer order in the manifest defines shard order
   - Each layer is a single safetensors file

2. **Config Tar Layer** (`application/vnd.docker.ai.vllm.config.tar`)
   - Contains model configuration files (tokenizer config, vocab files, etc.)
   - Packaged as a tar archive
   - Automatically extracted after downloading
   - Optional but recommended for complete model functionality

3. **Additional Layers** (optional)
   - License files (`application/vnd.docker.ai.license`)
   - Model cards, documentation, etc.

### Example Manifest

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {
    "mediaType": "application/vnd.docker.ai.model.config.v0.1+json",
    "digest": "sha256:...",
    "size": 465
  },
  "layers": [
    {
      "mediaType": "application/vnd.docker.ai.safetensors",
      "digest": "sha256:...",
      "size": 3968658944
    },
    {
      "mediaType": "application/vnd.docker.ai.safetensors",
      "digest": "sha256:...",
      "size": 2203268048
    },
    {
      "mediaType": "application/vnd.docker.ai.vllm.config.tar",
      "digest": "sha256:...",
      "size": 11530752
    }
  ]
}
```

## Usage

### Basic Usage

```python
from vllm import LLM

# Load model from Docker Hub (default registry)
llm = LLM(
    model="username/modelname:tag",
    load_format="oci"
)
```

### Model Reference Format

The model reference follows the standard OCI reference format:

```bash
[registry/]repository[:tag|@digest]
```

**Examples:**

- `username/model:tag` → `docker.io/username/model:tag` (default registry)
- `docker.io/username/model:v1.0` → explicit Docker Hub reference
- `ghcr.io/org/model:latest` → GitHub Container Registry
- `username/model@sha256:abc123...` → reference by digest (immutable)

### Advanced Usage

#### Specify Custom Download Directory

```python
from vllm import LLM

llm = LLM(
    model="username/model:tag",
    load_format="oci",
    download_dir="/path/to/cache"
)
```

#### Using Different Registries

```python
# GitHub Container Registry
llm = LLM(
    model="ghcr.io/organization/model:v1.0",
    load_format="oci"
)

# Google Container Registry
llm = LLM(
    model="gcr.io/project/model:latest",
    load_format="oci"
)

# Azure Container Registry
llm = LLM(
    model="myregistry.azurecr.io/model:tag",
    load_format="oci"
)
```

#### Reference by Digest (Immutable)

For reproducible deployments, use digest references:

```python
llm = LLM(
    model="username/model@sha256:1234567890abcdef...",
    load_format="oci"
)
```

## Installation

OCI Registry support is built into vLLM and uses the [go-containerregistry](https://github.com/google/go-containerregistry) library for robust OCI operations. The Go library is statically linked into vLLM during installation.

### Requirements

- Go 1.24 or later (for building from source)
- Docker CLI (for authentication via `docker login`)

No additional dependencies are required beyond the standard vLLM installation.

## Authentication

OCI Registry support includes full authentication capabilities using Docker config:

### Using Docker Login

For private registries, authenticate using the standard Docker CLI:

```bash
# Docker Hub
docker login

# GitHub Container Registry
docker login ghcr.io

# Google Container Registry
docker login gcr.io

# Azure Container Registry
docker login myregistry.azurecr.io

# Custom registry
docker login registry.example.com
```

vLLM will automatically use the credentials stored in `~/.docker/config.json` when pulling images.

### Credential Helpers

The go-containerregistry library supports Docker credential helpers, allowing integration with:
- Cloud provider credential helpers (AWS ECR, GCR, ACR)
- Password managers
- Custom credential stores

## Caching

### Cache Location

Downloaded OCI layers are cached in:

```bash
~/.cache/vllm/oci/{normalized-reference}/
├── manifest.json
├── layers/
│   ├── 0000_<digest>.safetensors
│   ├── 0001_<digest>.safetensors
│   └── ...
└── config/
    ├── config.json
    ├── tokenizer_config.json
    └── ...
```

### Cache Behavior

- Layers are downloaded only once and reused across multiple loads
- Layer downloads check for existing files before downloading
- Each model reference has a separate cache directory
- Cache is shared across all vLLM instances on the same machine
