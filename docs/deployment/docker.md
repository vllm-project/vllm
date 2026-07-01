---
toc_depth: 2
---

# Using Docker

## Pre-built images

--8<-- "docs/getting_started/installation/gpu.md:pre-built-images"

## Run as a non-root user

The CUDA `vllm/vllm-openai` image runs as root by default for backward
compatibility. It is also prepared to run as the built-in `vllm` user
(UID 2000, GID 0):

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

When mounting model or cache volumes for a non-root container, mount writable
paths under `/home/vllm` instead of `/root`. For example, mount the Hugging
Face cache at `/home/vllm/.cache/huggingface` and make the mounted directory
writable by group 0.

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -v ~/.cache/huggingface:/home/vllm/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

To build an image that defaults to the non-root `vllm` user, use the opt-in
`vllm-openai-nonroot` target:

```bash
docker build --target vllm-openai-nonroot \
    -t vllm-openai-nonroot:local \
    -f docker/Dockerfile .

docker run --rm --gpus all \
    -p 8000:8000 \
    vllm-openai-nonroot:local \
    meta-llama/Llama-3.1-8B-Instruct
```

The `vllm-openai-nonroot` target also supports OpenShift-style arbitrary UIDs
when the runtime UID is a member of group 0. In Kubernetes manifests, set the
container security context accordingly and keep mounted cache/model paths
writable by group 0:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000540000
  runAsGroup: 0
  fsGroup: 0
```

Runtime UIDs outside group 0 are not part of the documented support matrix
because they may be unable to write to `/home/vllm` or `/opt/uv/cache`.

## Build image from source

--8<-- "docs/getting_started/installation/gpu.md:build-image-from-source"
